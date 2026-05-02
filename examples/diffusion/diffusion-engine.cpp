#include "diffusion-engine.h"

#include "chat.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

enum diffusion_algorithm {
    ORIGIN           = 0,
    ENTROPY_BASED    = 1,
    MARGIN_BASED     = 2,
    RANDOM           = 3,
    CONFIDENCE_BASED = 4,
};

enum transfer_schedule {
    TIMESTEP_BASED = 0,
    BLOCK_BASED    = 1,
};

struct diffusion_params {
    int32_t     steps         = 0;
    float       temperature   = 0.0f;
    llama_token mask_token_id = LLAMA_TOKEN_NULL;
    int32_t     seed          = 0;
    bool        shift_logits  = false;

    float   top_p = 0.0f;
    int32_t top_k = 0;

    diffusion_algorithm algorithm = CONFIDENCE_BASED;
    transfer_schedule   schedule  = TIMESTEP_BASED;

    float   cfg_scale        = 0.0f;
    float   eps              = 0.0f;
    int32_t block_length     = 0;
    float   alg_temp         = 0.0f;
    bool    add_gumbel_noise = false;

    int32_t max_length = 0;
};

struct diffusion_timing {
    int64_t total_us          = 0;
    int64_t batch_us          = 0;
    int64_t mask_scan_us      = 0;
    int64_t decode_us         = 0;
    int64_t logits_us         = 0;
    int64_t cfg_us            = 0;
    int64_t sampling_us       = 0;
    int64_t sort_us           = 0;
    int64_t active_logits     = 0;
    int64_t requested_logits  = 0;
    int64_t skipped_logits    = 0;
    int64_t logits_copy_bytes = 0;
};

using diffusion_step_callback = std::function<bool(int32_t, int32_t, const llama_token *, int32_t)>;

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;
        case ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    const float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;
            }
        case MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;
        case RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);
            }
        case ORIGIN:
            return cur_p.data[cur_p.selected].p;
        default:
            return 0.0f;
    }
}

static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        transfer_schedule            schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case TIMESTEP_BASED:
            {
                const float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                const float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                const float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }
        case BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / std::max(1, total_steps - step);
        default:
            return remaining_masked / std::max(1, total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise = uniform(rng);
        noise = std::max(noise, 1e-20);
        const double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i] = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    const int32_t base      = mask_count / steps;
    const int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

static diffusion_timing diffusion_generate(llama_context *          ctx,
                                           const llama_token *      input_tokens,
                                           llama_token *            output_tokens,
                                           int32_t                  n_input,
                                           const diffusion_params & params,
                                           int32_t &                n_generated,
                                           const diffusion_step_callback & step_callback = nullptr) {
    n_generated = 0;
    diffusion_timing timing;

    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return timing;
    }

    const llama_model * model = llama_get_model(ctx);

    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);
    llama_set_embeddings(ctx, false);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);
    std::vector<int32_t> active_positions;
    active_positions.reserve(params.max_length);
    std::vector<int32_t> active_logit_rows;
    active_logit_rows.reserve(params.max_length);
    std::vector<int32_t> row_to_compact(params.max_length, -1);
    std::vector<float *> row_logits;
    row_logits.reserve(params.max_length);
    std::vector<float *> logits_by_pos(params.max_length, nullptr);

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        un_x_buffer.resize(params.max_length);
    }

    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == BLOCK_BASED) {
        const int32_t generated_length = std::max(0, params.max_length - n_input);
        num_blocks = std::max(1, (generated_length + params.block_length - 1) / params.block_length);
        steps_per_block = std::max(1, (params.steps + num_blocks - 1) / num_blocks);
    }
    const int32_t total_steps = steps_per_block * num_blocks;

    const int64_t time_start = ggml_time_us();

    bool stop_requested = false;
    bool failed         = false;

    for (int block_num = 0; block_num < num_blocks && !stop_requested && !failed; block_num++) {
        const int32_t block_start = (params.schedule == BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        const int32_t block_end   = (params.schedule == BLOCK_BASED) ?
                                        std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                        params.max_length;

        if (params.schedule == BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block && !stop_requested && !failed; step++) {
            const int32_t global_step = block_num * steps_per_block + step;

            const int64_t time_start_mask_scan = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    if (params.schedule != BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                timing.mask_scan_us += ggml_time_us() - time_start_mask_scan;
                break;
            }

            active_positions.clear();
            if (params.algorithm == ORIGIN) {
                const int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                const float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        active_positions.push_back(pos);
                    }
                }
            } else {
                active_positions = mask_positions;
            }

            auto logit_row_for_pos = [&](int32_t pos) -> int32_t {
                return params.shift_logits && pos > 0 ? pos - 1 : pos;
            };

            std::fill(row_to_compact.begin(), row_to_compact.end(), -1);
            active_logit_rows.clear();
            for (int32_t pos : active_positions) {
                const int32_t row = logit_row_for_pos(pos);
                if (row < 0 || row >= params.max_length) {
                    continue;
                }
                if (row_to_compact[row] < 0) {
                    row_to_compact[row] = (int32_t) active_logit_rows.size();
                    active_logit_rows.push_back(row);
                }
            }

            timing.mask_scan_us += ggml_time_us() - time_start_mask_scan;

            auto emit_step = [&]() {
                if (step_callback && global_step + 1 < total_steps &&
                    !step_callback(global_step + 1, total_steps, output_tokens, params.max_length)) {
                    stop_requested = true;
                }
            };

            if (active_positions.empty() || active_logit_rows.empty()) {
                emit_step();
                continue;
            }

            const int64_t time_start_batch = ggml_time_us();
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = output_tokens[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 0;
            }
            for (int32_t row : active_logit_rows) {
                batch.logits[row] = 1;
            }
            timing.batch_us += ggml_time_us() - time_start_batch;

            const int32_t decode_multiplier = params.cfg_scale > 0.0f ? 2 : 1;
            timing.active_logits += (int64_t) active_logit_rows.size();
            timing.requested_logits += (int64_t) active_logit_rows.size() * decode_multiplier;
            timing.skipped_logits += (int64_t) (params.max_length - (int32_t) active_logit_rows.size()) * decode_multiplier;
            timing.logits_copy_bytes += (int64_t) active_logit_rows.size() * decode_multiplier * n_vocab * (int64_t) sizeof(float);

            if (params.cfg_scale > 0.0f) {
                const int64_t time_start_decode = ggml_time_us();
                int ret = llama_decode(ctx, batch);
                timing.decode_us += ggml_time_us() - time_start_decode;
                if (ret != 0) {
                    failed = true;
                    break;
                }

                cond_logits_buffer.resize((size_t) active_logit_rows.size() * n_vocab);
                row_logits.assign(active_logit_rows.size(), nullptr);

                const int64_t time_start_logits = ggml_time_us();
                for (size_t i = 0; i < active_logit_rows.size(); i++) {
                    float * cond_logits = llama_get_logits_ith(ctx, active_logit_rows[i]);
                    if (!cond_logits) {
                        break;
                    }
                    float * dst = cond_logits_buffer.data() + i * n_vocab;
                    std::memcpy(dst, cond_logits, (size_t) n_vocab * sizeof(float));
                    row_logits[i] = dst;
                }
                timing.logits_us += ggml_time_us() - time_start_logits;

                if (std::any_of(row_logits.begin(), row_logits.end(), [](const float * logits) { return logits == nullptr; })) {
                    failed = true;
                    break;
                }

                const int64_t time_start_cfg = ggml_time_us();
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                timing.cfg_us += ggml_time_us() - time_start_cfg;

                const int64_t time_start_uncond_decode = ggml_time_us();
                ret = llama_decode(ctx, batch);
                timing.decode_us += ggml_time_us() - time_start_uncond_decode;
                if (ret != 0) {
                    failed = true;
                    break;
                }

                for (size_t i = 0; i < active_logit_rows.size(); i++) {
                    const int64_t time_start_uncond_logits = ggml_time_us();
                    float * uncond_logits = llama_get_logits_ith(ctx, active_logit_rows[i]);
                    timing.logits_us += ggml_time_us() - time_start_uncond_logits;
                    if (!uncond_logits) {
                        row_logits[i] = nullptr;
                        break;
                    }

                    const int64_t time_start_cfg_row = ggml_time_us();
                    float * cond_logits = cond_logits_buffer.data() + i * n_vocab;
                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        cond_logits[token_id] =
                            uncond_logits[token_id] + (params.cfg_scale + 1.0f) * (cond_logits[token_id] - uncond_logits[token_id]);
                    }
                    timing.cfg_us += ggml_time_us() - time_start_cfg_row;
                }

                if (std::any_of(row_logits.begin(), row_logits.end(), [](const float * logits) { return logits == nullptr; })) {
                    failed = true;
                    break;
                }
            } else {
                const int64_t time_start_decode = ggml_time_us();
                const int ret = llama_decode(ctx, batch);
                timing.decode_us += ggml_time_us() - time_start_decode;
                if (ret != 0) {
                    failed = true;
                    break;
                }

                row_logits.assign(active_logit_rows.size(), nullptr);
                const int64_t time_start_logits = ggml_time_us();
                for (size_t i = 0; i < active_logit_rows.size(); i++) {
                    row_logits[i] = llama_get_logits_ith(ctx, active_logit_rows[i]);
                }
                timing.logits_us += ggml_time_us() - time_start_logits;

                if (std::any_of(row_logits.begin(), row_logits.end(), [](const float * logits) { return logits == nullptr; })) {
                    failed = true;
                    break;
                }
            }

            std::fill(logits_by_pos.begin(), logits_by_pos.end(), nullptr);
            for (int32_t pos : active_positions) {
                const int32_t row = logit_row_for_pos(pos);
                const int32_t compact_idx = row >= 0 && row < params.max_length ? row_to_compact[row] : -1;
                logits_by_pos[pos] = compact_idx >= 0 ? row_logits[compact_idx] : nullptr;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                const int64_t time_start_gumbel = ggml_time_us();
                for (float * pos_logits : row_logits) {
                    add_gumbel_noise(pos_logits, n_vocab, params.temperature, rng);
                }
                timing.sampling_us += ggml_time_us() - time_start_gumbel;
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                return logits_by_pos[pos];
            };

            const int64_t time_start_sampling = ggml_time_us();

            if (params.algorithm == ORIGIN) {
                for (int32_t pos : active_positions) {
                    const float * pos_logits = get_logits_for_pos(pos);
                    if (!pos_logits) {
                        failed = true;
                        break;
                    }
                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].id    = token_id;
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        (size_t) n_vocab,
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    output_tokens[pos] = cur_p.data[cur_p.selected].id;
                }
            } else {
                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    const int32_t pos = mask_positions[i];
                    const float * pos_logits = get_logits_for_pos(pos);
                    if (!pos_logits) {
                        failed = true;
                        break;
                    }

                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                        candidates[token_id].id    = token_id;
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    const llama_token sampled_token = cur_p.data[cur_p.selected].id;
                    const float       conf          = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, i);
                }

                if (failed) {
                    break;
                }

                const int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                if (transfer_count > 0) {
                    timing.sampling_us += ggml_time_us() - time_start_sampling;
                    const int64_t time_start_sort = ggml_time_us();
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                          confidences.end(),
                                          [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            const int32_t mask_idx = confidences[i].second;
                            const int32_t pos      = mask_positions[mask_idx];
                            output_tokens[pos]     = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            const float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            const int32_t selected_idx = conf_array.selected;
                            const int32_t mask_idx     = selected_idx;
                            const int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]         = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                    timing.sort_us += ggml_time_us() - time_start_sort;
                } else {
                    timing.sampling_us += ggml_time_us() - time_start_sampling;
                }
            }

            if (failed) {
                break;
            }

            if (params.algorithm == ORIGIN) {
                timing.sampling_us += ggml_time_us() - time_start_sampling;
            }

            emit_step();
        }
    }

    timing.total_us = ggml_time_us() - time_start;

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = failed ? 0 : params.max_length;
    return timing;
}

static std::string format_input_text(const std::string & prompt,
                                     const std::string & system_prompt,
                                     bool                use_chat_template,
                                     llama_model *       model) {
    if (!use_chat_template) {
        return prompt;
    }

    auto chat_templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;

    if (!system_prompt.empty()) {
        common_chat_msg system_msg;
        system_msg.role    = "system";
        system_msg.content = system_prompt;
        inputs.messages.push_back(system_msg);
    }

    common_chat_msg user_msg;
    user_msg.role    = "user";
    user_msg.content = prompt;
    inputs.messages.push_back(user_msg);
    inputs.add_generation_prompt = true;

    auto result = common_chat_templates_apply(chat_templates.get(), inputs);
    return result.prompt;
}

static int32_t round_up_to_multiple(int32_t value, int32_t multiple) {
    if (multiple <= 0) {
        return value;
    }
    const int32_t remainder = value % multiple;
    return remainder == 0 ? value : value + multiple - remainder;
}

static std::string detokenize_generated_snapshot(const llama_vocab * vocab,
                                                 const llama_token * tokens,
                                                 int32_t             n_input,
                                                 int32_t             n_generated,
                                                 int32_t             n_predict,
                                                 llama_token         mask_token_id) {
    const int32_t generated_tokens = std::max(0, std::min(n_predict, n_generated - n_input));
    std::string   text;
    text.reserve(generated_tokens * 4);

    for (int32_t i = 0; i < generated_tokens; i++) {
        const llama_token token = tokens[n_input + i];
        if (token == mask_token_id) {
            text += ' ';
            continue;
        }
        text += common_token_to_piece(vocab, token, false);
    }

    return text;
}

} // namespace

diffusion_engine::~diffusion_engine() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

bool diffusion_engine::load(common_params params, std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (model_ || ctx_) {
        error = "diffusion engine is already loaded";
        return false;
    }

    defaults_ = params;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = params.n_gpu_layers;
    model_params.devices            = params.devices.data();
    model_params.use_mmap           = params.use_mmap;
    model_params.use_direct_io      = params.use_direct_io;
    model_params.use_mlock          = params.use_mlock;
    model_params.check_tensors      = params.check_tensors;

    model_ = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model_) {
        error = "failed to load model '" + params.model.path + "'";
        return false;
    }

    if (!llama_model_is_diffusion(model_)) {
        error = "model is not a diffusion model: '" + params.model.path + "'";
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = params.n_ctx;
    ctx_params.n_batch              = params.n_batch;
    ctx_params.n_ubatch             = params.n_ubatch;
    ctx_params.flash_attn_type      = params.flash_attn_type;
    ctx_params.no_perf              = params.no_perf;
    ctx_params.type_k               = params.cache_type_k;
    ctx_params.type_v               = params.cache_type_v;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        error = "failed to create context";
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    llama_set_n_threads(ctx_, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    vocab_ = llama_model_get_vocab(model_);
    return true;
}

bool diffusion_engine::loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return model_ != nullptr && ctx_ != nullptr;
}

diffusion_generation_request diffusion_engine::make_request_from_defaults() const {
    std::lock_guard<std::mutex> lock(mutex_);

    diffusion_generation_request request;
    request.n_predict        = defaults_.n_predict > 0 ? defaults_.n_predict : 128;
    request.steps            = defaults_.diffusion.steps;
    request.temperature      = defaults_.sampling.temp;
    request.seed             = defaults_.sampling.seed;
    request.top_p            = defaults_.sampling.top_p;
    request.top_k            = defaults_.sampling.top_k;
    request.eps              = defaults_.diffusion.eps;
    request.block_length     = defaults_.diffusion.block_length;
    request.algorithm        = defaults_.diffusion.algorithm;
    request.alg_temp         = defaults_.diffusion.alg_temp;
    request.cfg_scale        = defaults_.diffusion.cfg_scale;
    request.add_gumbel_noise = defaults_.diffusion.add_gumbel_noise;
    return request;
}

diffusion_generation_result diffusion_engine::generate(const diffusion_generation_request & request) {
    std::lock_guard<std::mutex> lock(mutex_);
    return generate_impl(request, nullptr);
}

diffusion_generation_result diffusion_engine::generate_stream(
    const diffusion_generation_request & request,
    diffusion_stream_callback            callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    return generate_impl(request, &callback);
}

diffusion_generation_result diffusion_engine::generate_impl(
    const diffusion_generation_request & request,
    const diffusion_stream_callback *    callback) {
    if (!model_ || !ctx_ || !vocab_) {
        throw std::runtime_error("diffusion engine is not loaded");
    }
    if (request.prompt.empty()) {
        throw std::invalid_argument("prompt is required");
    }
    if (request.n_predict <= 0) {
        throw std::invalid_argument("n_tokens must be greater than zero");
    }
    if (request.steps <= 0) {
        throw std::invalid_argument("steps must be greater than zero");
    }
    if (request.temperature < 0.0f) {
        throw std::invalid_argument("temperature must be non-negative");
    }
    if (request.algorithm < 0 || request.algorithm > 4) {
        throw std::invalid_argument("diffusion algorithm must be between 0 and 4");
    }

    const bool has_eps          = request.eps > 0.0f;
    const bool has_block_length = request.block_length > 0;
    if (has_eps == has_block_length) {
        throw std::invalid_argument("exactly one of diffusion eps or diffusion block length must be set");
    }

    std::string formatted_prompt =
        format_input_text(request.prompt, request.system_prompt, request.use_chat_template, model_);

    std::vector<llama_token> input_tokens = common_tokenize(
        vocab_, formatted_prompt, /* add special tokens */ true, /* parse special */ true);
    const int32_t n_input = (int32_t) input_tokens.size();

    if (n_input <= 0) {
        throw std::invalid_argument("prompt did not tokenize into any tokens");
    }
    if ((uint32_t) n_input >= llama_n_ctx(ctx_)) {
        throw std::invalid_argument("input is longer than the worker context");
    }

    int32_t max_length = n_input + request.n_predict;
    int32_t steps      = request.steps;
    if (has_block_length) {
        const int32_t generated_length = round_up_to_multiple(request.n_predict, request.block_length);
        max_length = n_input + generated_length;
        const int32_t num_blocks = std::max(1, generated_length / request.block_length);
        const int32_t steps_per_block = std::max(1, (steps + num_blocks - 1) / num_blocks);
        steps = steps_per_block * num_blocks;
    }

    if ((uint32_t) max_length > llama_n_ctx(ctx_)) {
        throw std::invalid_argument("requested prompt plus n_tokens exceeds the worker context size");
    }
    if ((uint32_t) max_length > llama_n_ubatch(ctx_)) {
        throw std::invalid_argument("requested prompt plus n_tokens exceeds the worker --ubatch-size");
    }

    const llama_token mask_token_id = llama_vocab_mask(vocab_);
    if (mask_token_id == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("model does not define a mask token");
    }

    char shift_logits_str[8];
    bool shift_logits = true;
    if (llama_model_meta_val_str(model_, "diffusion.shift_logits", shift_logits_str, sizeof(shift_logits_str)) >= 0) {
        shift_logits = strcmp(shift_logits_str, "true") == 0;
    }

    diffusion_params diff_params;
    diff_params.mask_token_id    = mask_token_id;
    diff_params.seed             = request.seed;
    diff_params.temperature      = request.temperature;
    diff_params.steps            = steps;
    diff_params.algorithm        = static_cast<diffusion_algorithm>(request.algorithm);
    diff_params.max_length       = max_length;
    diff_params.top_p            = request.top_p;
    diff_params.top_k            = request.top_k;
    diff_params.shift_logits     = shift_logits;
    diff_params.alg_temp         = request.alg_temp;
    diff_params.cfg_scale        = request.cfg_scale;
    diff_params.add_gumbel_noise = request.add_gumbel_noise;

    if (has_eps) {
        diff_params.schedule = TIMESTEP_BASED;
        diff_params.eps      = request.eps;
    } else {
        diff_params.schedule     = BLOCK_BASED;
        diff_params.block_length = request.block_length;
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    int32_t n_generated = 0;
    std::vector<llama_token> output_tokens(max_length);
    bool stream_cancelled = false;

    diffusion_step_callback step_callback;
    if (callback) {
        step_callback = [&](int32_t step, int32_t total_steps, const llama_token * tokens, int32_t n_tokens) {
            diffusion_stream_chunk chunk;
            chunk.text = detokenize_generated_snapshot(
                vocab_, tokens, n_input, n_tokens, request.n_predict, mask_token_id);
            chunk.step             = step;
            chunk.total_steps      = total_steps;
            chunk.prompt_tokens    = n_input;
            chunk.generated_tokens = std::min(request.n_predict, n_tokens - n_input);
            chunk.sequence_length  = n_tokens;
            chunk.done             = false;

            const bool keep_going = (*callback)(chunk);
            if (!keep_going) {
                stream_cancelled = true;
            }
            return keep_going;
        };
    }

    diffusion_timing timing = diffusion_generate(
        ctx_, input_tokens.data(), output_tokens.data(), n_input, diff_params, n_generated, step_callback);

    if (n_generated <= n_input) {
        throw std::runtime_error("diffusion generation failed");
    }

    const int32_t generated_tokens = std::min(request.n_predict, n_generated - n_input);
    diffusion_generation_result result;
    result.text             = detokenize_generated_snapshot(
        vocab_, output_tokens.data(), n_input, n_generated, request.n_predict, mask_token_id);
    result.prompt_tokens    = n_input;
    result.generated_tokens = generated_tokens;
    result.sequence_length  = max_length;
    result.steps_used       = steps;
    result.total_ms         = timing.total_us / 1000.0;
    result.batch_ms         = timing.batch_us / 1000.0;
    result.mask_scan_ms     = timing.mask_scan_us / 1000.0;
    result.decode_ms        = timing.decode_us / 1000.0;
    result.logits_ms        = timing.logits_us / 1000.0;
    result.cfg_ms           = timing.cfg_us / 1000.0;
    result.sampling_ms      = timing.sampling_us / 1000.0;
    result.sort_ms          = timing.sort_us / 1000.0;
    result.active_logits     = timing.active_logits;
    result.requested_logits  = timing.requested_logits;
    result.skipped_logits    = timing.skipped_logits;
    result.logits_copy_bytes = timing.logits_copy_bytes;

    if (callback && !stream_cancelled) {
        diffusion_stream_chunk final_chunk;
        final_chunk.text             = result.text;
        final_chunk.step             = steps;
        final_chunk.total_steps      = steps;
        final_chunk.prompt_tokens    = result.prompt_tokens;
        final_chunk.generated_tokens = result.generated_tokens;
        final_chunk.sequence_length  = result.sequence_length;
        final_chunk.done             = true;
        (*callback)(final_chunk);
    }

    return result;
}

std::string diffusion_engine::model_path() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return defaults_.model.path;
}

uint32_t diffusion_engine::context_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctx_ ? llama_n_ctx(ctx_) : 0;
}

uint32_t diffusion_engine::batch_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctx_ ? llama_n_batch(ctx_) : 0;
}

uint32_t diffusion_engine::ubatch_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctx_ ? llama_n_ubatch(ctx_) : 0;
}
