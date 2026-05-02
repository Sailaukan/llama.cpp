#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

struct diffusion_generation_request {
    std::string prompt;
    std::string system_prompt;

    bool use_chat_template = false;

    int32_t n_predict   = 128;
    int32_t steps       = 128;
    float   temperature = 0.2f;
    uint32_t seed       = LLAMA_DEFAULT_SEED;

    float   top_p = 0.95f;
    int32_t top_k = 40;

    float   eps              = 0.0f;
    int32_t block_length     = 0;
    int32_t algorithm        = 4;
    float   alg_temp         = 0.0f;
    float   cfg_scale        = 0.0f;
    bool    add_gumbel_noise = false;
};

struct diffusion_generation_result {
    std::string text;
    int32_t     prompt_tokens   = 0;
    int32_t     generated_tokens = 0;
    int32_t     sequence_length  = 0;
    int32_t     steps_used       = 0;
    double      total_ms         = 0.0;
    double      batch_ms         = 0.0;
    double      mask_scan_ms     = 0.0;
    double      decode_ms        = 0.0;
    double      logits_ms        = 0.0;
    double      cfg_ms           = 0.0;
    double      sampling_ms      = 0.0;
    double      sort_ms          = 0.0;
    int64_t     active_logits     = 0;
    int64_t     requested_logits  = 0;
    int64_t     skipped_logits    = 0;
    int64_t     logits_copy_bytes = 0;
};

struct diffusion_stream_chunk {
    std::string text;
    int32_t     step             = 0;
    int32_t     total_steps      = 0;
    int32_t     prompt_tokens    = 0;
    int32_t     generated_tokens = 0;
    int32_t     sequence_length  = 0;
    bool        done             = false;
};

using diffusion_stream_callback = std::function<bool(const diffusion_stream_chunk &)>;

class diffusion_engine {
public:
    diffusion_engine() = default;
    ~diffusion_engine();

    diffusion_engine(const diffusion_engine &) = delete;
    diffusion_engine & operator=(const diffusion_engine &) = delete;

    bool load(common_params params, std::string & error);
    bool loaded() const;

    diffusion_generation_request make_request_from_defaults() const;
    diffusion_generation_result generate(const diffusion_generation_request & request);
    diffusion_generation_result generate_stream(
        const diffusion_generation_request & request,
        diffusion_stream_callback            callback);

    std::string model_path() const;
    uint32_t context_size() const;
    uint32_t batch_size() const;
    uint32_t ubatch_size() const;

private:
    common_params defaults_;

    llama_model *        model_ = nullptr;
    llama_context *      ctx_   = nullptr;
    const llama_vocab *  vocab_ = nullptr;
    mutable std::mutex   mutex_;

    diffusion_generation_result generate_impl(
        const diffusion_generation_request & request,
        const diffusion_stream_callback *    callback);
};
