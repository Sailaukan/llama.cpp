#include "diffusion-engine.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <clocale>
#include <exception>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

static void set_cors(httplib::Response & res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "*");
}

static void set_json(httplib::Response & res, int status, const json & body) {
    set_cors(res);
    res.status = status;
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

static bool parse_custom_server_arg(int & i, int argc, char ** argv, common_params & params) {
    const std::string arg = argv[i];

    auto parse_value = [&](const std::string & name, std::string & value) -> bool {
        const std::string prefix = name + "=";
        if (arg == name) {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            value = argv[++i];
            return true;
        }
        if (arg.rfind(prefix, 0) == 0) {
            value = arg.substr(prefix.size());
            return true;
        }
        return false;
    };

    std::string value;
    if (parse_value("--host", value)) {
        params.hostname = value;
        return true;
    }
    if (parse_value("--port", value)) {
        params.port = std::stoi(value);
        return true;
    }
    return false;
}

static std::vector<char *> strip_server_only_args(int argc, char ** argv, common_params & params) {
    std::vector<std::string> kept;
    kept.reserve(argc);
    kept.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (parse_custom_server_arg(i, argc, argv, params)) {
            continue;
        }
        kept.emplace_back(argv[i]);
    }

    static std::vector<std::string> storage;
    storage = std::move(kept);

    std::vector<char *> filtered;
    filtered.reserve(storage.size());
    for (std::string & value : storage) {
        filtered.push_back(value.data());
    }
    return filtered;
}

static int json_int(const json & body, const char * key, int fallback) {
    if (!body.contains(key) || body[key].is_null()) {
        return fallback;
    }
    return body[key].get<int>();
}

static uint32_t json_uint(const json & body, const char * key, uint32_t fallback) {
    if (!body.contains(key) || body[key].is_null()) {
        return fallback;
    }
    return body[key].get<uint32_t>();
}

static float json_float(const json & body, const char * key, float fallback) {
    if (!body.contains(key) || body[key].is_null()) {
        return fallback;
    }
    return body[key].get<float>();
}

static bool json_bool(const json & body, const char * key, bool fallback) {
    if (!body.contains(key) || body[key].is_null()) {
        return fallback;
    }
    return body[key].get<bool>();
}

static std::string json_string(const json & body, const char * key, const std::string & fallback = "") {
    if (!body.contains(key) || body[key].is_null()) {
        return fallback;
    }
    return body[key].get<std::string>();
}

static void apply_model_flags(diffusion_generation_request & request, const json & body) {
    if (!body.contains("model_flags") || !body["model_flags"].is_array()) {
        return;
    }

    std::vector<std::string> flags = body["model_flags"].get<std::vector<std::string>>();
    for (size_t i = 0; i < flags.size(); i++) {
        const std::string & flag = flags[i];
        auto require_value = [&]() -> std::string {
            if (i + 1 >= flags.size()) {
                throw std::invalid_argument(flag + " requires a value");
            }
            return flags[++i];
        };

        if (flag == "--diffusion-eps") {
            request.eps = std::stof(require_value());
            request.block_length = 0;
        } else if (flag == "--diffusion-algorithm") {
            request.algorithm = std::stoi(require_value());
        } else if (flag == "--diffusion-alg-temp") {
            request.alg_temp = std::stof(require_value());
        } else if (flag == "--diffusion-block-length") {
            request.block_length = std::stoi(require_value());
            request.eps = 0.0f;
        } else if (flag == "--diffusion-cfg-scale") {
            request.cfg_scale = std::stof(require_value());
        } else if (flag == "--diffusion-add-gumbel-noise") {
            if (i + 1 < flags.size() && flags[i + 1].rfind("--", 0) != 0) {
                request.add_gumbel_noise = std::stof(flags[++i]) != 0.0f;
            } else {
                request.add_gumbel_noise = true;
            }
        }
    }
}

static diffusion_generation_request request_from_json(const diffusion_engine & engine, const json & body) {
    diffusion_generation_request request = engine.make_request_from_defaults();

    request.prompt            = json_string(body, "prompt");
    request.system_prompt     = json_string(body, "system_prompt");
    request.use_chat_template = json_bool(body, "use_chat_template", request.use_chat_template);

    request.n_predict   = json_int(body, "n_tokens", json_int(body, "max_tokens", request.n_predict));
    request.steps       = json_int(body, "steps", json_int(body, "diffusion_steps", request.steps));
    request.temperature = json_float(body, "temperature", request.temperature);
    request.seed        = json_uint(body, "seed", request.seed);
    request.top_p       = json_float(body, "top_p", request.top_p);
    request.top_k       = json_int(body, "top_k", request.top_k);

    request.eps              = json_float(body, "diffusion_eps", request.eps);
    request.block_length     = json_int(body, "diffusion_block_length", request.block_length);
    request.algorithm        = json_int(body, "diffusion_algorithm", request.algorithm);
    request.alg_temp         = json_float(body, "diffusion_alg_temp", request.alg_temp);
    request.cfg_scale        = json_float(body, "diffusion_cfg_scale", request.cfg_scale);
    request.add_gumbel_noise = json_bool(body, "diffusion_add_gumbel_noise", request.add_gumbel_noise);

    apply_model_flags(request, body);
    return request;
}

static json health_json(const diffusion_engine & engine) {
    return {
        { "ok", true },
        { "loaded", engine.loaded() },
        { "model_path", engine.model_path() },
        { "context_size", engine.context_size() },
        { "batch_size", engine.batch_size() },
        { "ubatch_size", engine.ubatch_size() },
    };
}

static json stream_chunk_json(const diffusion_stream_chunk & chunk) {
    return {
        { "step", chunk.step },
        { "total_steps", chunk.total_steps },
        { "text", chunk.text },
        { "prompt_tokens", chunk.prompt_tokens },
        { "generated_tokens", chunk.generated_tokens },
        { "sequence_length", chunk.sequence_length },
        { "done", chunk.done },
    };
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();
    common_init();

    common_params params;
    params.hostname = "127.0.0.1";
    params.port     = 8088;
    params.n_ubatch = 1024;

    std::vector<char *> filtered_argv;
    try {
        filtered_argv = strip_server_only_args(argc, argv, params);
    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("error: missing model path, pass -m /path/to/model.gguf\n");
        return 1;
    }

    llama_backend_init();

    bool ok = false;
    {
        diffusion_engine engine;
        std::string error;
        if (!engine.load(params, error)) {
            LOG_ERR("error: %s\n", error.c_str());
            llama_backend_free();
            return 1;
        }

        httplib::Server server;
        server.set_default_headers({ { "Server", "llama-diffusion-server" } });
        server.set_read_timeout(5, 0);
        server.set_write_timeout(300, 0);

        server.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
            if (req.method == "OPTIONS") {
                set_cors(res);
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
            set_json(res, 200, health_json(engine));
        });

        server.Post("/generate", [&](const httplib::Request & req, httplib::Response & res) {
            try {
                json body = json::parse(req.body);
                const std::string expected_model_path = json_string(body, "model_path");
                if (!expected_model_path.empty() && expected_model_path != engine.model_path()) {
                    set_json(res,
                             409,
                             {
                                 { "error", "worker has a different model loaded" },
                                 { "loaded_model_path", engine.model_path() },
                                 { "requested_model_path", expected_model_path },
                             });
                    return;
                }

                diffusion_generation_request generation_request = request_from_json(engine, body);
                diffusion_generation_result result = engine.generate(generation_request);

                set_json(res,
                         200,
                         {
                             { "text", result.text },
                             { "prompt_tokens", result.prompt_tokens },
                             { "generated_tokens", result.generated_tokens },
                             { "sequence_length", result.sequence_length },
                             { "steps_used", result.steps_used },
                             { "timings",
                               {
                                   { "total_ms", result.total_ms },
                                   { "batch_ms", result.batch_ms },
                                   { "mask_scan_ms", result.mask_scan_ms },
                                   { "decode_ms", result.decode_ms },
                                   { "logits_ms", result.logits_ms },
                                   { "cfg_ms", result.cfg_ms },
                                   { "sampling_ms", result.sampling_ms },
                                   { "sort_ms", result.sort_ms },
                                   { "active_logits", result.active_logits },
                                   { "requested_logits", result.requested_logits },
                                   { "skipped_logits", result.skipped_logits },
                                   { "logits_copy_bytes", result.logits_copy_bytes },
                               } },
                         });
            } catch (const nlohmann::json::exception & e) {
                set_json(res, 400, { { "error", std::string("invalid JSON request: ") + e.what() } });
            } catch (const std::invalid_argument & e) {
                set_json(res, 400, { { "error", e.what() } });
            } catch (const std::exception & e) {
                set_json(res, 500, { { "error", e.what() } });
            }
        });

        server.Post("/generate/stream", [&](const httplib::Request & req, httplib::Response & res) {
            try {
                json body = json::parse(req.body);
                const std::string expected_model_path = json_string(body, "model_path");
                if (!expected_model_path.empty() && expected_model_path != engine.model_path()) {
                    set_json(res,
                             409,
                             {
                                 { "error", "worker has a different model loaded" },
                                 { "loaded_model_path", engine.model_path() },
                                 { "requested_model_path", expected_model_path },
                             });
                    return;
                }

                diffusion_generation_request generation_request = request_from_json(engine, body);

                set_cors(res);
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");
                res.set_chunked_content_provider(
                    "application/x-ndjson",
                    [&engine, generation_request](size_t, httplib::DataSink & sink) {
                        try {
                            engine.generate_stream(generation_request, [&sink](const diffusion_stream_chunk & chunk) {
                                const std::string line = stream_chunk_json(chunk).dump() + "\n";
                                return sink.write(line.c_str(), line.size());
                            });
                        } catch (const std::exception & e) {
                            const std::string line = json{
                                { "error", e.what() },
                                { "done", true },
                            }.dump() + "\n";
                            sink.write(line.c_str(), line.size());
                        }

                        sink.done();
                        return true;
                    });
            } catch (const nlohmann::json::exception & e) {
                set_json(res, 400, { { "error", std::string("invalid JSON request: ") + e.what() } });
            } catch (const std::invalid_argument & e) {
                set_json(res, 400, { { "error", e.what() } });
            } catch (const std::exception & e) {
                set_json(res, 500, { { "error", e.what() } });
            }
        });

        LOG_INF("llama-diffusion-server listening on http://%s:%d\n", params.hostname.c_str(), params.port);
        ok = server.listen(params.hostname, params.port);
    }
    llama_backend_free();

    if (!ok) {
        LOG_ERR("error: failed to listen on %s:%d\n", params.hostname.c_str(), params.port);
        return 1;
    }

    return 0;
}
