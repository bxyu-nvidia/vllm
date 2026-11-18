# vllm-frontend-rs

This is a Rust drop-in alternative frontend for vLLM. The current goal is to rebuild the northbound serving layer in Rust while still talking to the core Python vLLM engine process(es) via ZMQ over the existing engine boundary.

It should still be considered experimental, and is not feature-complete. We are working to add more functionality from the python front-end.

See <https://github.com/Inferact/vllm-frontend-rs> for the original commit history before it was moved into the main vllm repo.

## Architecture

The component is organized as a Cargo workspace with several crates, layered bottom-up:

```text
┌─────────────────────────────────┐
│  vllm-cmd / vllm-rs             │  CLI entrypoint:
│                                 │  Python vLLM frontend subprocess
│                                 │  Rust managed-engine serve mode
│                                 │  Engine-free render mode
├─────────────────────────────────┤
│  vllm-server                    │  OpenAI-compatible HTTP API (axum)
├─────────────────────────────────┤
│  vllm-chat                      │  Chat completions: template rendering,
│                                 │  structured assistant events,
│                                 │  reasoning & tool parsing
├─────────────────────────────────┤
│  vllm-text                      │  Tokenizer & incremental detokenizer
├─────────────────────────────────┤
│  vllm-llm                       │  Thin token-in/token-out facade over
│                                 │  the engine client
├─────────────────────────────────┤
│  vllm-engine-core-client        │  ZMQ transport + MessagePack protocol
│                                 │  for the headless vLLM engine
└─────────────────────────────────┘
```

`vllm-rs` integrates into Python `vllm` as a Rust frontend subprocess.
Python owns process startup and launches the Rust API server as a Python-supervised worker, while
passing the inherited listening socket and transport addresses into `vllm-rs`.

For example:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

To also expose the Rust frontend's gRPC inference and control services, add
`--grpc-port`. The HTTP listener remains on `--port`:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --grpc-port 50051
```

The gRPC listener binds on `--host` like HTTP, or on `127.0.0.1` when HTTP uses
`--uds`. When HTTP uses TCP, its port and the gRPC port must differ unless
requesting an automatically assigned port with `0`.

This also works with Python-supervised hybrid data parallelism. It requires a
frontend process and cannot be combined with `--headless` or
`--data-parallel-multi-port-external-lb`.

The separate `vllm serve MODEL --grpc --port 50051` command launches the Python
SMG `VllmEngine` gRPC server in place of HTTP. The Rust listener exposes the
`vllm.Inference` and `vllm.Control` services alongside HTTP. These are different
gRPC APIs; `--grpc` and `--grpc-port` are mutually exclusive.

### RL weight synchronization

With `VLLM_SERVER_DEV_MODE=1`, the Rust frontend supports the HTTP weight-transfer
lifecycle used by `HTTPVLLMWeightSyncClient`: initialization, starting an update,
transferring weights, and finishing the update. It also supports draft-model
updates and `/update_weight_version` and `/weight_info` for version tracking.

For trainer-side usage, see the [IPC](../examples/rl/rlhf_http_ipc.py) and
[NCCL](../examples/rl/rlhf_http_nccl.py) examples. Configure the weight-transfer
backend on the server and pause generation during updates, as shown there.

### Exact token prefixes for multi-turn training

Enable exact-prefix alignment on the Rust frontend with:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve MODEL --enable-prefix-token-alignment
```

Send `required_prefix_token_ids` on `/v1/chat/completions` containing the exact
previous prompt plus generated tokens, through the last assistant turn in
`messages`. The frontend renders the updated messages and replaces that
historical portion with the supplied IDs. It identifies the boundary by counting
model/tokenizer EOS tokens, so history formatting and removal of old reasoning
do not change the tokens used for training. A generation that ended without EOS
gets the template's turn terminator appended before the new messages.

Use `return_token_ids: true` to receive the aligned `prompt_token_ids` and
`choices[].token_ids`. Use `logprobs: true`, `top_logprobs`, and
`return_tokens_as_token_ids: true` for token-ID-addressable logprobs. Keep
`include_reasoning: true` (the default) to retain metadata for the full generated
stream. These response fields work in both streaming and non-streaming chat.
The chat-shaped `/tokenize` endpoint also accepts `required_prefix_token_ids`.

Without the startup flag, supplying this field returns HTTP 400. Requests that
omit it retain normal rendering; an empty list requests no replacement. The
initial implementation supports text-only histories with EOS-delimited turns;
nonempty prefixes cannot be combined with multimodal inputs, prompt truncation,
or `continue_final_message`. Missing turn boundaries return HTTP 400. Context
and vocabulary validation use the final aligned tokens.

Managed PD capture also accepts a router-generated `nemo_rl_capture` object on
chat requests. Version 1 carries the capture `admission`, authoritative
`weight_version`, `phase` (`prefill` or `decode`), `top_logprobs`, and
`routed_experts`. The frontend forwards it through
`sampling_params.extra_args["nemo_rl_capture"]`; that key is reserved in HTTP
`vllm_xargs`.

For a child call, the frontend adds `alignment`: `suffix_start` indexes the
first continuation token in the final engine prompt, `turn_end_token_id` is the
template terminator, and `eos_token_ids` includes alternate model terminators.
Callers cannot supply this boundary. Python reads and validates the metadata
through `EngineCoreRequest.capture_context`. This is a transport handoff only:
parent prefixes must still be resolved into `required_prefix_token_ids`, and
the existing capture service still stages results. No TQ write or staging
acknowledgment is performed by this interface.

### External Engine

`vllm-rs serve` can be run standalone with `--data-parallel-size-local 0` when the Python engines
are started elsewhere and this node should run only the Rust frontend. The frontend still uses
the global `--data-parallel-size` to determine how many engines it expects to join the shared handshake.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1
```

Then start the Rust frontend-only server:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
```

To build the `vllm-rs` in isolation:

```bash
# from the local checkout
./tools/build_rust.sh
```

### Engine-free renderer

`vllm-rs render` serves text-only request preprocessing without starting or
connecting to a Python inference engine:

```bash
cargo run --manifest-path rust/Cargo.toml -p vllm-cmd --release -- \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768
```

It exposes `/v1/chat/completions/render` and `/v1/completions/render`. Only
tokenizer and model configuration files are loaded; model weights, PyTorch,
and vLLM kernels are not required.

To serve HTTPS, pass `--ssl-certfile`. If the certificate and private key are
in separate files, pass `--ssl-keyfile` for the key; otherwise the key is read from
the certificate file. For mTLS, pass `--ssl-ca-certs` with a CA
bundle and set `--ssl-cert-reqs` to `1` (optional) or `2` (required) to
verify client certificates. Use `--ssl-ciphers` to override the default
OpenSSL cipher list.

```bash
# Combined PEM (cert + key in one file):
vllm-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/combined.pem

# Separate cert and key files:
vllm-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

The render endpoints return the public token-in `GenerateRequest` consumed by
the Rust `/inference/v1/generate` endpoint. A chat render response, or one item
from a completion render response, can be submitted to that endpoint without
changing its fields.

The render and inference paths use the same `vllm-chat` and `vllm-text`
request-preparation logic; render mode stops before engine submission.
Tool-call and reasoning parsers use model-based auto-detection by default. Use
`--tool-call-parser` and `--reasoning-parser` to override either selection;
unified parsers require the same selection for both options.

```bash
curl http://127.0.0.1:8000/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 16
  }'
```

### Example Request

After either full-frontend startup path, you can use any OpenAI-compatible
client against the inference endpoints:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```
