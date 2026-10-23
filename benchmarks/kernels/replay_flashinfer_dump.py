# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay one trusted FlashInfer API capture on one GPU in the same container.

Pass the call-NN directory. NCU --profile-from-start off profiles only measured
API calls, excluding loading, state restoration, warmup and output comparison.
Each iteration reloads the original inputs, including in-place buffers.
Stateful wrappers and distributed operations require their original runtime and
are captured for inspection, not supported by this standalone replay.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

import torch


def load_snapshot(path: Path, device: torch.device) -> object:
    """Preserve host tensors and map captured CUDA tensors to the replay GPU."""
    return torch.load(
        path,
        weights_only=False,
        map_location=lambda storage, location: storage.cuda(device.index or 0)
        if location.startswith("cuda")
        else storage,
    )


def compare_snapshot(
    actual: object, expected: object, *, rtol: float, atol: float
) -> None:
    """Compare tensors and API parameters, including dtype/device/enum values."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True, check_device=False
        )
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            compare_snapshot(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_snapshot(left, right, rtol=rtol, atol=atol)
    else:
        assert actual == expected, f"Parameter mismatch: {actual!r} != {expected!r}"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--compare-buffers",
        action="store_true",
        help="Also compare argument buffers, including scratch space",
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--module", type=str, required=False, default=None)
    parser.add_argument("--function", type=str, required=False, default=None)
    parser.add_argument(
        "--weight-layout",
        type=str,
        required=False,
        default=None,
        choices=["BlockMajorK", "MajorK"],
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    metadata = json.loads((args.dump / "metadata.json").read_text())
    if metadata["status"] != "complete" or not metadata["standalone_replay"]:
        parser.error(
            "This call is incomplete or requires wrapper/distributed state: "
            f"{metadata.get('unsupported_types', [])}"
        )
    if metadata["torch_version"] != str(torch.__version__):
        parser.error("Use the same PyTorch/container version as capture")

    os.environ.pop("VLLM_FLASHINFER_DUMP_DIR", None)
    os.environ["FLASHINFER_LOGLEVEL"] = "0"
    device = torch.device(args.device)
    torch.accelerator.set_device_index(device)

    if args.module and args.function:
        module_str = args.module
        function_str = args.function
    elif not args.module and not args.function:
        module_str = metadata["module"]
        function_str = metadata["function"]
    else:
        raise NotImplementedError(f"{args.module=} {args.function=}")

    function = importlib.import_module(module_str)
    for part in function_str.split("."):
        function = getattr(function, part)

    print(f"{module_str}.{function_str}: forward_tokens={metadata['forward_tokens']}")
    for iteration in range(args.warmup + args.iterations):
        positional, keyword = load_snapshot(args.dump / "inputs.pt", device)
        if args.weight_layout:
            from flashinfer.fused_moe import WeightLayout

            keyword["weight_layout"] = getattr(WeightLayout, args.weight_layout)
        if iteration == 0:
            print(
                [(i, type(p)) for i, p in enumerate(positional)]
                + [(k, type(v)) for k, v in keyword.items()]
            )

        torch.accelerator.synchronize()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        measured = iteration >= args.warmup
        if measured:
            torch.cuda.profiler.start()
        try:
            start.record()
            result = function(*positional, **keyword)
            end.record()
            end.synchronize()
        finally:
            if measured:
                torch.cuda.profiler.stop()
        if measured:
            print(
                f"iteration={iteration - args.warmup} "
                f"gpu_ms={start.elapsed_time(end):.3f}"
            )
            expected = torch.load(
                args.dump / "outputs.pt", map_location="cpu", weights_only=False
            )
            compare_snapshot(result, expected, rtol=args.rtol, atol=args.atol)
            if args.compare_buffers:
                after = torch.load(
                    args.dump / "arguments_after.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                compare_snapshot(
                    (positional, keyword), after, rtol=args.rtol, atol=args.atol
                )
            if result is None and not args.compare_buffers:
                print(
                    "API returns None; inspect arguments_after.pt "
                    "or use --compare-buffers."
                )
            else:
                print("Saved outputs match within the requested tolerances.")
        del result, positional, keyword


if __name__ == "__main__":
    main()
