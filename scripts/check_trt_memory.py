"""Load all pose engines together; optionally exercise 12/6-candidate inference.

Uses existing engines only. Synthetic inputs check execution, not pose accuracy.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def output_statistics(output):
    """Report every invalid output instead of stopping at the first network."""
    import torch
    statistics = {}
    for name, value in output.items():
        statistics[name] = {"shape": list(value.shape), "elements": value.numel(),
                            "nan": int(torch.isnan(value).sum().item()),
                            "inf": int(torch.isinf(value).sum().item())}
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true", help="Also execute synthetic batches of 12 and 6")
    parser.add_argument("--models", nargs="+", default=["wapr_w_mask", "wapr_wo_mask", "sapr", "wbps"],
                        choices=["wapr_w_mask", "wapr_wo_mask", "sapr", "wbps"])
    parser.add_argument("--memory", choices=["shared", "private"], default="shared")
    parser.add_argument("--input", choices=["zeros", "random"], default="zeros")
    parser.add_argument("--batches", nargs="+", type=int, default=[12, 6, 2, 3])
    args = parser.parse_args()
    if args.memory == "private" and len(args.models) != 1:
        parser.error("Use --models NAME with --memory private to avoid loading four large private buffers")
    if any(count <= 0 for count in args.batches):
        parser.error("--batches must be positive")

    import torch
    from wapr import recipe
    from wapr.nets import load_net
    from wapr.trt_memory import SharedExecutionMemory

    def report(stage):
        info = {"stage": stage}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                if key in ("MemAvailable", "SwapFree"):
                    info[key + "_mib"] = int(value.split()[0]) // 1024
        free, total = torch.cuda.mem_get_info(args.device)
        info["cuda_free_mib"] = round(free / 1024 ** 2, 2)
        print("TRT_MEMORY_CHECK", json.dumps(info), flush=True)

    torch.manual_seed(0)
    memory = SharedExecutionMemory(args.device) if args.memory == "shared" else None
    nets = {}
    sizes = []
    report("before_load")
    for name in args.models:
        nets[name] = load_net(name, recipe.weight_file(name), engine=recipe.engine_file(name),
                              device=args.device, execution_memory=memory)
        sizes.append(int(nets[name].engine.engine.device_memory_size))
        report("loaded_" + name)
    print("TRT_EXECUTION_BUDGET", {"separate_mib": round(sum(sizes) / 1024 ** 2, 2),
          "shared_mib": round(max(sizes) / 1024 ** 2, 2)}, flush=True)
    failed = []
    if args.run:
        with torch.inference_mode():
            for count in args.batches:
                for name, net in nets.items():
                    shape = (count, int(recipe.channels[name]), int(recipe.crop_px), int(recipe.crop_px))
                    if args.input == "zeros":
                        A = torch.zeros(shape, device=args.device)
                        B = torch.zeros_like(A)
                    else:
                        A = torch.rand(shape, device=args.device) - 0.5
                        B = torch.rand(shape, device=args.device) - 0.5
                    output = net(A, B)
                    torch.cuda.synchronize(args.device)
                    stats = output_statistics(output)
                    print("TRT_OUTPUT_CHECK", json.dumps({"model": name, "batch": count,
                          "memory": args.memory, "input": args.input, "outputs": stats}), flush=True)
                    if any(value["nan"] or value["inf"] for value in stats.values()):
                        failed.append("%s:%d" % (name, count))
                    report("ran_%s_batch_%d" % (name, count))
                    del A, B, output
    if failed:
        print("TRT_MEMORY_CHECK_FAILED non-finite outputs:", ", ".join(failed), flush=True)
        return 1
    print("TRT_MEMORY_CHECK_OK (selected engines retained; no pose accuracy validation)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
