"""Load all pose engines together; optionally exercise 12/6-candidate inference.

Uses existing engines only. Synthetic inputs check execution, not pose accuracy.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true", help="Also execute synthetic batches of 12 and 6")
    args = parser.parse_args()

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

    memory = SharedExecutionMemory(args.device)
    nets = {}
    sizes = []
    report("before_load")
    for name in ("wapr_w_mask", "wapr_wo_mask", "sapr", "wbps"):
        nets[name] = load_net(name, recipe.weight_file(name), engine=recipe.engine_file(name),
                              device=args.device, execution_memory=memory)
        sizes.append(int(nets[name].engine.engine.device_memory_size))
        report("loaded_" + name)
    print("TRT_EXECUTION_BUDGET", {"separate_mib": round(sum(sizes) / 1024 ** 2, 2),
          "shared_mib": round(max(sizes) / 1024 ** 2, 2)}, flush=True)
    if args.run:
        with torch.inference_mode():
            for count in (12, 6):
                for name, net in nets.items():
                    shape = (count, int(recipe.channels[name]), int(recipe.crop_px), int(recipe.crop_px))
                    A = torch.zeros(shape, device=args.device)
                    B = torch.zeros_like(A)
                    output = net(A, B)
                    torch.cuda.synchronize(args.device)
                    if not all(torch.isfinite(value).all().item() for value in output.values()):
                        raise RuntimeError("Non-finite synthetic output: " + name)
                    report("ran_%s_batch_%d" % (name, count))
                    del A, B, output
    print("TRT_MEMORY_CHECK_OK (all four engines retained; no pose accuracy validation)", flush=True)


if __name__ == "__main__":
    main()
