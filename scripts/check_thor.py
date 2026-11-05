"""Check the Thor GPU stack; this does not test RealSense or the OGL renderer."""

import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_platform(system, machine, python_version):
    if system != "Linux" or machine.lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Thor setup requires Linux ARM64")
    if tuple(python_version[:2]) != (3, 12):
        raise RuntimeError("The supplied ONNX Runtime wheel requires CPython 3.12")


def validate_versions(cuda_version, capability, tensorrt_version):
    if not cuda_version or int(cuda_version.split(".")[0]) != 13:
        raise RuntimeError("Install torch/torchvision from https://download.pytorch.org/whl/cu130")
    if tuple(capability) != (11, 0):
        raise RuntimeError("Expected Thor compute capability 11.0; got %s" % (capability,))
    if int(tensorrt_version.split(".")[0]) != 10:
        raise RuntimeError("This runtime supports TensorRT 10.x on Thor; TensorRT 11.x needs additional API changes")


def check_onnxruntime_cuda():
    """Execute a small ONNX model with CPU fallback explicitly disabled."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The installed ONNX Runtime does not expose CUDAExecutionProvider")
    tensor = onnx.TensorProto.FLOAT
    model = onnx.helper.make_model(onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["a", "b"], ["out"])], "thor_cuda_check",
        [onnx.helper.make_tensor_value_info(name, tensor, [2, 2]) for name in ("a", "b")],
        [onnx.helper.make_tensor_value_info("out", tensor, [2, 2])]),
        opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                  providers=["CUDAExecutionProvider"])
    # ORT may retry session creation with CPU after an EP initialization failure.
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime failed to initialize CUDA; check this wheel's CUDA/cuDNN libraries")
    session.disable_fallback()
    output, = session.run(None, {"a": np.ones((2, 2), np.float32), "b": np.ones((2, 2), np.float32)})
    np.testing.assert_allclose(output, np.full((2, 2), 2, np.float32))
    return {"version": ort.__version__, "providers": session.get_providers(), "cuda_matmul": "passed"}


def main():
    validate_platform(platform.system(), platform.machine(), sys.version_info)
    import numpy as np
    import torch
    import torchvision
    import tensorrt as trt
    from torchvision.ops import nms

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA; check the host driver and Docker NVIDIA runtime")
    capability = torch.cuda.get_device_capability(0)
    validate_versions(torch.version.cuda, capability, trt.__version__)
    value = torch.ones((32, 32), device="cuda")
    if (value @ value)[0, 0].item() != 32:
        raise RuntimeError("PyTorch CUDA matmul failed")
    np.testing.assert_array_equal(torch.from_numpy(np.ones(3)).numpy(), np.ones(3))
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    if nms(boxes, scores, 0.5).cpu().tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS failed")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("TensorRT builder creation failed")
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "capability": capability,
              "tensorrt": trt.__version__, "onnxruntime": check_onnxruntime_cuda(),
              "note": "GPU stack checks only; camera, native renderer and pose inference are not validated"}
    print("THOR_GPU_STACK_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
