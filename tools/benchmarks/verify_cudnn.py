"""Verify a local cuDNN installation with a real tiny CUDA convolution.

This is a separate library smoke check, not a FeatherTalk cuDNN backend or a
performance comparison. Only the probe process environment is changed.
"""

import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path, default=Path(r"D:\environment\cuda-v12.6"))
    parser.add_argument("--cudnn", type=Path, default=Path(r"C:\Program Files\NVIDIA\CUDNN\v9.3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cuda_bin, cudnn_bin = args.cuda / "bin", args.cudnn / "bin/12.6"
    os.environ["PATH"] = os.pathsep.join([str(cuda_bin), str(cudnn_bin), os.environ.get("PATH", "")])
    dll_directories = [os.add_dll_directory(str(path)) for path in (cuda_bin, cudnn_bin)]
    cudnn = c.WinDLL(str(cudnn_bin / "cudnn64_9.dll"))
    cudart = c.WinDLL(str(cuda_bin / "cudart64_12.dll"))
    pointer, pointer_ref = c.c_void_p, c.POINTER(c.c_void_p)

    def bind(library, name, parameters, result=c.c_int):
        function = getattr(library, name)
        function.argtypes, function.restype = parameters, result
        return function

    def checked(function, *values):
        status = function(*values)
        if status != 0:
            raise RuntimeError(f"{function.__name__} failed with status {status}")

    version = bind(cudnn, "cudnnGetVersion", [], c.c_size_t)()
    cudart_build = bind(cudnn, "cudnnGetCudartVersion", [], c.c_size_t)()
    assert version == 90300, version
    allocate = bind(cudart, "cudaMalloc", [pointer_ref, c.c_size_t])
    free = bind(cudart, "cudaFree", [pointer])
    memcpy = bind(cudart, "cudaMemcpy", [pointer, pointer, c.c_size_t, c.c_int])
    synchronize = bind(cudart, "cudaDeviceSynchronize", [])
    cleanup = []

    def descriptor(create, destroy):
        handle = pointer()
        checked(bind(cudnn, create, [pointer_ref]), c.byref(handle))
        cleanup.append((bind(cudnn, destroy, [pointer]), handle))
        return handle

    def device_memory(size):
        address = pointer()
        checked(allocate, c.byref(address), size)
        cleanup.append((free, address))
        return address

    try:
        handle = descriptor("cudnnCreate", "cudnnDestroy")
        x_desc = descriptor("cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor")
        y_desc = descriptor("cudnnCreateTensorDescriptor", "cudnnDestroyTensorDescriptor")
        w_desc = descriptor("cudnnCreateFilterDescriptor", "cudnnDestroyFilterDescriptor")
        conv_desc = descriptor("cudnnCreateConvolutionDescriptor", "cudnnDestroyConvolutionDescriptor")
        tensor = bind(cudnn, "cudnnSetTensor4dDescriptor", [pointer] + [c.c_int] * 6)
        checked(tensor, x_desc, 0, 0, 1, 1, 5, 5)
        checked(tensor, y_desc, 0, 0, 1, 1, 3, 3)
        checked(bind(cudnn, "cudnnSetFilter4dDescriptor", [pointer] + [c.c_int] * 6), w_desc, 0, 0, 1, 1, 3, 3)
        checked(bind(cudnn, "cudnnSetConvolution2dDescriptor", [pointer] + [c.c_int] * 8),
                conv_desc, 0, 0, 1, 1, 1, 1, 1, 0)
        x_host, w_host, y_host = (c.c_float * 25)(*([1.0] * 25)), (c.c_float * 9)(*([1.0] * 9)), (c.c_float * 9)()
        x, w, y = device_memory(c.sizeof(x_host)), device_memory(c.sizeof(w_host)), device_memory(c.sizeof(y_host))
        checked(memcpy, x, x_host, c.sizeof(x_host), 1)
        checked(memcpy, w, w_host, c.sizeof(w_host), 1)
        workspace_size = c.c_size_t()
        checked(bind(cudnn, "cudnnGetConvolutionForwardWorkspaceSize", [pointer] * 5 + [c.c_int, c.POINTER(c.c_size_t)]),
                handle, x_desc, w_desc, conv_desc, y_desc, 0, c.byref(workspace_size))
        workspace = device_memory(workspace_size.value) if workspace_size.value else pointer()
        alpha, beta = c.c_float(1), c.c_float(0)
        checked(bind(cudnn, "cudnnConvolutionForward", [pointer] * 7 + [c.c_int, pointer, c.c_size_t] + [pointer] * 3),
                handle, c.byref(alpha), x_desc, x, w_desc, w, conv_desc, 0,
                workspace, workspace_size.value, c.byref(beta), y_desc, y)
        checked(synchronize)
        checked(memcpy, y_host, y, c.sizeof(y_host), 2)
        values = list(y_host)
        assert values == [9.0] * 9, values
        result = {"cudnn_root": str(args.cudnn), "cudnn_version": version, "cudnn_cudart_build_version": cudart_build,
                  "operation": "float32 NCHW convolution: ones[1,1,5,5] * ones[1,1,3,3], stride=1, pad=0",
                  "output": values, "expected": [9.0] * 9, "passed": True,
                  "note": "Independent cuDNN library smoke check; FeatherTalk does not execute this code."}
        try:
            import psutil
            result["loaded_gpu_modules"] = sorted({m.path for m in psutil.Process().memory_maps()
                if any(key in m.path.lower() for key in ("cudnn", "cudart", "nvcuda", "cublas"))})
        except ImportError:
            pass
    finally:
        for destroy, value in reversed(cleanup):
            checked(destroy, value)
        for directory in dll_directories:
            directory.close()
    result["dll_files"] = []
    for path in sorted(cudnn_bin.glob("*.dll")):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result["dll_files"].append({"file": str(path), "bytes": path.stat().st_size, "sha256": digest})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("cudnn_version", "cudnn_cudart_build_version", "output", "passed")}, indent=2))


if __name__ == "__main__":
    main()
