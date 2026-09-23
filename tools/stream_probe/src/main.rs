// Feasibility probe for cuDNN <-> cubecl interop.
//
// Question: using ONLY cubecl public APIs, can we (a) obtain the raw CUDA
// device pointer of a cubecl tensor allocation, and (b) obtain the CUDA
// stream cubecl runs its kernels on?
//
// (a) is required to hand buffers to cuDNN. (b) decides whether cuDNN can
// share cubecl's stream (cheap) or must be ordered with events/full syncs.

use cubecl_core::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};

fn main() {
    let device = CudaDevice::default();
    let client = CudaRuntime::client(&device);
    println!("client loaded for {device:?}");

    // Allocate a small buffer through the public client API.
    let n_bytes = 256usize;
    let handle = client.empty(n_bytes);
    println!("allocated handle of {n_bytes} bytes");

    // (a) Device pointer via the public `get_resource` escape hatch.
    match client.get_resource(handle.clone()) {
        Ok(managed) => {
            let res = managed.resource();
            // `res` is cubecl_cuda's GpuResource: pub ptr/binding/size.
            println!(
                "DEVICE PTR OK: ptr=0x{:x} size={} (via ComputeClient::get_resource().resource().ptr)",
                res.ptr, res.size
            );
        }
        Err(e) => println!("get_resource FAILED: {e:?}"),
    }

    // (b) Stream: there is no public API on ComputeClient / CudaRuntime that
    // returns the CUstream. The `compute` module (Stream { pub sys: CUstream },
    // CudaServer, CudaContext) is private (`mod compute;`), so it cannot be
    // named or reached from downstream crates. Documented here as the verdict;
    // there is nothing to call.
    println!("STREAM: no public accessor exists (compute module is private) -> cannot share stream");

    client.sync();
    println!("done");
}
