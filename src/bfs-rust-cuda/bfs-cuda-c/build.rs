use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Linking paths are dependent on your Linux distribution.
    // Be aware of LIBRARY_PATH and LD_LIBRARY_PATH.

    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .includes(&["./src", "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/include"])
        .includes(&["./src", "/home/35e/HeCBench/src/bfs-sycl"])
        .files(&[
            "./src/bfs.cu"
        ])
        // Needed because nvcc requires specific gcc version.
        .flag("-ccbin=gcc")
        .flag("-std=c++14")
        .flag("-Xcompiler")
        .flag("-Wall")
        .flag("-arch=sm_80")
        .flag("-I../bfs-sycl")
        .compile("main");
    
    println!("cargo:rustc-link-search=native=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/lib64");

    println!("cargo:rustc-link-lib=cuda");
    // Dynamic
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed={}", "./src");

    Ok(())
}
