use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

const PTX_SRC: &str = "
extern \"C\" __global__ void triad(float* a, float* b, float* c, float s) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < 4) c[id] = a[id] + s*b[id];
}

";

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "triad", &["triad"])?;
    let f = dev.get_func("triad", "triad").unwrap();
    println!("Loaded in {:?}", start.elapsed());

    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut c_host = [0.0f32; 4];

    let a_dev = dev.htod_sync_copy(&a_host)?;
    let b_dev = dev.htod_sync_copy(&b_host)?;
    let mut c_dev = dev.htod_sync_copy(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let cfg = LaunchConfig {
        block_dim: (4,1,1),
        grid_dim: (1,1,1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, 10 as f32)) }?;

    dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}
