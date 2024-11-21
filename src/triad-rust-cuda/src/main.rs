use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::{distributions::Uniform, Rng};

const PTX_SRC: &str = "
extern \"C\" __global__ void triad(float* a, float* b, float* c, float s, float length) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < length) c[id] = a[id] + s*b[id];
}

";

fn main() -> Result<(), DriverError> {
    
    let vec = vec![2];
    
    let ptx = compile_ptx(PTX_SRC).unwrap();

    let dev = CudaDevice::new(0)?;

    dev.load_ptx(ptx, "triad", &["triad"])?;
    let f = dev.get_func("triad", "triad").unwrap();

    for i in vec {

        let mut a_host = vec![0f32; i];
        rand::thread_rng().fill(&mut a_host[..]); 
        let mut b_host = vec![0f32; i];
        rand::thread_rng().fill(&mut b_host[..]);
        
        println!("A {:?}", a_host);
        println!("B {:?}", b_host);

        let mut c_host = vec![0.0f32; i];

        let a_dev = dev.htod_sync_copy(&a_host)?;
        let b_dev = dev.htod_sync_copy(&b_host)?;
        let mut c_dev = dev.htod_sync_copy(&c_host)?;

        let stream = dev.fork_default_stream()?;
        //println!("stream {:?}", stream);
        let cfg = LaunchConfig {
            block_dim: (i as u32,1,1),
            grid_dim: (1,1,1),
            shared_mem_bytes: 0,
        };
        unsafe { f.clone().launch_on_stream(&stream, cfg, (&a_dev, &b_dev, &mut c_dev, 10 as f32, i as f32)) }?;

        dev.wait_for(&stream)?;

        dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
        println!("Found {:?}", c_host);

    }
    Ok(())
}
