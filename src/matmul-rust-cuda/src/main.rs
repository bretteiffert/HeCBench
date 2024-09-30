use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::{distributions::Uniform, Rng};

const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}
";

fn main() -> Result<(), DriverError> {
    let mut start = std::time::Instant::now();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "matmul", &["matmul"])?;
    let f = dev.get_func("matmul", "matmul").unwrap();
    println!("Loaded in {:?}", start.elapsed());
    
    let vec = vec![32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];

    for i in vec {
        let mut a_host = vec![0f32; i * i];
        rand::thread_rng().fill(&mut a_host[..]); 
        let mut b_host = vec![0f32; i * i];
        rand::thread_rng().fill(&mut b_host[..]);

        let mut c_host = vec![0.0f32; i * i];
        
        start = std::time::Instant::now();
        let a_dev = dev.htod_sync_copy(&a_host)?;
        let b_dev = dev.htod_sync_copy(&b_host)?;
        let mut c_dev = dev.htod_sync_copy(&c_host)?;
        println!("Copied in {:?}", start.elapsed());
        
        start = std::time::Instant::now();
        let cfg = LaunchConfig {
            block_dim: (32 as u32, 32 as u32, 1),
            grid_dim: ((i / 32) as u32, (i / 32) as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.clone().launch(cfg, (&a_dev, &b_dev, &mut c_dev, i)) }?;
        
        dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
        println!("matmul in {:?}\n", start.elapsed());
    }
    Ok(())
}
