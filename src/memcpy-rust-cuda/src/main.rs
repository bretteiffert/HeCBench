use cudarc::driver::{CudaDevice, CudaSlice, DriverError};
use std::time::Instant;

fn main() -> Result<(), DriverError> {
    let vec = vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
    let dev = CudaDevice::new(0)?;
    
    // warmup GPU
    let mut warmup = dev.alloc_zeros::<f64>(1)?;
    dev.htod_copy_into(vec![1.0; 1], &mut warmup)?;
    dev.sync_reclaim(warmup)?;

    for i in vec {
        
        // host to device
        let to_be_copied = vec![1.0; i];
        let mut a = dev.alloc_zeros::<f64>(i)?;
        let mut now = Instant::now();
        dev.htod_copy_into(to_be_copied, &mut a)?;
        let time_to_device = now.elapsed();
        
        // device to host
        now = Instant::now();
        let a_host: Vec<f64> = dev.sync_reclaim(a)?;
        //let a_host = dev.dtoh_sync_copy(&a)?; // slow
        let time_to_host = now.elapsed();
        
        println!("Copy {:?} bytes from host to device: {:?} us", i*size_of::<usize>(), time_to_device.subsec_nanos() as f32 / 1000 as f32);
        println!("Copy {:?} bytes from device to host: {:?} us\n", i*size_of::<usize>(), time_to_host.subsec_nanos() as f32 / 1000 as f32);

        assert_eq!(a_host.len(), i);

    }
    
    Ok(())
}
