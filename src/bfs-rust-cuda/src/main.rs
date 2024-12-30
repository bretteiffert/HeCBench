use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use std::env;


use std::time::Instant;

const MAX_THREADS_PER_BLOCK: i32 = 256;

fn lines_from_file(filename: impl AsRef<Path>) -> Vec<String> {
    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines()
        .map(|l| l.expect("Could not parse line"))
        .collect()
}

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(1)?;
    dev.load_ptx(Ptx::from_file("/home/35e/HeCBench/src/bfs-rust-cuda/src/bfs-kernel.ptx"), "BFS", &["BFS_Kernel"])?;
    dev.load_ptx(Ptx::from_file("/home/35e/HeCBench/src/bfs-rust-cuda/src/bfs-kernel.ptx"), "BFS2", &["BFS_Kernel2"])?;
    let f = dev.get_func("BFS", "BFS_Kernel").unwrap();
    let f2 = dev.get_func("BFS2", "BFS_Kernel2").unwrap();

    let args: Vec<String> = env::args().collect();
    let lines = lines_from_file(&args[1]);
    let source: i32 = 0;

    //println!("{:?}", lines[0]);
    let no_of_nodes: i32 = lines[0].parse::<i32>().unwrap();
    //println!("{:?}", no_of_nodes);
    
    let mut h_graph_nodes_starting: Vec<i32> = Vec::new();
    let mut h_graph_nodes_no_of_edges: Vec<i32> = Vec::new();
    let mut h_graph_mask: Vec<i8> = vec![0; no_of_nodes as usize];
    let h_updating_graph_mask: Vec<i8> = vec![0; no_of_nodes as usize];
    let mut h_graph_visited: Vec<i8> = vec![0; no_of_nodes as usize];
    let h_over: Vec<i8> = vec![0; 1];

    for i in 1..(no_of_nodes + 1) {
        let parse_line = lines[i as usize].split(" ").collect::<Vec<_>>();
        h_graph_nodes_starting.push(parse_line[0].parse::<i32>().unwrap());
        h_graph_nodes_no_of_edges.push(parse_line[1].parse::<i32>().unwrap());
        //println!("{:?}", i);
    }
    
    //source = lines[(no_of_nodes + 1) as usize].parse::<i32>().unwrap();
    //source = 0;
    //println!("{:?}", lines[no_of_nodes + 1]);
    
    h_graph_mask[source as usize]=1;
    h_graph_visited[source as usize]=1;

    let edge_list_size: i32 = lines[(no_of_nodes + 1 + 1) as usize].parse::<i32>().unwrap();
    
    //println!("{:?}", lines[no_of_nodes + 1 + 1 + 1]);
    
    let mut h_graph_edges: Vec<i32> = Vec::new();
    let edge_list = lines[(no_of_nodes + 1 + 1 + 1) as usize].split(" ").collect::<Vec<_>>();
    for i in (0..edge_list_size*2).step_by(2) {
        let id: i32 = edge_list[i as usize].parse::<i32>().unwrap();
        h_graph_edges.push(id);
    }
    
    //println!("{:?}", h_graph_edges[h_graph_edges.len() - 1]);

    let mut h_cost: Vec<i32> = vec![1; no_of_nodes as usize];
    let mut h_cost_ref: Vec<i32> = vec![-1; no_of_nodes as usize];
    
    h_cost[source as usize] = 0;
    h_cost_ref[source as usize] = 0;
    
    let grid_size = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    let block_size = MAX_THREADS_PER_BLOCK;

    let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
    
    //start gpu transfers
    //


    let now = Instant::now();
    
    let d_graph_nodes_starting = dev.htod_sync_copy(&h_graph_nodes_starting)?;
    let d_graph_nodes_no_of_edges = dev.htod_sync_copy(&h_graph_nodes_no_of_edges)?;
    let d_graph_edges = dev.htod_sync_copy(&h_graph_edges)?;
    let d_graph_mask = dev.htod_sync_copy(&h_graph_mask)?;
    let d_updating_graph_mask = dev.htod_sync_copy(&h_updating_graph_mask)?;
    let d_graph_visited = dev.htod_sync_copy(&h_graph_visited)?;
    let d_cost = dev.htod_sync_copy(&h_cost)?;
    let d_over = dev.htod_sync_copy(&h_over)?;
    
    println!("Mem Transfer Time {:?}\n", (now.elapsed().subsec_nanos() as f32) / (1000 as f32));


    //println!("Rust GPU transfer time {:?} us", end / (1000 as f32));

    //let mut times: Vec<i32> = Vec::new();

    //println!("{:?}",h_graph_mask[0]);
    let now = Instant::now();
    for _i in 0..12 {
        
        //let now = Instant::now();

        unsafe { f.clone().launch(cfg, (&d_graph_nodes_starting, &d_graph_nodes_no_of_edges, &d_graph_edges, &d_graph_mask, &d_updating_graph_mask, &d_graph_visited, &d_cost, no_of_nodes)) }?;
        unsafe { f2.clone().launch(cfg, (&d_graph_mask, &d_updating_graph_mask, &d_graph_visited, &d_over, no_of_nodes))}?;

        let _ = dev.synchronize();
        
        //let temp_time = now.elapsed().subsec_nanos() as f32;
        //times.push(temp_time);   
    }
    
    let end = now.elapsed().subsec_nanos() as f32;
    println!("Rust GPU kernel time {:?} us", end / (1000 as f32));

    //let solution = dev.dtoh_sync_copy(&d_cost)?;
    //for i in solution {
    //    println!("{:?}", i);
    //}

    Ok(())
}
