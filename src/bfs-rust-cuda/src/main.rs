use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;


use std::time::Instant;

const MAX_THREADS_PER_BLOCK: usize = 256;

struct Node {
  starting: usize,
  no_of_edges: usize,
}

fn lines_from_file(filename: impl AsRef<Path>) -> Vec<String> {
    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines()
        .map(|l| l.expect("Could not parse line"))
        .collect()
}

fn main() -> Result<(), DriverError> {
    

    let lines = lines_from_file("graph1MW_6.txt");
    let mut source: usize = 0;

    //println!("{:?}", lines[0]);
    let no_of_nodes: usize = lines[0].parse::<usize>().unwrap();
    println!("{:?}", no_of_nodes);
    
    let mut h_graph_nodes: Vec<Node> = Vec::new();
    let mut h_graph_mask: Vec<usize> = vec![0; no_of_nodes];
    let mut h_updating_graph_mask: Vec<usize> = vec![0; no_of_nodes];
    let mut h_graph_visited: Vec<usize> = vec![0; no_of_nodes];

    for i in 1..(no_of_nodes + 1) {
        let parse_line = lines[i].split(" ").collect::<Vec<_>>();
        let node = Node {
            starting: parse_line[0].parse::<usize>().unwrap(),
            no_of_edges: parse_line[1].parse::<usize>().unwrap(),
        };
        //println!("{:?}", i);
        h_graph_nodes.push(node);
    }
    
    source = lines[no_of_nodes + 1].parse::<usize>().unwrap();
    source = 0;
    //println!("{:?}", lines[no_of_nodes + 1]);
    
    h_graph_mask[source]=1;
    h_graph_visited[source]=1;

    let edge_list_size: usize = lines[no_of_nodes + 1 + 1].parse::<usize>().unwrap();
    
    //println!("{:?}", lines[no_of_nodes + 1 + 1 + 1]);
    
    let mut h_graph_edges: Vec<usize> = Vec::new();
    let mut edge_list = lines[no_of_nodes + 1 + 1 + 1].split(" ").collect::<Vec<_>>();
    for i in (0..edge_list_size*2).step_by(2) {
        let id: usize= edge_list[i].parse::<usize>().unwrap();
        h_graph_edges.push(id);
    }
    
    //println!("{:?}", h_graph_edges[h_graph_edges.len() - 1]);

    let mut h_cost: Vec<usize> = vec![1; no_of_nodes];
    let mut h_cost_ref: Vec<isize> = vec![-1; no_of_nodes];



    /*
    // declare device
    let dev = CudaDevice::new(0)?;
    
    // load code onto gpu
    dev.load_ptx(Ptx::from_file("/home/35e/HeCBench/src/bfs-rust-cuda/src/bfs-kernel.ptx"), "BFS", &["BFS_Kernel"])?;
    dev.load_ptx(Ptx::from_file("/home/35e/HeCBench/src/bfs-rust-cuda/src/bfs-kernel.ptx"), "BFS2", &["BFS_Kernel2"])?;
    let f = dev.get_func("BFS", "BFS_Kernel").unwrap();
    let f2 = dev.get_func("BFS2", "BFS_Kernel2").unwrap();
    */

    



    println!("hello world");
    
    Ok(())
}
