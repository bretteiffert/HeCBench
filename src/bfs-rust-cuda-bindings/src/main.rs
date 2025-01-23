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

fn main() {

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


    
}
