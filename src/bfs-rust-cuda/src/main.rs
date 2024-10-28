use cudarc::driver::{CudaDevice, DriverError};
use std::time::Instant;


def bfs_cpu(adj, s) {
    

}

fn add_edge(adj: &mut Vec<Vec<usize>>, u: usize, v: usize) {
    adj[u].push(v);
    adj[v].push(u);
}


fn main() {
    let vert = 7;
    let mut adj: Vec<Vec<usize>> = vec![vec![]];
    for i in 1..vert {
        adj.push(vec![])
    }
    add_edge(&mut adj, 0, 1);
    add_edge(&mut adj, 0, 2);
    add_edge(&mut adj, 1, 2);
    add_edge(&mut adj, 2, 3);
    add_edge(&mut adj, 2, 4);
    add_edge(&mut adj, 3, 6);
    add_edge(&mut adj, 4, 6);
    add_edge(&mut adj, 6, 5);

    println!("{:?}\n", adj);
    


}
