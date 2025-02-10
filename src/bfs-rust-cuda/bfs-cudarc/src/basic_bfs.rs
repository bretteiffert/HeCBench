use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use cudarc::driver::{CudaDevice, DriverError};
use std::time::Instant;


fn bfs_cpu(adj: &mut Vec<Vec<usize>>, s: usize) {
      
    let mut q: Vec<usize> = vec![];
    
    let mut visited = vec![false; adj.len()];
    
    visited[s] = true;
    q.push(s);    
    
    let start = std::time::Instant::now();
    while !q.is_empty() {
        let curr = q[0];
        q.remove(0);
        for x in &adj[curr] {
            if visited[*x] == false {
                visited[*x] = true;
                q.push(*x);
                println!("{:?}\n", *x);
            }
        }
    }
    println!("Full BFS in {:?}", start.elapsed());
}

fn add_edge(adj: &mut Vec<Vec<usize>>, u: usize, v: usize) {
    adj[u].push(v);
    adj[v].push(u);
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open("graph1MW_6.txt")?;
    Ok(io::BufReader::new(file).lines())
}

fn main() {
    let vert = 7;
    let mut adj: Vec<Vec<usize>> = vec![vec![]];
    for _i in 1..vert {
        adj.push(vec![])
    }
    
    /*if let Ok(mut lines) = read_lines("graph1MW_6.txt") {
        // Consumes the iterator, returns an (Optional) String
        for line in lines.flatten() {
            let edge = line.split(" ").collect::<Vec<_>>();
            //println!("{:?}", edge);
            add_edge(&mut adj, edge[0].to_string().parse::<usize>().unwrap(), edge[1].to_string().parse::<usize>().unwrap());
        }
    }*/ 


    
    add_edge(&mut adj, 0, 1);
    add_edge(&mut adj, 0, 2);
    add_edge(&mut adj, 0, 3);
    add_edge(&mut adj, 1, 4);
    add_edge(&mut adj, 1, 5);
    add_edge(&mut adj, 4, 6);
    
    bfs_cpu(&mut adj, 1);

    println!("{:?}\n", adj);
    


}
