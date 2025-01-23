#include "bfs.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include <chrono>

#include "util.h"

#define MAX_THREADS_PER_BLOCK 256

// BFS Kernel 1
extern "C" __global__ void
Kernel(const Node* __restrict__ d_graph_nodes, 
       const int* __restrict__ d_graph_edges,
       char* __restrict__ d_graph_mask,
       char* __restrict__ d_updatind_graph_mask,
       const char *__restrict__ d_graph_visited,
       int* __restrict__ d_cost,
       const int no_of_nodes) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid<no_of_nodes && d_graph_mask[tid])
  {
    d_graph_mask[tid]=0;
    const int num_edges = d_graph_nodes[tid].no_of_edges;
    const int starting = d_graph_nodes[tid].starting;

    for(int i=starting; i<(num_edges + starting); i++)
    {
      int id = d_graph_edges[i];
      if(!d_graph_visited[id])
      {
        d_cost[id]=d_cost[tid]+1;
        d_updatind_graph_mask[id]=1;
      }
    }
  }
}

// BFS Kernel 2
extern "C" __global__ void
Kernel2(char* __restrict__ d_graph_mask,
        char* __restrict__ d_updatind_graph_mask,
        char* __restrict__ d_graph_visited,
        char* __restrict__ d_over,
        const int no_of_nodes)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid<no_of_nodes && d_updatind_graph_mask[tid])
  {
    d_graph_mask[tid]=1;
    d_graph_visited[tid]=1;
    *d_over=1;
    d_updatind_graph_mask[tid]=0;
  }
}

void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost_ref)
{
  char stop;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        for(int i=h_graph_nodes[tid].starting; 
            i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
          int id = h_graph_edges[i];  // node id is connected with node tid
          if(!h_graph_visited[id]){   // if node id has not been visited, enter the body below
            h_cost_ref[id]=h_cost_ref[tid]+1;
            h_updating_graph_mask[id]=1;
          }
        }
      }    
    }

    for(int tid=0; tid< no_of_nodes ; tid++ )
    {
      if (h_updating_graph_mask[tid] == 1){
        h_graph_mask[tid]=1;
        h_graph_visited[tid]=1;
        stop=1;
        h_updating_graph_mask[tid]=0;
      }
    }
  }
  while(stop);
}

//Apply BFS on a Graph
void run_bfs_gpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost)
{

  Node* d_graph_nodes;
  cudaMalloc((void**) &d_graph_nodes, sizeof(Node)*no_of_nodes);

  int* d_graph_edges;
  cudaMalloc((void**) &d_graph_edges, sizeof(int)*edge_list_size);

  char* d_graph_mask;
  cudaMalloc((void**) &d_graph_mask, sizeof(char)*no_of_nodes);

  char* d_updating_graph_mask;
  cudaMalloc((void**) &d_updating_graph_mask, sizeof(char)*no_of_nodes);

  char* d_graph_visited;
  cudaMalloc((void**) &d_graph_visited, sizeof(char)*no_of_nodes) ;

  int* d_cost;
  cudaMalloc((void**) &d_cost, sizeof(int)*no_of_nodes);
  
  long time = 0;
  
  int repeat = 512;

  for (int i = 0; i < repeat; i++) {
  	auto start = std::chrono::steady_clock::now();
	  
	cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice); 
  	cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice); 
  	cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice); 
  	cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice);  
  	cudaMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
  	time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Total memory transfer time : %f (us)\n", ((time * 1e-3f) / (float)repeat));

  char h_over;
  char *d_over;
  cudaMalloc((void**) &d_over, sizeof(char));

  // setup execution parameters
  dim3 grid((no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
  dim3 threads(MAX_THREADS_PER_BLOCK);
  int counter = 0;
  time = 0;
  do {
    h_over = 0;
    cudaMemcpy(d_over, &h_over, sizeof(char), cudaMemcpyHostToDevice) ;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    Kernel<<< grid, threads >>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, 
                                d_graph_visited, d_cost, no_of_nodes);
    Kernel2<<< grid, threads >>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //counter++;
    cudaMemcpy(&h_over, d_over, sizeof(char), cudaMemcpyDeviceToHost) ;
  } while(h_over);
  
  //printf("Iterations: %d\n", counter);
  printf("Total kernel execution time : %f (us)\n", time * 1e-3f);

  // copy result from device to host
  cudaMemcpy(h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;

  cudaFree(d_graph_nodes);
  cudaFree(d_graph_edges);
  cudaFree(d_graph_mask);
  cudaFree(d_updating_graph_mask);
  cudaFree(d_graph_visited);
  cudaFree(d_cost);
  cudaFree(d_over);
}


