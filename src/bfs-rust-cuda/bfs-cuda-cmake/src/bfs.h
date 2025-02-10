#pragma once 

//Structure to hold a node information
struct Node
{
  int starting;
  int no_of_edges;
};

extern "C" {
  void run_bfs_main();
  /*
  void run_bfs_cpu(int no_of_nodes, struct Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost_ref);
  
  void run_bfs_gpu(int no_of_nodes, struct Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost); 
  */
}
