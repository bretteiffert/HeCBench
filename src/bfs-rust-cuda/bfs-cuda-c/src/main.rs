use std::env;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Node {
    pub starting: ::std::os::raw::c_int,
    pub no_of_edges: ::std::os::raw::c_int,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of Node"][::std::mem::size_of::<Node>() - 8usize];
    ["Alignment of Node"][::std::mem::align_of::<Node>() - 4usize];
    ["Offset of field: Node::starting"][::std::mem::offset_of!(Node, starting) - 0usize];
    ["Offset of field: Node::no_of_edges"][::std::mem::offset_of!(Node, no_of_edges) - 4usize];
};

extern "C" {
    pub fn run_bfs_main();
}

fn main() {
    unsafe {
        run_bfs_main();
    }
}


