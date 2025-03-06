extern crate cmake;
use cmake::Config;

fn main()
{
    let dst = Config::new("libradixsort").build();       

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=radixsort");

    println!("cargo:rustc-link-search=native={}", "/opt/rocm-6.3.0/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    println!("cargo:rustc-link-search=native={}", "/auto/software/swtree/ubuntu22.04/x86_64/gcc/13.2.0/lib64");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed={}", "./src");
}






