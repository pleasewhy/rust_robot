fn main() {
    // Get the current directory
    let LD_LIBRARY_PATH = std::env::var("LD_LIBRARY_PATH").unwrap();
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=mujoco");
        println!("cargo:rustc-link-search=native=./");
        
        // Add current directory to LD_LIBRARY_PATH for Linux
        println!("cargo:rustc-env=LD_LIBRARY_PATH=./:{}", LD_LIBRARY_PATH);
    } else {
        println!("cargo:rustc-link-lib=dylib=mujoco.3.2.7");
        println!("cargo:rustc-link-search=native=./");
    }
}
