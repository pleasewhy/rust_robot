fn main() {
    // Get the current directory
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=mujoco");
        println!("cargo:rustc-link-search=native=./");
    } else {
        println!("cargo:rustc-link-lib=dylib=mujoco.3.2.7");
        println!("cargo:rustc-link-search=native=./");
    }
}
