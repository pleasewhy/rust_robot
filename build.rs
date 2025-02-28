fn main() {
    println!("cargo:rustc-link-lib=dylib=mujoco.3.2.7");
    println!("cargo:rustc-link-search=native=/Users/hy/Desktop/workplace/rust_robot");
}