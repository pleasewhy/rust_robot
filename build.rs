fn main() {
    if std::path::Path::new("target/debug").exists()
        && !std::path::Path::new("target/debug/libmujoco.so").exists()
    {
        std::fs::copy("libmujoco.so", "target/debug/libmujoco.so").unwrap();
        std::fs::copy("libmujoco.so.3.2.7", "target/debug/libmujoco.so.3.2.7").unwrap();
        std::fs::copy(
            "libmujoco.3.2.7.dylib",
            "target/debug/libmujoco.3.2.7.dylib",
        )
        .unwrap();
    }

    if std::path::Path::new("target/release").exists()
        && !std::path::Path::new("target/release/libmujoco.so").exists()
    {
        std::fs::copy("libmujoco.so", "target/release/libmujoco.so").unwrap();
        std::fs::copy("libmujoco.so.3.2.7", "target/release/libmujoco.so.3.2.7").unwrap();
        std::fs::copy(
            "libmujoco.3.2.7.dylib",
            "target/release/libmujoco.3.2.7.dylib",
        )
        .unwrap();
    }

    // Get the current directory
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=mujoco");
        println!("cargo:rustc-link-search=native=./");
    } else {
        println!("cargo:rustc-link-lib=dylib=mujoco.3.2.7");
        println!("cargo:rustc-link-search=native=./");
    }
}
