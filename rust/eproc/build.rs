use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const C_SOURCES: &[&str] = &[
    "eproc_eprocess.c",
    "eproc_conformal.c",
    "eproc_fusion.c",
    "eproc_subsample.c",
];

const C_HEADERS: &[&str] = &[
    "eproc.h",
    "eproc_internal.h",
    "eproc_seqlock.h",
    "eproc_simd.h",
];

fn run_checked(cmd: &mut Command, label: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to spawn `{label}`: {err}"));
    if !status.success() {
        panic!("`{label}` exited with status {status}");
    }
}

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let csrc_dir = manifest_dir.join("../../csrc");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=CFLAGS");

    for header in C_HEADERS {
        let p = csrc_dir.join(header);
        println!("cargo:rerun-if-changed={}", p.display());
    }
    for src in C_SOURCES {
        let p = csrc_dir.join(src);
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let extra_cflags = env::var("CFLAGS").unwrap_or_default();

    let mut objects = Vec::with_capacity(C_SOURCES.len());
    for src_name in C_SOURCES {
        let src_path = csrc_dir.join(src_name);
        let obj_name = Path::new(src_name)
            .with_extension("o")
            .file_name()
            .expect("source file name")
            .to_os_string();
        let obj_path = out_dir.join(obj_name);

        let mut cmd = Command::new(&cc);
        cmd.arg("-std=c11")
            .arg("-O3")
            .arg("-ffp-contract=off")
            .arg("-fPIC")
            .arg("-Wall")
            .arg("-Wextra")
            .arg(format!("-I{}", csrc_dir.display()));

        for flag in extra_cflags.split_whitespace() {
            cmd.arg(flag);
        }

        cmd.arg("-c").arg(&src_path).arg("-o").arg(&obj_path);
        run_checked(&mut cmd, &format!("{cc} ({src_name})"));
        objects.push(obj_path);
    }

    let lib_path = out_dir.join("libeproc_c.a");
    if lib_path.exists() {
        let _ = fs::remove_file(&lib_path);
    }

    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("rcs").arg(&lib_path);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    run_checked(&mut ar_cmd, &format!("{ar} rcs libeproc_c.a"));

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=eproc_c");
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=m");
    }
}
