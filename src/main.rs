#![allow(warnings)]

mod mujoco;
mod rl_env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::iter::Enumerate;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;

use glfw::ffi::glfwWindowHint;
use glfw::{self, Context};
use glfw::{fail_on_errors, Glfw};
use image::Rgb;
use mujoco::ffi;
use mujoco::ffi::{mjData, mjModel};
use mujoco::ffi::{mj_deleteData, mj_deleteModel, mj_loadXML, mj_makeData, mj_step};
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::path::Path;

use ndarray::Array3;

use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;

fn main() {
    let start = std::time::Instant::now();
    let mut model = mujoco::model::Model::from_xml_file("humanoid.xml").unwrap();
    let mut data = mujoco::data::Data::new(model.clone());
    let mut render = mujoco::render::Render::new(model.get_ref());

    println!("cost={:?}", std::time::Instant::now() - start);
    unsafe { ffi::mj_resetDataKeyframe(model.get_ref(), data.get_mut(), 1) };
    render.camera.distance = 2.0;

    let mut frametime = 0f64;
    let mut framecount = 0;
    let duration = 10f64;
    let fps = 60f64;

    unsafe {
        let mut frames = Vec::<Array3<u8>>::new();
        while data.time < duration {
            let ctrl = std::slice::from_raw_parts_mut(data.ctrl, model.nu.try_into().unwrap());
            ctrl.copy_from_slice(
                Array::random(ctrl.len(), Uniform::new(-1.0, 1.0))
                    .as_slice()
                    .unwrap(),
            );
            mj_step(model.get_ref(), data.get_mut());

            if (data.time - frametime) > 1.0 / fps || frametime == 0f64 {
                render.camera.lookat = data.get_subtree_com_by_body("torso");
                render.update_scene(&model.get_ref(), data.get_mut());
                let (img, depth) = render.render();
                frames.push(
                    Array3::from_shape_vec((render.get_height(), render.get_width(), 3), img)
                        .unwrap(),
                );
                frametime = data.time;
                framecount += 1;
            }
        }
        println!(
            "model={:?}",
            std::slice::from_raw_parts(model.actuator_ctrlrange, model.nu as usize * 2)
        );

        println!("cost={:?}", std::time::Instant::now() - start);
        video_rs::init().unwrap();
        println!("cost={:?}", std::time::Instant::now() - start);

        let settings =
            Settings::preset_h264_yuv420p(render.get_width(), render.get_height(), false);
        let duration: Time = Time::from_nth_of_a_second(fps as usize);
        let mut position = Time::zero();
        let mut encoder =
            Encoder::new(Path::new("rainbow1.mp4"), settings).expect("failed to create encoder");
        for frame in frames {
            encoder
                .encode(&frame, position)
                .expect("failed to encode frame");

            // Update the current position and add the inter-frame duration to it.
            position = position.aligned_with(duration).add();
        }

        encoder.finish().expect("failed to finish encoder");
    }
    println!("end")
}
