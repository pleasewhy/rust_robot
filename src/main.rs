// #![allow(warnings)]

mod mujoco;
mod rl_algorithm;
mod rl_env;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Shape, Tensor, TensorData};
use mujoco::ffi::mj_step;
use ndarray::{stack, Array, Array1};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rl_algorithm::policy_gradient::policy;
use rl_env::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray as nd;
use ndarray::Array3;

use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;

// let start = std::time::Instant::now();
// let mut model = mujoco::model::Model::from_xml_file("humanoid.xml").unwrap();
// let mut data = mujoco::data::Data::new(model.clone());
// unsafe { mj_step(model.get_ref(), data.get_mut()) };
// model.get_body_name();
// model.get_body_name();
// let mut render = mujoco::render::Render::new(model.get_ref());
// println!("head_body_id={}", data.get_body_id("head"));
// println!("head_body_id={:?}", data.get_all_body_name());
// println!("model={:?}", model.get_ref());
// model.name_bodyadr
// unsafe { mj_step(model.get_ref(), data.get_mut()) };
// println!("qpos={}", data.get_qpos());
// render.update_scene(model.get_ref(), data.get_mut());
// let (img, depth) = render.render();
// image::save_buffer(
//     "before.png",
//     img.as_slice(),
//     render.get_width() as u32,
//     render.get_height() as u32,
//     image::ExtendedColorType::Rgb8,
// )
// .unwrap();

// data.get_qpos()[(2, 0)] = 2.0;

// unsafe { mj_step(model.get_ref(), data.get_mut()) };
// render.update_scene(model.get_ref(), data.get_mut());
// let (img, depth) = render.render();
// image::save_buffer(
//     "after.png",
//     img.as_slice(),
//     render.get_width() as u32,
//     render.get_height() as u32,
//     image::ExtendedColorType::Rgb8,
// )
// .unwrap();

// println!("cost={:?}", std::time::Instant::now() - start);
// unsafe { ffi::mj_resetDataKeyframe(model.get_ref(), data.get_mut(), 1) };
// render.camera.distance = 2.0;

// let mut frametime = 0f64;
// let mut framecount = 0;
// let duration = 10f64;
// let fps = 60f64;

// unsafe {
//     let mut frames = Vec::<Array3<u8>>::new();
//     while data.time < duration {
//         let ctrl = std::slice::from_raw_parts_mut(data.ctrl, model.nu.try_into().unwrap());
//         ctrl.copy_from_slice(
//             Array::random(ctrl.len(), Uniform::new(-1.0, 1.0))
//                 .as_slice()
//                 .unwrap(),
//         );
//         mj_step(model.get_ref(), data.get_mut());

//         if (data.time - frametime) > 1.0 / fps || frametime == 0f64 {
//             render.camera.lookat = data.get_subtree_com_by_body("torso");
//             render.update_scene(&model.get_ref(), data.get_mut());
//             let (img, depth) = render.render();
//             frames.push(
//                 Array3::from_shape_vec((render.get_height(), render.get_width(), 3), img)
//                     .unwrap(),
//             );
//             frametime = data.time;
//             framecount += 1;
//         }
//     }
//     println!(
//         "model={:?}",
//         std::slice::from_raw_parts(model.actuator_ctrlrange, model.nu as usize * 2)
//     );
//     println!("qpos={:?}", data.get_qpos());

//     println!("cost={:?}", std::time::Instant::now() - start);
//     video_rs::init().unwrap();
//     println!("cost={:?}", std::time::Instant::now() - start);

//     let settings =
//         Settings::preset_h264_yuv420p(render.get_width(), render.get_height(), false);
//     let duration: Time = Time::from_nth_of_a_second(fps as usize);
//     let mut position = Time::zero();
//     let mut encoder =
//         Encoder::new(Path::new("rainbow1.mp4"), settings).expect("failed to create encoder");
//     for frame in frames {
//         encoder
//             .encode(&frame, position)
//             .expect("failed to encode frame");

//         // Update the current position and add the inter-frame duration to it.
//         position = position.aligned_with(duration).add();
//     }

//     encoder.finish().expect("failed to finish encoder");
// }
// println!("end")
fn main() {
    let mut env = env::Env::new("humanoid.xml", false);
    let action_dim = env.get_action_len();
    let ob_dim = env.get_obs_len();
    println!("ob_dim={}", ob_dim);
    println!("action_dim={}", action_dim);
    let random_policy = move |obs: &Array1<f64>| {
        return Array::random(action_dim, Uniform::new(-1.0, 1.0));
    };
    let learn_rate = 1e-3;
    let reward_discout = 0.99;
    let batch_size = 50000;
    let traj_length = 1000;
    let layer_dim = 256;
    let n_layers = 5;
    let n_iter = 1000;
    let baseline_gradient_steps = 50;
    let pg_agent_conf =
        rl_algorithm::policy_gradient::policy_gradient_agent::PolicyGradientAgentConfig::new(
            ob_dim,
            action_dim,
            false,
            n_layers,
            layer_dim,
            learn_rate,
            reward_discout,
            50,
        );
    println!("conf={}", pg_agent_conf);
    // let pg_agent = pg_agent_conf.init::<Autodiff<NdArray>>(&NdArrayDevice::default());
    // let policy = |obs: &Array1<f64>| pg_agent.get_action(obs);

    let env_vec = (0..6)
        .map(|_| {
            let env = env::Env::new("humanoid.xml", false);
            return Arc::new(Mutex::new(env));
        })
        .collect::<Vec<Arc<Mutex<env::Env>>>>();
    let mut handle_vec = Vec::new();
    let core_ids = core_affinity::get_core_ids().unwrap();
    println!("{:?}", core_ids);
    // let 
    for i in 0..5 {
        let my_env = env_vec[i].clone();
        let my_id = core_ids[i];
        let handle = std::thread::spawn(move || {
            let res = core_affinity::set_for_current(my_id);
            println!("my_id={:?} init={}", my_id, res);
            my_env
                .lock()
                .unwrap()
                .sample_n_trajectories(&random_policy, 1000, 50000 / 5)
        });
        handle_vec.push(handle);
    }
    for handle in handle_vec {
        handle.join().unwrap();
    }
    // std::thread::spawn(||)
    // println!("traj.image_obs.len()={}", traj.image_obs.len());
    // env.save_video(&"my_video.mp4".to_string(), traj.image_obs);
}
