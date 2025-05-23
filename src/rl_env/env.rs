use crate::mujoco::Render;
use burn::config::Config;
use ndarray::{self as nd, Array1, Array2, Array3, ArrayView2};
use std::path::Path;
use std::sync::{Arc, Mutex};
use video_rs::encode::Settings;
use video_rs::{Encoder, Time};

#[derive(Default, Debug)]
pub struct StepInfo {
    pub obs: Vec<f64>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub image_obs: Option<Array3<u8>>,
}

#[derive(Config, Debug)]
pub struct EnvConfig {
    pub n_env: usize,
    pub traj_length: usize,
    pub reset_state_use_n_step_before_last_failed: usize,
    pub use_init_state_ratio: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            n_env: 100,
            traj_length: 1000,
            reset_state_use_n_step_before_last_failed: 50,
            use_init_state_ratio: 0.3,
        }
    }
}

pub trait MujocoEnv {
    fn new(is_render: bool, env_config: EnvConfig) -> Self;
    fn reset(&mut self);
    fn get_obs_dim(&self) -> usize;
    fn get_action_dim(&self) -> usize;
    fn step(&mut self, action: &[f64]) -> StepInfo;
    fn get_render(&self) -> Option<&Render>;
    fn get_fps(&self) -> usize {
        return 30;
    }
    fn get_reward(
        &self,
        last_obs: ArrayView2<f64>,
        obs: ArrayView2<f64>,
        action: ArrayView2<f64>,
    ) -> (Array1<f64>, Array1<bool>) {
        let shape = [last_obs.shape()[0]];
        return (
            Array1::<f64>::zeros(shape),
            Array1::<bool>::from_elem(shape, true),
        );
    }
    fn get_obs(&self) -> Vec<f64>;
    fn is_terminated(&self) -> bool;

    fn save_video(&self, filename: &String, image_obs: Vec<nd::Array3<u8>>) {
        let render = self.get_render();
        if render.is_none() {
            println!("env not open render option");
            return;
        }
        let render = self.get_render().unwrap();
        let settings =
            Settings::preset_h264_yuv420p(render.get_width(), render.get_height(), false);
        let duration = Time::from_nth_of_a_second(self.get_fps());
        let mut position = Time::zero();
        let mut encoder =
            Encoder::new(Path::new(filename.as_str()), settings).expect("failed to create encoder");
        for frame in image_obs {
            encoder
                .encode(&frame, position)
                .expect("failed to encode frame");

            // Update the current position and add the inter-frame duration to it.
            position = position.aligned_with(duration).add();
        }
    }

    fn run_policy<F>(&mut self, filename_prefix: &String, n_step: usize, policy: &F)
    where
        F: Fn(Array2<f64>) -> Array2<f64>,
    {
        self.reset();
        let obs_dim = self.get_obs_dim();
        let mut reward: f64 = 0.0;
        let mut vec_image_obs = vec![];

        for _ in 0..10 {
            for _ in 0..n_step {
                let mut obs = self.get_obs();
                let obs = Array2::from_shape_vec([1, obs_dim], obs).unwrap();
                let action = policy(obs);
                let info = self.step(action.row(0).to_slice().unwrap());
                reward += info.reward;
                vec_image_obs.push(info.image_obs.unwrap());
                if info.terminated {
                    break;
                }
            }
        }

        let filename = format!("{}_reward_{:.2}.mp4", filename_prefix, reward);
        println!("{}", filename);
        self.save_video(&filename, vec_image_obs);
    }
}
