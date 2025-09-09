// modify from openai gym inverted_pendulum_v4
// https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum_v4.py

use super::config::EnvConfig;
use super::env::{MujocoEnv, StepInfo};
use crate::mujoco;
use lazy_static::lazy_static;
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use std::sync::{Arc, Mutex};

lazy_static! {
    static ref InvertedPendulumV4Model: Arc<mujoco::Model> =
        mujoco::Model::from_xml_file("./inverted_pendulum.xml")
            .expect("load inverted_pendulum.xml Failed.");
}
pub struct InvertedPendulumV4 {
    pub model: Arc<mujoco::model::Model>,
    pub data: mujoco::data::Data,
    pub render: Option<mujoco::render::Render>,
    pub is_render: bool,
    pub fps: usize,
    env_conf: EnvConfig,
    terminated: bool,
}

unsafe impl Send for InvertedPendulumV4 {}
unsafe impl Sync for InvertedPendulumV4 {}

impl MujocoEnv for InvertedPendulumV4 {
    fn reset(&mut self) {
        unsafe {
            mujoco::ffi::mj_resetData(self.model.get_ref(), self.data.get_mut());
        };
        self.terminated = false;
    }
    fn get_obs_dim(&self) -> usize {
        return self.get_obs().len();
    }

    fn get_action_dim(&self) -> usize {
        return self.data.get_ctrl().len();
    }
    fn get_render(&self) -> Option<&mujoco::render::Render> {
        return self.render.as_ref();
    }
    fn get_fps(&self) -> usize {
        return self.fps;
    }
    fn is_terminated(&self) -> bool {
        self.terminated
    }

    fn get_reward(
        &self,
        last_obs: ArrayView2<f64>,
        obs: ArrayView2<f64>,
        action: ArrayView2<f64>,
    ) -> (Array1<f64>, Array1<bool>) {
        let reward = Array1::<f64>::ones(obs.shape()[0]);
        let terminated = obs.slice(s![0.., 1..2]).abs().flatten().map(|x| *x > 0.2);
        return (reward, terminated);
    }

    fn step(&mut self, action: &[f64]) -> StepInfo {
        self.data
            .get_ctrl()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(action);
        if self.terminated {
            return StepInfo {
                obs: Vec::new(),
                reward: 0f64,
                terminated: true,
                truncated: false,
                image_obs: None,
            };
        }
        unsafe { mujoco::ffi::mj_step(self.model.get_ref(), self.data.get_mut()) };
        let obs = self.get_obs();
        let reward: f64 = 1.0;
        let terminated = obs[1].abs() > 0.2;
        self.terminated = terminated;

        let mut image_obs = None;
        if self.is_render {
            let render = self.render.as_mut().unwrap();
            render.update_scene(&self.model, &mut self.data);
            let (image, _) = render.render(None);
            let image = Array3::from_shape_vec((render.get_height(), render.get_width(), 3), image)
                .unwrap();
            image_obs = Some(image);
        }
        let step_info = StepInfo {
            obs: obs,
            reward: reward,
            terminated: terminated,
            truncated: false,
            image_obs: image_obs,
        };
        return step_info;
    }
    fn get_obs(&self) -> Vec<f64> {
        let data = &self.data;
        let position = data.get_qpos();
        let velocity = data.get_qvel();
        let total_size = position.len() + velocity.len();
        let mut vec = vec![0f64; total_size];
        vec[0..position.len()].copy_from_slice(position.as_slice().unwrap());
        vec[position.len()..].copy_from_slice(velocity.as_slice().unwrap());
        return vec;
    }
    fn new(is_render: bool, env_conf: EnvConfig) -> Self {
        return Self::from_model(InvertedPendulumV4Model.clone(), is_render, env_conf);
    }
}

impl InvertedPendulumV4 {
    pub fn from_model(
        model: Arc<mujoco::model::Model>,
        is_render: bool,
        env_conf: EnvConfig,
    ) -> Self {
        let data = mujoco::Data::new(model.clone());
        let mut render = Option::None;
        if is_render {
            render = Some(mujoco::render::Render::new(model.get_ref()));
        }
        return Self {
            model,
            data,
            render,
            is_render: is_render,
            fps: 30,
            env_conf,
            terminated: false,
        };
    }
}
