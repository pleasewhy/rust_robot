// modify from openai gym inverted_pendulum_v4
// https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum_v4.py

use super::config::EnvConfig;
use super::env::{MujocoEnv, StepInfo};

use super::env_utils;
use crate::mujoco;
use lazy_static::lazy_static;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, AssignElem};
use num_traits::Float;
use std::sync::{Arc, Mutex};
lazy_static! {
    static ref MobileArmModel: Arc<mujoco::Model> =
        mujoco::Model::from_xml_file("./car.xml").expect("load car.xml Failed.");
}
pub struct MobileArm {
    pub model: Arc<mujoco::model::Model>,
    pub data: mujoco::data::Data,
    pub render: Option<mujoco::render::Render>,
    pub is_render: bool,
    pub fps: usize,
    env_conf: EnvConfig,
    terminated: bool,
    last_render_time: f64,
}

unsafe impl Send for MobileArm {}
unsafe impl Sync for MobileArm {}

impl MujocoEnv for MobileArm {
    fn reset(&mut self) {
        unsafe {
            mujoco::ffi::mj_resetData(self.model.get_ref(), self.data.get_mut());
        };
        self.terminated = false;
        self.last_render_time = 0.0;
    }
    fn get_obs_dim(&self) -> usize {
        return self.get_obs().len();
    }

    fn get_action_dim(&self) -> usize {
        return self.data.get_ctrl().len() + 2;
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
        let (ctrl, stop) = action.split_at(action.len() - 2);

        self.data
            .get_ctrl()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(ctrl);

        let ctrl = ArrayView1::from_shape(ctrl.len(), ctrl).unwrap();
        if self.terminated {
            return StepInfo {
                obs: Vec::new(),
                reward: 0f64,
                terminated: true,
                truncated: false,
                image_obs: None,
            };
        }

        let before_xipos = self.data.get_xipos().to_owned();
        unsafe { mujoco::ffi::mj_step(self.model.get_ref(), self.data.get_mut()) };
        let after_xipos = self.data.get_xipos();

        // let ctrl_cost = -ctrl.pow2().sum() * 0.01;
        let (xv, yv, zv) = env_utils::velocity(
            self.model.opt.timestep,
            self.model.clone(),
            before_xipos.view(),
            after_xipos.view(),
        );
        println!("xv={}, yv={}, zv={}", xv, yv, zv);
        let velocity_reward = -zv.abs() - ((xv.powi(2) + yv.powi(2)).sqrt() - 0.5).abs();
        // println!("velocity_reward={} ctrl_cost={}", velocity_reward, ctrl_cost);
        let reward: f64 = velocity_reward;
        let terminated = false;
        self.terminated = terminated;
        let obs = self.get_obs();

        let mut image_obs = None;
        if self.is_render
            && ((self.data.time - self.last_render_time) > 1.0 / (self.get_fps() as f64)
                || self.last_render_time == 0.0)
        {
            let start = std::time::Instant::now();
            self.last_render_time = self.data.time;
            let render = self.render.as_mut().unwrap();
            render.update_scene(&self.model, &mut self.data);
            let (image, _) = render.render();
            let image = Array3::from_shape_vec((render.get_height(), render.get_width(), 3), image)
                .unwrap();
            image_obs = Some(image);
            println!("render time={:?}", start.elapsed());
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
        return Self::from_model(MobileArmModel.clone(), is_render, env_conf);
    }
}

impl MobileArm {
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
            last_render_time: 0.0,
        };
    }
}
