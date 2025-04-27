// modify from openai gym humanoid-v4
// https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v4.py

use super::config::EnvConfig;
use super::env::{MujocoEnv, StepInfo};
use crate::mujoco;
use lazy_static::lazy_static;
use ndarray::{self as nd, s, Array1, Array2, Array3, Axis};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use rand::{random, Rng};
use std::collections::vec_deque::VecDeque;
use std::sync::{Arc, Mutex};

lazy_static! {
    static ref GymHumanoidV4Model: Arc<mujoco::Model> =
        mujoco::Model::from_xml_file("./gym_humanoid.xml").expect("load gym_humanoid.xml Failed.");
}
pub struct HumanoidV4 {
    pub model: Arc<mujoco::model::Model>,
    pub data: mujoco::data::Data,
    pub render: Option<mujoco::render::Render>,
    pub is_render: bool,
    pub fps: usize,
    terminated: bool,
    forward_reward_weight: f64,
    ctrl_cost_weight: f64,
    healthy_reward: f64,
    terminate_when_unhealthy: bool,
    healthy_z_range: (f64, f64),
    reset_noise_scale: f64,
    exclude_current_positions_from_observation: bool,
    skip_steps: usize,
    env_config: EnvConfig,
    init_qpos: Array2<f64>,
    init_qvel: Array2<f64>,
    init_xquat: Array2<f64>,
    last_n_qpos: VecDeque<Array2<f64>>,
    last_n_qvel: VecDeque<Array2<f64>>,
}

unsafe impl Send for HumanoidV4 {}
unsafe impl Sync for HumanoidV4 {}

impl MujocoEnv for HumanoidV4 {
    fn reset(&mut self) {
        let noise_mean = 0.0;
        let noise_std = self.reset_noise_scale;
        let qpos_shape = self.init_qpos.shape();
        let qvel_shape = self.init_qvel.shape();
        let mut init_qpos = self
            .last_n_qpos
            .pop_front()
            .unwrap_or_else(|| self.init_qpos.clone());
        let mut init_qvel = self
            .last_n_qvel
            .pop_front()
            .unwrap_or_else(|| self.init_qvel.clone());
        if self.is_render || rand::rng().random_bool(self.env_config.use_init_state_ratio) {
            init_qpos = self.init_qpos.clone();
            init_qvel = self.init_qvel.clone();
        }
        let q_pos_noise = &init_qpos
            + Array2::random(
                (qpos_shape[0], qpos_shape[1]),
                Normal::new(noise_mean, noise_std).unwrap(),
            );
        let q_vel_noise = &init_qvel
            + Array2::random(
                (qvel_shape[0], qvel_shape[1]),
                Normal::new(noise_mean, noise_std).unwrap(),
            );
        self.data
            .get_qpos()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(q_pos_noise.as_slice().unwrap());
        self.data
            .get_qvel()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(q_vel_noise.as_slice().unwrap());
        self.last_n_qvel.clear();
        self.last_n_qpos.clear();
        self.last_n_qvel.push_back(self.init_qvel.clone());
        self.last_n_qpos.push_back(self.init_qpos.clone());
        unsafe {
            mujoco::ffi::mj_forward(self.model.get_ref(), self.data.get_mut());
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
        if self.terminate_when_unhealthy {
            return !self.is_healthy();
        }
        return false;
    }

    fn step(&mut self, action: &[f64]) -> StepInfo {
        if self.terminated {
            return StepInfo {
                obs: Vec::new(),
                reward: 0f64,
                terminated: true,
                truncated: false,
                image_obs: None,
            };
        }
        self.data
            .get_ctrl()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(action);

        let ctrl_cost = self.ctrl_cost();
        let xy_position_before = self.mass_center();
        unsafe {
            for _ in 0..self.skip_steps {
                mujoco::ffi::mj_step(self.model.get_ref(), self.data.get_mut());
            }
            mujoco::ffi::mj_rnePostConstraint(self.model.get_ref(), self.data.get_mut());
            mujoco::ffi::mj_sensorAcc(self.model.get_ref(), self.data.get_mut());
        };
        self.last_n_qvel.push_back(self.data.get_qvel().to_owned());
        self.last_n_qpos.push_back(self.data.get_qpos().to_owned());
        if self.last_n_qvel.len() > self.env_config.reset_state_use_n_step_before_last_failed {
            self.last_n_qvel.pop_front();
            self.last_n_qpos.pop_front();
        }
        let xy_position_after = self.mass_center();
        let xy_velocity = (&xy_position_after - &xy_position_before)
            / (self.model.opt.timestep * self.skip_steps as f64);
        let (x_velocity, y_velocity) = (xy_velocity[0], xy_velocity[1]);

        let forward_reward = self.forward_reward_weight * x_velocity;
        let healthy_reward = self.cal_healthy_reward();
        let rewards = forward_reward + healthy_reward;
        // let rewards =
        // forward_reward + healthy_reward + self.orientation_reward() + self.height_reward();

        let reward = rewards - ctrl_cost;

        let after_step_obs = self.get_obs();
        self.terminated = self.is_terminated();

        let mut image_obs = None;
        if self.is_render {
            let render = self.render.as_mut().unwrap();
            render.update_scene(&self.model, &mut self.data);
            let (image, _) = render.render();
            let image = Array3::from_shape_vec((render.get_height(), render.get_width(), 3), image)
                .unwrap();
            image_obs = Some(image);
        }
        let step_info = StepInfo {
            obs: after_step_obs,
            reward: reward,
            terminated: self.terminated,
            truncated: false,
            image_obs: image_obs,
        };
        return step_info;
    }
    fn get_obs(&self) -> Vec<f64> {
        let mut vec = Vec::new();
        self.extend_from_obs(&mut vec);
        return vec;
    }
    fn new(is_render: bool, env_config: EnvConfig) -> Self {
        return Self::from_model(GymHumanoidV4Model.clone(), is_render, env_config);
    }
}

impl HumanoidV4 {
    pub fn from_model(
        model: Arc<mujoco::model::Model>,
        is_render: bool,
        env_config: EnvConfig,
    ) -> Self {
        let data = mujoco::Data::new(model.clone());
        let mut render = Option::None;
        if is_render {
            render = Some(mujoco::render::Render::new(model.get_ref()));
        }
        let init_qpos = data.get_qpos().to_owned();
        let init_qvel = data.get_qvel().to_owned();
        let init_xquat = data.get_xquat().to_owned();

        return Self {
            model,
            data,
            render,
            is_render: is_render,
            fps: 30,
            terminated: false,
            forward_reward_weight: 1.25,
            ctrl_cost_weight: 0.1,
            healthy_reward: 5.0,
            terminate_when_unhealthy: true,
            healthy_z_range: (1.0, 2.0),
            reset_noise_scale: 1e-2,
            exclude_current_positions_from_observation: true,
            skip_steps: 5,
            last_n_qvel: VecDeque::with_capacity(
                env_config.reset_state_use_n_step_before_last_failed,
            ),
            last_n_qpos: VecDeque::with_capacity(
                env_config.reset_state_use_n_step_before_last_failed,
            ),
            env_config,
            init_qpos,
            init_qvel,
            init_xquat,
        };
    }

    fn ctrl_cost(&self) -> f64 {
        let control_cost = self.ctrl_cost_weight * self.data.get_ctrl().pow2().sum();
        return control_cost;
    }

    fn orientation_reward(&self) -> f64 {
        //   BODY 1: torso
        //   BODY 2: head
        //   BODY 3: waist_lower
        //   BODY 4: pelvis
        //   BODY 5: thigh_right
        //   BODY 6: shin_right
        //   BODY 7: foot_right
        //   BODY 8: thigh_left
        //   BODY 9: shin_left
        //   BODY 10: foot_left
        //   BODY 11: upper_arm_right
        //   BODY 12: lower_arm_right
        //   BODY 13: hand_right
        //   BODY 14: upper_arm_left
        //   BODY 15: lower_arm_left
        //   BODY 16: hand_left
        // println!("shape={:?}", self.data.get_xquat().shape());

        let bodies_indics = s![1..14, 0..3];
        let xquat = self.data.get_xquat();
        let torso_quat = xquat.slice(bodies_indics);
        let init_quat = self.init_xquat.slice(bodies_indics);
        // assert_eq!(torso_quat.shape()[0], 4);
        // assert_eq!(init_quat.shape()[0], 4);
        let diff_torso_quat = (&torso_quat - &init_quat).pow2().sqrt().sum();
        return (5.0 - diff_torso_quat).max(0.0);
    }

    fn height_reward(&self) -> f64 {
        let torso_height_id = 2;
        let qpos = self.data.get_qpos();
        let torso_height = qpos.row(torso_height_id)[0];
        let init_torso_height = self.init_qpos.row(torso_height_id)[0];
        let diff_torso_height = (torso_height - init_torso_height).abs();
        let reward = 1.0 - diff_torso_height;

        return reward.max(0.0);
    }

    fn extend_from_obs(&self, vec: &mut Vec<f64>) {
        let data = &self.data;
        let position = data.get_qpos();
        let velocity = data.get_qvel();
        let com_inertia = data.get_cinert();
        let com_velocity = data.get_cvel();
        let actuator_forces = data.get_qfrc_actuator();
        let external_contact_forces = data.get_cfrc_ext();

        let total_size = position.len()
            + velocity.len()
            + com_inertia.len()
            + com_velocity.len()
            + actuator_forces.len()
            + external_contact_forces.len();
        vec.reserve_exact(total_size);
        if self.exclude_current_positions_from_observation {
            vec.extend_from_slice(&position.as_slice().unwrap()[2..]);
        } else {
            vec.extend_from_slice(position.as_slice().unwrap());
        }
        vec.extend_from_slice(velocity.as_slice().unwrap());
        vec.extend_from_slice(com_inertia.as_slice().unwrap());
        vec.extend_from_slice(com_velocity.as_slice().unwrap());
        vec.extend_from_slice(actuator_forces.as_slice().unwrap());
        vec.extend_from_slice(external_contact_forces.as_slice().unwrap());
    }

    fn cal_healthy_reward(&self) -> f64 {
        if self.is_healthy() || self.terminate_when_unhealthy {
            return self.healthy_reward;
        }
        return 0.0;
    }

    fn is_healthy(&self) -> bool {
        let (min_z, max_z) = self.healthy_z_range;
        let torso_height = self.data.get_qpos()[(2usize, 0)];
        let is_healthy = torso_height < max_z && torso_height > min_z;
        return is_healthy;
    }

    fn mass_center(&self) -> Array1<f64> {
        let body_mass = self.model.get_body_mass();
        let xpos = self.data.get_xipos();
        let res = (&body_mass * &xpos).sum_axis(Axis(0)) / body_mass.sum();
        return res.slice(s![0..2]).to_owned();
    }
}
