use crate::mujoco;
use ndarray::{self as nd, Array1, Array3, ArrayView1, Axis};
use ndarray::{s, ArcArray, Array2};
use ndarray_rand::rand_distr::num_traits::zero;
use std::path::Path;
use std::time::SystemTime;
use std::{cmp::min, sync::Arc, sync::Mutex};
use video_rs::encode::Settings;
use video_rs::{Encoder, Time};

pub struct Env {
    pub model: Arc<mujoco::model::Model>,
    pub data: mujoco::data::Data,
    pub data_vec: Vec<Arc<Mutex<mujoco::data::Data>>>, // 用于并行模拟
    pub render: Option<mujoco::render::Render>,
    pub torso_height: f64,
    pub is_render: bool,
    pub fps: usize,
}

unsafe impl Send for Env {}

pub struct Trajectory {
    pub observation: nd::Array2<f64>,
    pub image_obs: Vec<nd::Array3<u8>>,
    pub reward: nd::Array1<f64>,
    pub action: nd::Array2<f64>,
    pub next_observation: nd::Array2<f64>,
    pub terminal: nd::Array1<f64>,
    // pub episode_statistics: nd::Array2<f64>,
}

pub struct StepInfo {
    pub next_obs: nd::Array1<f64>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

impl Env {
    pub fn get_data() {}
    pub fn new(xml_path: &str, is_render: bool) -> Self {
        let model = mujoco::Model::from_xml_file(xml_path).unwrap();
        let data = mujoco::Data::new(model.clone());
        let mut render = Option::None;
        if is_render {
            render = Some(mujoco::render::Render::new(model.get_ref()));
        }
        let mut data_vec = Vec::with_capacity(6);
        for i in 0..6 {
            data_vec.push(mujoco::Data::new_arc(model.clone()));
        }
        let torso_height = data.get_qpos()[(2, 0)];
        return Self {
            model,
            data,
            data_vec,
            render,
            torso_height: torso_height,
            is_render: is_render,
            fps: 30,
        };
    }
    pub fn get_obs(&self) -> nd::Array1<f64> {
        let data = &self.data;
        let position = data.get_qpos();
        let velocity = data.get_qvel();
        let com_inertia = data.get_cinert();
        let com_velocity = data.get_cvel();
        let actuator_forces = data.get_qfrc_actuator();
        let external_contact_forces = data.get_cfrc_ext();
        return ndarray::concatenate![
            ndarray::Axis(0),
            position.flatten(),
            velocity.flatten(),
            com_inertia.flatten(),
            com_velocity.flatten(),
            actuator_forces.flatten(),
            external_contact_forces.flatten()
        ]
        .to_owned();
    }

    pub fn step(&mut self, action: &nd::Array1<f64>) -> StepInfo {
        self.data
            .get_ctrl()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(action.as_slice().unwrap());
        unsafe { mujoco::ffi::mj_step(self.model.get_ref(), self.data.get_mut()) };
        let qpos = self.data.get_qpos();
        // println!("qpos={}", qpos);
        let now_torso_height = *qpos.get((2, 0)).unwrap();
        // println!("now_torso_height={}", now_torso_height);

        let stand_reward = (now_torso_height - self.torso_height) + 1.0;
        let quad_ctrl_cost = 0.1 * self.data.get_ctrl().sqrt().sum();
        let mut quad_impact_cost = 0.5e-6 * self.data.get_cfrc_ext().sqrt().sum();
        quad_impact_cost = quad_impact_cost.min(10.0);
        let reward = stand_reward - quad_ctrl_cost - quad_impact_cost + 1.0;
        return StepInfo {
            next_obs: self.get_obs(),
            reward: reward,
            terminated: false,
            truncated: false,
        };
    }

    pub fn get_obs_len(&self) -> usize {
        return self.get_obs().len();
    }

    pub fn get_action_len(&mut self) -> usize {
        return self.data.get_ctrl().len();
    }

    pub fn sample_trajectory<F>(&mut self, policy: &F, max_length: usize) -> Trajectory
    where
        F: Fn(&nd::Array1<f64>) -> Array1<f64>,
    {
        unsafe {
            mujoco::ffi::mj_resetData(self.model.get_ref(), self.data.get_mut());
        };
        let mut obs_list = Vec::<nd::Array1<f64>>::with_capacity(max_length);
        let mut action_list = Vec::<nd::Array1<f64>>::with_capacity(max_length);
        let mut next_obs_list = Vec::<nd::Array1<f64>>::with_capacity(max_length);
        let mut reward_list = Vec::<f64>::with_capacity(max_length);
        let mut terminal_list = Vec::<f64>::with_capacity(max_length);
        let mut img_list = Vec::<nd::Array3<u8>>::with_capacity(max_length);

        let mut last_render_time = 0f64;
        let fps = self.fps as f64;
        for i in 0..max_length {
            let obs = self.get_obs();
            let action = policy(&obs);
            let step_info = self.step(&action);
            if step_info.terminated || step_info.truncated {
                break;
            }
            obs_list.push(obs);
            reward_list.push(step_info.reward);
            action_list.push(action);
            next_obs_list.push(step_info.next_obs);
            terminal_list.push(step_info.terminated as i8 as f64);
            if self.is_render
                && ((self.data.time - last_render_time) > 1.0 / fps || last_render_time == 0f64)
            {
                let render = self.render.as_mut().unwrap();
                render.update_scene(&self.model, self.data.get_mut());
                let img = render.render().0;
                let img = Array3::from_shape_vec((render.get_height(), render.get_width(), 3), img)
                    .unwrap();
                img_list.push(img);
                last_render_time = self.data.time;
            }
        }
        let stack_vec = |vec: &Vec<nd::Array1<f64>>| {
            nd::stack(
                nd::Axis(0),
                vec.iter()
                    .map(|x: &Array1<f64>| x.view())
                    .collect::<Vec<ArrayView1<f64>>>()
                    .as_ref(),
            )
            .unwrap()
        };
        return Trajectory {
            observation: stack_vec(&obs_list),
            image_obs: img_list,
            reward: nd::Array1::<f64>::from_vec(reward_list),
            action: stack_vec(&action_list),
            next_observation: stack_vec(&next_obs_list),
            terminal: nd::Array1::<f64>::from_vec(terminal_list),
        };
    }

    pub fn sample_n_trajectories<F>(
        &mut self,
        policy: F,
        max_length: usize,
        ntraj: usize,
    ) -> Vec<Trajectory>
    where
        F: Fn(&nd::Array1<f64>) -> Array1<f64>,
    {
        let mut trajs = vec![];
        let start = SystemTime::now();
        for i in 0..ntraj {
            if i % 100 == 0 {
                println!("elapsed={:?}", start.elapsed());
                println!("{}", i);
            }
            trajs.push(self.sample_trajectory(&policy, max_length));
        }
        return trajs;
    }

    pub fn save_video(&self, filename: &String, image_obs: Vec<nd::Array3<u8>>) {
        let render = self.render.as_ref().unwrap();
        let settings =
            Settings::preset_h264_yuv420p(render.get_width(), render.get_height(), false);
        let duration = Time::from_nth_of_a_second(self.fps);
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
}
