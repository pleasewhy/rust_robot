use std::{
    rc::Rc,
    sync::{mpsc::channel, Arc, Mutex},
    time::SystemTime,
};

use crate::mujoco;

use super::env::{self, Env, StepInfo};
use ndarray::{self as nd, Array1, Array2, Array3, ArrayView1, Axis};
use video_rs::ffmpeg::decoder::new;

struct EnvSampler {}

#[derive(Debug)]
pub struct Trajectory {
    pub observation: nd::Array2<f64>,
    pub image_obs: Option<Vec<nd::Array3<u8>>>,
    pub reward: nd::Array1<f64>,
    pub action: nd::Array2<f64>,
    // pub next_observation: nd::Array2<f64>,
    pub terminal: nd::Array1<u8>,
    // pub episode_statistics: nd::Array2<f64>,
}

struct EnvPool {
    envs: Vec<Arc<Mutex<env::Env>>>,
    cnt: usize,
    obs_dim: usize,
    action_dim: usize,
}

impl EnvPool {
    pub fn new(env_size: usize, filename: &str) -> Arc<Mutex<Self>> {
        let mut env_vec = Vec::<Arc<Mutex<env::Env>>>::with_capacity(env_size);
        let model = mujoco::model::Model::from_xml_file(filename).unwrap();
        for i in 0..env_size {
            env_vec.push(Arc::new(Mutex::new(env::Env::from_model(
                model.clone(),
                false,
            ))));
        }
        let obs_dim = env_vec[0].lock().unwrap().get_obs_len();
        let action_dim = env_vec[0].lock().unwrap().get_action_len();
        return Arc::new(Mutex::new(Self {
            envs: env_vec,
            cnt: 0,
            obs_dim,
            action_dim,
        }));
    }
    pub fn reset_handled_cnt(&mut self) {
        self.cnt = 0;
    }
    pub fn reset_all_env(&mut self) {
        for env in &self.envs {
            env.lock().unwrap().reset();
        }
    }
    pub fn get_obs_dim(&self) -> usize {
        return self.obs_dim;
    }
    pub fn get_action_dim(&self) -> usize {
        return self.action_dim;
    }
    pub fn len(&self) -> usize {
        self.envs.len()
    }
    pub fn get_env(&mut self) -> Option<(Arc<Mutex<env::Env>>, usize)> {
        if self.cnt == self.envs.len() {
            return None;
        }
        let env = self.envs[self.cnt].clone();
        self.cnt += 1;
        return Some((env, self.cnt - 1));
    }
}

fn stack_vec(vec: &Vec<nd::Array1<f64>>) -> nd::Array2<f64> {
    return nd::stack(
        nd::Axis(0),
        vec.iter()
            .map(|x: &Array1<f64>| x.view())
            .collect::<Vec<ArrayView1<f64>>>()
            .as_ref(),
    )
    .unwrap();
}

pub struct BatchEnvSample {
    batch_size: usize,
    thread_num: usize,
    single_traj_len: usize,
    env_pool: Arc<Mutex<EnvPool>>,
    pool: threadpool::ThreadPool,
}

impl BatchEnvSample {
    pub fn new(filename: &str, single_traj_len: usize, batch_size: usize) -> Self {
        let env_pool: Arc<Mutex<EnvPool>> = EnvPool::new(batch_size, filename);
        return Self {
            batch_size,
            thread_num: 6,
            single_traj_len: single_traj_len,
            env_pool: env_pool,
            pool: threadpool::ThreadPool::new(6),
        };
    }

    fn get_action<F>(&mut self, policy: F) -> nd::Array2<f64>
    where
        F: Fn(&nd::Array2<f64>) -> Array2<f64>,
    {
        let mut my_env_pool = self.env_pool.lock().unwrap();
        my_env_pool.reset_handled_cnt();
        let mut obs_vec = Vec::<Array1<f64>>::with_capacity(my_env_pool.len());
        loop {
            let my_env = my_env_pool.get_env();
            if my_env.is_none() {
                break;
            }
            let (my_env, env_idx) = my_env.unwrap();
            obs_vec.push(my_env.lock().unwrap().get_obs());
        }
        let batch_obs = stack_vec(&obs_vec);
        // println!("batch_obs.shape={:?}", batch_obs.shape());
        let actions = policy(&batch_obs);
        return actions;
    }

    fn thread_run(
        env_pool: Arc<Mutex<EnvPool>>,
        actions: Arc<nd::Array2<f64>>,
    ) -> Vec<(usize, StepInfo)> {
        let get_env = || {
            let mut pool_guard = env_pool.lock().unwrap();
            return pool_guard.get_env();
        };

        let mut res = Vec::<(usize, StepInfo)>::new();
        loop {
            let my_env = get_env();
            if my_env.is_none() {
                break;
            }
            let (my_env, env_idx) = my_env.unwrap();
            let my_action = actions.row(env_idx);
            let step_info = my_env.lock().unwrap().step(my_action);
            res.push((env_idx, step_info));
        }
        return res;
    }
    fn step_multi_thread(&mut self, actions: Arc<nd::Array2<f64>>) -> Arc<Mutex<Vec<StepInfo>>> {
        self.env_pool.lock().unwrap().reset_handled_cnt();
        let mut thread_vec = Vec::new();
        let start = SystemTime::now();
        let core_ids = core_affinity::get_core_ids().unwrap();
        let step_infos = Arc::new(Mutex::new(Vec::<StepInfo>::new()));
        step_infos
            .lock()
            .unwrap()
            .resize_with(actions.shape()[0], StepInfo::default);
        let (tx, rx) = channel();
        for i in 0..self.thread_num {
            let tx = tx.clone();
            let my_id = core_ids[i];
            let my_actions = actions.clone();
            let my_env_pool = self.env_pool.clone();
            let my_step_infos = step_infos.clone();
            let handle = self.pool.execute(move || {
                // let res = core_affinity::set_for_current(my_id);
                // println!("my_id={:?} init={}", my_id, res);
                let part_step_infos = Self::thread_run(my_env_pool, my_actions);
                let mut my_step_infos_guard = my_step_infos.lock().unwrap();
                for (env_id, step_info) in part_step_infos {
                    // println!("env_id={}", env_id);
                    my_step_infos_guard[env_id] = step_info;
                }
                tx.send(1)
                    .expect("channel will be there waiting for the pool");
            });
            thread_vec.push(handle);
        }
        assert_eq!(
            rx.iter().take(self.thread_num).fold(0, |a, b| a + b),
            self.thread_num
        );
        // println!(
        //     "step {}, thread_num={} env cost={:?}",
        //     self.env_pool.lock().unwrap().len(),
        //     self.thread_num,
        //     start.elapsed()
        // );
        return step_infos;
    }

    pub fn sample_n_trajectories<F>(&mut self, policy: &F) -> Vec<Trajectory>
    where
        F: Fn(&nd::Array2<f64>) -> Array2<f64>,
    {
        let start = SystemTime::now();
        let mut trajs: Vec<Trajectory> = Vec::<Trajectory>::with_capacity(self.batch_size);
        let obs_dim = self.env_pool.lock().unwrap().obs_dim;
        let action_dim = self.env_pool.lock().unwrap().action_dim;
        self.env_pool.lock().unwrap().reset_all_env();

        for _ in 0..self.batch_size {
            let traj = Trajectory {
                observation: nd::Array2::<f64>::zeros((self.single_traj_len, obs_dim)),
                image_obs: None,
                reward: nd::Array1::<f64>::zeros(self.single_traj_len),
                action: nd::Array2::<f64>::zeros((self.single_traj_len, action_dim)),
                // next_observation: nd::Array2::<f64>::zeros((self.single_traj_len, obs_dim)),
                terminal: nd::Array1::<u8>::zeros(self.single_traj_len),
            };
            trajs.push(traj);
        }

        for in_traj_index in 0..self.single_traj_len {
            let actions = Arc::new(self.get_action(policy)); // (Batch_size, action_dim)
            let step_infos = self.step_multi_thread(actions.clone());
            let step_infos_guard = step_infos.lock().unwrap();
            assert_eq!(step_infos_guard.len(), trajs.len());
            let mut terminated_cnt = 0usize;
            for i in 0..trajs.len() {
                let traj = &mut trajs[i];
                let step_info = &step_infos_guard[i];
                // println!("step_info={:?}", step_info);
                if step_info.terminated {
                    terminated_cnt += 1;
                    continue;
                }
                let action = actions.row(i);
                traj.observation
                    .row_mut(in_traj_index)
                    .assign(&step_info.obs);
                // traj.image_obs = ;
                traj.reward[in_traj_index] = step_info.reward;
                traj.action.row_mut(in_traj_index).assign(&action);
                traj.terminal[in_traj_index] = step_info.terminated as u8;
                // traj.next_observation
                //     .row_mut(in_traj_index)
                //     .assign(&step_info.next_obs);
            }
            if in_traj_index % 10 == 0 {
                println!(
                    "in_traj_index={} terminated_cnt={} cost={:?}",
                    in_traj_index,
                    terminated_cnt,
                    start.elapsed()
                );
            }
            if terminated_cnt == trajs.len() {
                println!("in_traj_index={}", in_traj_index);
                break;
            }
        }
        return trajs;
    }
}
