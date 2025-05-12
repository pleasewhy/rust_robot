use std::{
    sync::{mpsc::channel, Arc, Mutex},
    time::SystemTime,
};

use ndarray::{s, Array1, Array2, Array3};

use super::config::EnvConfig;
use super::env::MujocoEnv;

pub struct BatchEnvSample<E: MujocoEnv + Send> {
    batch_size: usize,
    thread_num: usize,
    max_traj_len: usize,
    envs: Vec<Arc<Mutex<E>>>,
    pool: threadpool::ThreadPool,
    env_config: EnvConfig,
}

#[derive(Debug)]
pub struct BatchTrajInfo {
    pub obs: Array3<f32>,
    pub next_obs: Array3<f32>,
    pub reward: Array2<f32>,
    pub action: Array3<f32>,
    pub terminal: Array2<u8>,
    pub traj_length: Array1<i32>,
}

impl BatchTrajInfo {
    pub fn truncate(&self) -> BatchTrajInfo {
        let new_batch_traj_length = *self.traj_length.iter().max().unwrap() as usize;
        let index_3d = s![.., ..new_batch_traj_length, ..];
        let index_2d = s![.., ..new_batch_traj_length];
        let obs = self.obs.slice(index_3d).to_owned();
        let next_obs = self.next_obs.slice(index_3d).to_owned();
        let action = self.action.slice(index_3d).to_owned();
        let reward = self.reward.slice(index_2d).to_owned();
        let terminal = self.terminal.slice(index_2d).to_owned();
        return Self {
            obs,
            next_obs,
            reward,
            action,
            terminal,
            traj_length: self.traj_length.clone(),
        };
    }
}

// pub struct FlattenBatchTrajInfo {
//     pub obs_dim: usize,
//     pub action_dim: usize,
//     pub obs_vec: Vec<f32>,
//     pub next_obs_vec: Vec<f32>,
//     pub action_vec: Vec<f32>,
//     pub reward_vec: Vec<f32>,
//     pub done_vec: Vec<bool>,
// }

// impl FlattenBatchTrajInfo {
//     pub fn len(&self) -> usize {
//         return self.done_vec.len();
//     }
// }

impl<E: MujocoEnv + Send + 'static> BatchEnvSample<E> {
    pub fn new(env_config: EnvConfig, envs: Vec<Arc<Mutex<E>>>) -> Self {
        let batch_size = envs.len();
        return Self {
            batch_size,
            thread_num: env_config.sample_thread_num,
            max_traj_len: env_config.max_traj_length,
            envs: envs,
            pool: threadpool::ThreadPool::new(env_config.sample_thread_num),
            env_config,
        };
    }

    fn get_action<F>(&mut self, policy: F, obs_dim: usize) -> Array2<f64>
    where
        F: Fn(Array2<f64>) -> Array2<f64>,
    {
        let mut obs_vec = Array2::zeros((self.envs.len(), obs_dim));
        for (idx, env) in self.envs.iter().enumerate() {
            let my_env = env.lock().unwrap();
            if my_env.is_terminated() {
                continue;
            }
            obs_vec
                .row_mut(idx)
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(my_env.get_obs().as_slice());
        }
        let actions = policy(obs_vec);
        return actions;
    }

    fn step_multi_thread(&mut self, step_task_deque: Arc<crossbeam_deque::Injector<StepTask<E>>>) {
        let mut thread_vec = Vec::new();
        // let start = SystemTime::now();
        // let core_ids = core_affinity::get_core_ids().unwrap();
        let (tx, rx) = channel();
        for _ in 0..self.thread_num {
            let tx = tx.clone();
            // let my_id = core_ids[i];
            let my_step_task_deque = step_task_deque.clone();
            let handle = self.pool.execute(move || {
                // let res = core_affinity::set_for_current(my_id);
                // println!("my_id={:?} init={}", my_id, res);
                while !my_step_task_deque.is_empty() {
                    let steal_task = my_step_task_deque.steal();
                    if !steal_task.is_success() {
                        break;
                    }
                    let mut task = steal_task.success().unwrap();
                    task.run();
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
    }

    pub fn sample_n_trajectories<F>(&mut self, policy: &F) -> BatchTrajInfo
    where
        F: Fn(Array2<f64>) -> Array2<f64>,
    {
        for env in &self.envs {
            env.lock().unwrap().reset();
        }
        let start = SystemTime::now();
        let obs_dim = self.envs[0].lock().unwrap().get_obs_dim();
        let action_dim = self.envs[0].lock().unwrap().get_action_dim();
        let start2 = SystemTime::now();
        let batch_traj_info = Arc::new(Mutex::new(BatchTrajInfo {
            obs: Array3::<f32>::zeros((self.batch_size, self.max_traj_len, obs_dim)),
            next_obs: Array3::<f32>::zeros((self.batch_size, self.max_traj_len, obs_dim)),
            reward: Array2::<f32>::zeros((self.batch_size, self.max_traj_len)),
            action: Array3::<f32>::zeros((self.batch_size, self.max_traj_len, action_dim)),
            terminal: Array2::<u8>::ones((self.batch_size, self.max_traj_len)),
            traj_length: Array1::<i32>::ones(self.batch_size),
        }));
        println!("new batch_traj_info cost={:?}", start2.elapsed());

        for in_traj_index in 0..self.max_traj_len {
            let actions = Arc::new(self.get_action(policy, obs_dim));
            let step_task_deque = Arc::new(crossbeam_deque::Injector::<StepTask<E>>::new());

            for (idx, env) in self.envs.iter().enumerate() {
                if env.lock().unwrap().is_terminated() {
                    continue;
                }
                step_task_deque.push(StepTask {
                    env: env.clone(),
                    env_idx: idx,
                    in_traj_index: in_traj_index,
                    actions: actions.clone(),
                    batch_traj_info: batch_traj_info.clone(),
                });
            }
            if step_task_deque.is_empty() {
                break;
            }
            self.step_multi_thread(step_task_deque.clone());
        }
        let start1 = SystemTime::now();
        let batch_traj = batch_traj_info.lock().unwrap().truncate();
        println!("truncate cost={:?}", start1.elapsed());
        // let batch_size = batch_traj.reward.shape()[0];
        // let traj_length = batch_traj.reward.shape()[1];
        // let num_element = batch_size * traj_length;

        // let mut obs_vec = Vec::<f32>::with_capacity(num_element);
        // let mut next_obs_vec = Vec::<f32>::with_capacity(num_element);
        // let mut action_vec = Vec::<f32>::with_capacity(num_element);
        // let mut reward_vec = Vec::<f32>::with_capacity(num_element);
        // let mut done_vec = Vec::<bool>::with_capacity(num_element);
        // for b in 0..batch_traj.reward.shape()[0] {
        //     for t in 0..batch_traj.reward.shape()[1] {
        //         let shape = s![b, t, ..];
        //         let done = *batch_traj.terminal.get((b, t)).unwrap() > 0 || t == traj_length - 1;
        //         obs_vec.extend_from_slice(batch_traj.obs.slice_mut(shape).as_slice().unwrap());
        //         next_obs_vec
        //             .extend_from_slice(batch_traj.next_obs.slice_mut(shape).as_slice().unwrap());
        //         action_vec
        //             .extend_from_slice(batch_traj.action.slice_mut(shape).as_slice().unwrap());
        //         reward_vec.push(*batch_traj.reward.get((b, t)).unwrap());
        //         done_vec.push(done);

        //         if done {
        //             break;
        //         }
        //     }
        // }
        // return FlattenBatchTrajInfo {
        //     obs_dim,
        //     action_dim,
        //     obs_vec,
        //     next_obs_vec,
        //     action_vec,
        //     reward_vec,
        //     done_vec,
        // };
        println!(
            "collect batch_traj_info finish shape={:?} cost={:?}",
            batch_traj.obs.shape(),
            start.elapsed()
        );
        return batch_traj;
    }
}

struct StepTask<E: MujocoEnv> {
    env: Arc<Mutex<E>>,
    env_idx: usize,
    in_traj_index: usize,
    actions: Arc<Array2<f64>>,
    batch_traj_info: Arc<Mutex<BatchTrajInfo>>,
}

impl<E: MujocoEnv + Send> StepTask<E> {
    fn run(&mut self) {
        let my_action = self.actions.row(self.env_idx);
        let my_action = my_action.as_slice().unwrap();
        let mut my_env = self.env.lock().unwrap();
        let obs = my_env.get_obs();
        let step_info = my_env.step(my_action);
        if step_info.obs.is_empty() {
            return;
        }

        let mut batch_traj_info = self.batch_traj_info.lock().unwrap();

        let get_index = (self.env_idx, self.in_traj_index);
        let slice_index = s![self.env_idx, self.in_traj_index, ..];
        assert_eq!(batch_traj_info.action.shape()[2], self.actions.shape()[1]);
        assert_eq!(batch_traj_info.obs.shape()[2], obs.len());
        // println!("step_info={:?}", step_info);
        batch_traj_info
            .obs
            .slice_mut(slice_index)
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(
                obs.as_slice()
                    .iter()
                    .map(|x| *x as f32)
                    .collect::<Vec<f32>>()
                    .as_slice(),
            );

        batch_traj_info
            .next_obs
            .slice_mut(slice_index)
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(
                step_info
                    .obs
                    .as_slice()
                    .iter()
                    .map(|x| *x as f32)
                    .collect::<Vec<f32>>()
                    .as_slice(),
            );

        batch_traj_info.reward[get_index] = step_info.reward as f32;
        batch_traj_info
            .action
            .slice_mut(slice_index)
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(
                my_action
                    .iter()
                    .map(|x| *x as f32)
                    .collect::<Vec<f32>>()
                    .as_slice(),
            );
        batch_traj_info.terminal[get_index] = step_info.terminated as u8;
        batch_traj_info.traj_length[self.env_idx] = (self.in_traj_index + 1) as i32;
    }
}

unsafe impl<E: MujocoEnv> Send for StepTask<E> where E: Send {}
unsafe impl<E: MujocoEnv> Sync for StepTask<E> where E: Sync {}

pub fn create_n_env<ENV: MujocoEnv>(env_config: EnvConfig) -> Vec<Arc<Mutex<ENV>>> {
    let mut envs = vec![];
    for _ in 0..env_config.n_env {
        envs.push(Arc::new(Mutex::new(ENV::new(false, env_config.clone()))));
    }
    return envs;
}
