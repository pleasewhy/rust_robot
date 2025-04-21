use std::{
    sync::{mpsc::channel, Arc, Mutex},
    time::SystemTime,
};

use super::{
    env::MujocoEnv,
    nd_vec::{NdVec2, NdVec3},
};

pub struct BatchEnvSample<E: MujocoEnv + Send> {
    batch_size: usize,
    thread_num: usize,
    single_traj_len: usize,
    envs: Vec<Arc<Mutex<E>>>,
    pool: threadpool::ThreadPool,
}

#[derive(Debug)]
pub struct BatchTrajInfo {
    pub observation: NdVec3<f32>,
    pub reward: NdVec2<f32>,
    pub action: NdVec3<f32>,
    pub terminal: NdVec2<u8>,
}

pub struct FlattenBatchTrajInfo {
    pub obs_dim: usize,
    pub action_dim: usize,
    pub obs_vec: Vec<f32>,
    pub action_vec: Vec<f32>,
    pub reward_vec: Vec<f32>,
    pub done_vec: Vec<bool>,
}

impl FlattenBatchTrajInfo {
    pub fn len(&self) -> usize {
        return self.done_vec.len();
    }
}

impl<E: MujocoEnv + Send + 'static> BatchEnvSample<E> {
    pub fn new(single_traj_len: usize, thread_num: usize, envs: Vec<Arc<Mutex<E>>>) -> Self {
        let batch_size = envs.len();
        return Self {
            batch_size,
            thread_num: thread_num,
            single_traj_len: single_traj_len,
            envs: envs,
            pool: threadpool::ThreadPool::new(6),
        };
    }

    fn get_action<F>(&mut self, policy: F, obs_dim: usize) -> NdVec2<f64>
    where
        F: Fn(NdVec2<f64>) -> NdVec2<f64>,
    {
        let mut obs_vec = NdVec2::zeros((self.envs.len(), obs_dim));
        for (idx, env) in self.envs.iter().enumerate() {
            let my_env = env.lock().unwrap();
            if my_env.is_terminated() {
                continue;
            }
            obs_vec
                .row_mut(idx)
                .unwrap()
                .copy_from_slice(my_env.get_obs().as_slice());
        }
        // println!("batch_obs.shape={:?}", batch_obs.shape());
        let actions = policy(obs_vec);
        // println!("actions={:?}", actions);
        return actions;
    }

    fn step_multi_thread(&mut self, step_task_deque: Arc<crossbeam_deque::Injector<StepTask<E>>>) {
        let mut thread_vec = Vec::new();
        let start = SystemTime::now();
        let core_ids = core_affinity::get_core_ids().unwrap();
        let (tx, rx) = channel();
        for i in 0..self.thread_num {
            let tx = tx.clone();
            let my_id = core_ids[i];
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

    pub fn sample_n_trajectories<F>(&mut self, policy: &F) -> FlattenBatchTrajInfo
    where
        F: Fn(NdVec2<f64>) -> NdVec2<f64>,
    {
        for env in &self.envs {
            env.lock().unwrap().reset();
        }
        let start = SystemTime::now();
        let obs_dim = self.envs[0].lock().unwrap().get_obs_dim();
        let action_dim = self.envs[0].lock().unwrap().get_action_dim();
        let batch_traj_info = Arc::new(Mutex::new(BatchTrajInfo {
            observation: NdVec3::<f32>::zeros((self.batch_size, self.single_traj_len, obs_dim)),
            reward: NdVec2::<f32>::zeros((self.batch_size, self.single_traj_len)),
            action: NdVec3::<f32>::zeros((self.batch_size, self.single_traj_len, action_dim)),
            terminal: NdVec2::<u8>::zeros((self.batch_size, self.single_traj_len)),
        }));

        for in_traj_index in 0..self.single_traj_len {
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

            let step_task_deque_len = step_task_deque.len();
            if step_task_deque.is_empty() {
                break;
            }
            self.step_multi_thread(step_task_deque.clone());

            if in_traj_index % 100 == 0 {
                // println!(
                //     "in_traj_index={} step_task_deque.len={} cost={:?}",
                //     in_traj_index,
                //     step_task_deque_len,
                //     start.elapsed()
                // );
            }
        }
        let mut batch_traj = Arc::try_unwrap(batch_traj_info)
            .unwrap()
            .into_inner()
            .unwrap();
        let batch_size = batch_traj.reward.shape()[0];
        let traj_length = batch_traj.reward.shape()[1];
        let num_element = batch_size * traj_length;

        let mut obs_vec = Vec::<f32>::with_capacity(num_element);
        let mut action_vec = Vec::<f32>::with_capacity(num_element);
        let mut reward_vec = Vec::<f32>::with_capacity(num_element);
        let mut done_vec = Vec::<bool>::with_capacity(num_element);
        for b in 0..batch_traj.reward.shape()[0] {
            for t in 0..batch_traj.reward.shape()[1] {
                let shape = (b, t);
                let done = *batch_traj.terminal.get(shape).unwrap() > 0;
                obs_vec.extend_from_slice(batch_traj.observation.slice_mut(shape).unwrap());
                action_vec.extend_from_slice(batch_traj.action.slice_mut(shape).unwrap());
                reward_vec.push(*batch_traj.reward.get(shape).unwrap());
                done_vec.push(done);

                if done {
                    break;
                }
            }
        }
        return FlattenBatchTrajInfo {
            obs_dim,
            action_dim,
            obs_vec,
            action_vec,
            reward_vec,
            done_vec,
        };
    }
}

struct StepTask<E: MujocoEnv> {
    env: Arc<Mutex<E>>,
    env_idx: usize,
    in_traj_index: usize,
    actions: Arc<NdVec2<f64>>,
    batch_traj_info: Arc<Mutex<BatchTrajInfo>>,
}

impl<E: MujocoEnv + Send> StepTask<E> {
    fn run(&mut self) {
        let my_action = self.actions.row(self.env_idx).unwrap();
        let mut my_env = self.env.lock().unwrap();
        let obs = my_env.get_obs();
        let step_info = my_env.step(my_action);
        if step_info.obs.is_empty() {
            return;
        }

        let mut batch_traj_info = self.batch_traj_info.lock().unwrap();

        let now_shape = (self.env_idx, self.in_traj_index);
        assert_eq!(batch_traj_info.action.shape()[2], self.actions.shape()[1]);
        assert_eq!(batch_traj_info.observation.shape()[2], obs.len());
        // println!("step_info={:?}", step_info);
        batch_traj_info
            .observation
            .slice_mut(now_shape)
            .unwrap()
            .copy_from_slice(
                obs.as_slice()
                    .iter()
                    .map(|x| *x as f32)
                    .collect::<Vec<f32>>()
                    .as_slice(),
            );

        batch_traj_info.reward[now_shape] = step_info.reward as f32;
        batch_traj_info
            .action
            .slice_mut(now_shape)
            .unwrap()
            .copy_from_slice(
                my_action
                    .iter()
                    .map(|x| *x as f32)
                    .collect::<Vec<f32>>()
                    .as_slice(),
            );
        batch_traj_info.terminal[now_shape] = step_info.terminated as u8;
    }
}

unsafe impl<E: MujocoEnv> Send for StepTask<E> where E: Send {}
unsafe impl<E: MujocoEnv> Sync for StepTask<E> where E: Sync {}
