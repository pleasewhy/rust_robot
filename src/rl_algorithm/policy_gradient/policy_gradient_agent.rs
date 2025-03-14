use std::time::SystemTime;
use std::usize;

use burn::config::Config;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Element;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use num_traits::ToPrimitive;
use num_traits::Zero;

use crate::rl_env::nd_vec::{NdVec2, NdVec3};

use super::critic;
use super::policy;

#[derive(Config, Debug)]
pub struct PolicyGradientAgentConfig {
    observation_dim: usize,
    action_dim: usize,
    discrete: bool,
    n_layers: usize,
    layer_size: usize,
    learning_rate: f64,
    reward_gamma: f32,
    baseline_gradient_steps: usize,
    gae_gamma: f32,
}

impl PolicyGradientAgentConfig {
    pub fn init<B: AutodiffBackend>(&self, device: &B::Device) -> PolicyGradientAgent<B> {
        let actor_config = policy::MLPPolicyConfig::new(
            self.action_dim,
            self.observation_dim,
            self.discrete,
            self.n_layers,
            self.layer_size,
        );
        let critic_config =
            critic::MLPCriticConfig::new(self.observation_dim, self.n_layers, self.layer_size);
        let actor: policy::MLPPolicyTrainer<B> = policy::MLPPolicyTrainer::new(
            device,
            actor_config,
            AdamConfig::new(),
            self.learning_rate,
        );
        let critic = critic::MLPCriticTrainer::new(
            device,
            critic_config,
            AdamConfig::new(),
            self.learning_rate,
        );
        println!("actor={}", actor.mlp_policy);
        println!("critic={}", critic.mlp_critic);
        return PolicyGradientAgent {
            actor,
            critic,
            reward_gamma: self.reward_gamma,
            device: device.clone(),
            baseline_gradient_steps: self.baseline_gradient_steps,
            gae_gamma: self.gae_gamma,
        };
    }
}

pub struct PolicyGradientAgent<B: AutodiffBackend> {
    actor: policy::MLPPolicyTrainer<B>,
    critic: critic::MLPCriticTrainer<B>,
    reward_gamma: f32,
    device: B::Device,
    baseline_gradient_steps: usize,
    gae_gamma: f32,
}

#[derive(Debug)]
pub struct UpdateInfo {
    pub actor_loss: f32,
    pub critic_loss: f32,
    pub mean_q_val: f32,
    pub mean_reward: f32,
    pub batch_size: usize,
}

unsafe impl<B: AutodiffBackend> Send for PolicyGradientAgent<B> {}
unsafe impl<B: AutodiffBackend> Sync for PolicyGradientAgent<B> {}

fn get_indices<B: Backend>(obs: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 1, Int> {
    let mask = obs
        .sum_dim(1)
        .flatten::<1>(0, 1)
        .not_equal_elem(0)
        .into_data();
    let mask_slice = mask.as_slice::<bool>().unwrap();
    let indices = mask_slice
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i as u32)
        .collect::<Vec<u32>>();
    let shape = indices.len();
    let tensor_data = TensorData::new(indices, [shape]);
    return Tensor::from_data(tensor_data, device);
}
impl<B: AutodiffBackend> PolicyGradientAgent<B> {
    pub fn update(
        &mut self,
        obs: NdVec3<f32>,      // (B, T, obs_dim)
        actions: NdVec3<f32>,  // (B, T, action_dim)
        rewards: NdVec2<f32>,  // (B, T)
        terminals: NdVec2<u8>, // (B, T)
    ) -> UpdateInfo {
        let start = SystemTime::now();
        let batch_size = obs.shape()[0];
        let q_values = self.calculate_q_vals(&rewards);
        let advantages = self
            .estimate_advantage(&obs, &q_values, &rewards, &terminals)
            .flatten::<1>(0, 1);
        self.actor.mlp_policy.clone().into_record();
        let obs = Self::vec2tensor3(obs, &self.device).flatten::<2>(0, 1);
        let actions = Self::vec2tensor3(actions, &self.device).flatten::<2>(0, 1);
        let rewards = Self::vec2tensor2(rewards, &self.device).flatten::<1>(0, 1);
        let q_values = Self::vec2tensor2(q_values, &self.device).flatten::<1>(0, 1);

        let mean_q_val = q_values.clone().mean().into_scalar().to_f32();
        let mean_reward = rewards.clone().sum().into_scalar().to_f32() / batch_size as f32;
        let len = obs.shape().dims[0];
        let indices = get_indices(obs.clone(), &self.device);
        let obs = obs.select(0, indices.clone());
        let actions = actions.select(0, indices.clone());
        // let rewards = rewards.select(0, indices.clone());
        let q_values = q_values.select(0, indices.clone());
        let advantages = advantages.select(0, indices.clone());
        println!("remove zeros row cnt={}", len - obs.shape().dims[0]);

        let mut update_info = UpdateInfo {
            actor_loss: 0.0,
            critic_loss: 0.0,
            mean_q_val: mean_q_val as f32,
            mean_reward: mean_reward as f32,
            batch_size: obs.shape().dims[0],
        };
        update_info.actor_loss =
            self.actor
                .train_update(obs.clone(), actions.clone(), advantages.clone());
        // println!("update_actor={:?}", start.elapsed());
        for i in 0..self.baseline_gradient_steps {
            update_info.critic_loss += self.critic.train_update(obs.clone(), q_values.clone());
        }
        println!("update_critic cost={:?}", start.elapsed());
        update_info.critic_loss = update_info.critic_loss / (self.baseline_gradient_steps as f32);
        return update_info;
    }

    pub fn get_action(&self, obs: NdVec2<f64>) -> NdVec2<f64> {
        // println!("obs.shape={:?}", obs.shape());
        let obs = Self::vec2tensor2(obs, &self.device);
        // println!("obs.shape={:?}", obs.shape());
        let action = self.actor.mlp_policy.forward(obs).sample();
        return Self::tensor2vec2(&action).to_f64();
    }

    // return q_values (B, T)
    fn calculate_q_vals(&self, traj_reward_list: &NdVec2<f32>) -> NdVec2<f32> {
        let B = traj_reward_list.shape()[0];
        let T = traj_reward_list.shape()[1];
        let mut res = NdVec2::<f32>::zeros((B, T));
        for b in 0..B {
            res[(b, T - 1)] = traj_reward_list[(b, T - 1)];
            for t in (0..T - 1).rev() {
                res[(b, t)] = (traj_reward_list[(b, t)] + self.reward_gamma * res[(b, t + 1)]);
            }
        }
        // println!("traj_reward_list={:?}", traj_reward_list);
        // println!("q_vals={:?}", res);
        return res;
    }

    fn estimate_advantage(
        &self,
        obs: &NdVec3<f32>, // (B, T, obs_dim)
        // actions: &Tensor<B, 2>,   // (B, T, action_dim)
        q_values: &NdVec2<f32>, // (B, T)
        rewards: &NdVec2<f32>,  // (B, T)
        terminals: &NdVec2<u8>, // (B, T)
    ) -> Tensor<B, 2> {
        let values = self
            .critic
            .mlp_critic
            .forward(Self::vec2tensor3(obs.clone(), &self.device))
            .flatten::<2>(1, 2);
        // return Self::vec2tensor2(q_values.clone(), &self.device) - values;
        let values = Self::tensor2vec2(&values); // (B * T)

        // implement GAE
        let batch_size = obs.shape()[0];
        let traj_length = obs.shape()[1];
        println!("batch_size={} traj_length={}", batch_size, traj_length);
        // HINT: append a dummy T+1 value for simpler recursive calculation
        // values.push(0f32);
        let mut advantages = NdVec2::<f32>::zeros((values.shape()[0], values.shape()[1]));
        // println!("advantages_size={:?}", advantages.shape());
        // println!("values={:?}", values.shape());
        // println!("q_values={:?}", q_values.shape());
        // println!("terminals={:?}", terminals.shape());
        // println!(
        //     "reward_gamma={:?} gae_gamma={}",
        //     self.reward_gamma, self.gae_gamma
        // );
        for b in 0..batch_size {
            for t in (0..traj_length).rev() {
                let sigma = rewards[(b, t)]
                    + (1 - terminals[(b, t)]) as f32
                        * self.reward_gamma
                        * values.get((b, t + 1)).unwrap_or(&0f32)
                    - values[(b, t)];
                advantages[(b, t)] = sigma
                    + self.reward_gamma
                        * self.gae_gamma
                        * advantages.get((b, t + 1)).unwrap_or(&0f32);
            }
        }

        let advantages = Self::vec2tensor2(advantages, &self.device);
        return (advantages.clone() - advantages.clone().mean().into_scalar())
            / (advantages
                .flatten::<1>(0, 1)
                .var(0)
                .sqrt()
                .into_scalar()
                .to_f64()
                + 1e-6);
        // return advantages;
    }

    fn vec2tensor2<T: Element + Zero + ToPrimitive>(
        arr: NdVec2<T>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let shape = arr.shape();
        let tensor_data = TensorData::new(arr.into_vec(), shape);
        return Tensor::<B, 2>::from_data(tensor_data, device);
    }

    fn vec2tensor3<T: Element + Zero + ToPrimitive>(
        arr: NdVec3<T>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let shape: [usize; 3] = arr.shape();
        let tensor_data = TensorData::new(arr.into_vec(), shape);
        return Tensor::<B, 3>::from_data(tensor_data, device);
    }

    fn tensor2vec2(tensor: &Tensor<B, 2>) -> NdVec2<f32> {
        let vec = tensor.to_data().into_vec::<f32>().unwrap();
        return NdVec2::from_shape_vec(vec, tensor.shape().dims());
    }
    fn tensor2vec1(tensor: &Tensor<B, 1>) -> Vec<f32> {
        let vec = tensor.to_data().into_vec::<f32>().unwrap();
        return vec;
    }
}
