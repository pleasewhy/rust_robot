use std::time::SystemTime;
use std::usize;

use burn::backend::ndarray::NdArrayDevice;
use burn::config::Config;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Element;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use ndarray as nd;
use ndarray::Array2;
use ndarray::Dim;
use ndarray::Ix;

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
    reward_gamma: f64,
    baseline_gradient_steps: usize,
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
        println!("actor={}", actor.mlp_policy);
        return PolicyGradientAgent {
            actor,
            critic: critic::MLPCriticTrainer::new(
                device,
                critic_config,
                AdamConfig::new(),
                self.learning_rate,
            ),
            reward_gamma: self.reward_gamma,
            device: device.clone(),
            baseline_gradient_steps: self.baseline_gradient_steps,
        };
    }
}

pub struct PolicyGradientAgent<B: AutodiffBackend> {
    actor: policy::MLPPolicyTrainer<B>,
    critic: critic::MLPCriticTrainer<B>,
    reward_gamma: f64,
    device: B::Device,
    baseline_gradient_steps: usize,
}

#[derive(Debug)]
pub struct UpdateInfo {
    actor_loss: f32,
    critic_loss: f32,
    mean_q_val: f32,
    mean_reward: f32,
}

unsafe impl<B: AutodiffBackend> Send for PolicyGradientAgent<B> {}
unsafe impl<B: AutodiffBackend> Sync for PolicyGradientAgent<B> {}

impl<B: AutodiffBackend> PolicyGradientAgent<B> {
    pub fn update(
        &mut self,
        obs: &nd::Array3<f64>,      // (B, T, obs_dim)
        actions: &nd::Array3<f64>,  // (B, T, action_dim)
        rewards: &nd::Array2<f64>,  // (B, T)
        terminals: &nd::Array2<u8>, // (B, T)
    ) -> UpdateInfo {
        let start = SystemTime::now();
        let q_values = self.calculate_q_vals(&rewards);
        // println!("rewards={}", rewards.row(0 as usize));
        // println!("q_values={}", q_values.row(0 as usize));
        println!("t1={:?}", start.elapsed());
        let mean_q_val = q_values.mean().unwrap();
        let mean_reward = rewards.mean().unwrap();
        let batch_size = rewards.shape()[0];
        let traj_length = rewards.shape()[1];
        println!("t2={:?}", start.elapsed());
        let new_batch_size = batch_size * traj_length;
        let obs = Self::ndarray2tensor3(obs, &self.device).flatten::<2>(0, 1);
        let actions = Self::ndarray2tensor3(actions, &self.device).flatten::<2>(0, 1);
        let rewards = Self::ndarray2tensor2(rewards, &self.device).flatten::<1>(0, 1);
        let terminals = Self::ndarray2tensor2(terminals, &self.device).flatten::<1>(0, 1);
        let q_values = Self::ndarray2tensor2(&q_values, &self.device).flatten::<1>(0, 1);
        println!("t3={:?}", start.elapsed());
        let advantages = self.estimate_advantage(&obs, &actions, &q_values, &terminals);
        println!("t4={:?}", start.elapsed());
        let mut update_info = UpdateInfo {
            actor_loss: 0.0,
            critic_loss: 0.0,
            mean_q_val: mean_q_val as f32,
            mean_reward: mean_reward as f32,
        };
        update_info.actor_loss =
            self.actor
                .train_update(obs.clone(), actions.clone(), advantages.clone());
        println!("update_actor={:?}", start.elapsed());
        // for i in 0..self.baseline_gradient_steps {
        //     update_info.critic_loss += self.critic.train_update(obs.clone(), q_values.clone());
            // println!("update_critic it={} cost={:?}", i, start.elapsed());
        // }
        // update_info.critic_loss = update_info.critic_loss / (self.baseline_gradient_steps as f32);
        return update_info;
    }

    pub fn get_action(&self, obs: &nd::Array2<f64>) -> nd::Array2<f64> {
        // println!("obs.shape={:?}", obs.shape());
        let obs = Self::ndarray2tensor2(obs, &self.device);
        // println!("obs.shape={:?}", obs.shape());
        let action = self.actor.mlp_policy.forward(obs).sample();
        return Self::tensor2ndarray2(&action).mapv(|x| x as f64);
    }

    // return q_values (B, T)
    fn calculate_q_vals(&self, traj_reward_list: &nd::Array2<f64>) -> nd::Array2<f64> {
        let B = traj_reward_list.raw_dim()[0];
        let T = traj_reward_list.raw_dim()[1];
        let mut res = nd::Array2::<f64>::zeros((B, T));
        for b in 0..B {
            res[(b, T - 1)] = traj_reward_list[(b, T - 1)];
            for t in (0..T - 1).rev() {
                res[(b, t)] = (0.5 * traj_reward_list[(b, t)] + 0.5 * res[(b, t + 1)]);
            }
        }
        return res;
    }

    fn estimate_advantage(
        &self,
        obs: &Tensor<B, 2>,       // (B, T, obs_dim)
        actions: &Tensor<B, 2>,   // (B, T, action_dim)
        q_values: &Tensor<B, 1>,  // (B, T)
        terminals: &Tensor<B, 1>, // (B, T)
    ) -> Tensor<B, 1> {
        return q_values.clone();
        // let values = self
        //     .critic
        //     .mlp_critic
        //     .forward(obs.clone())
        //     .flatten::<1>(0, 1); // (B * T)
        // let advantages = q_values.clone() - values;
        // return (advantages.clone() - advantages.clone().mean().into_scalar())
        //     / (advantages.var(0).sqrt().into_scalar().to_f64() + 1e-6);
    }

    fn var<const D: usize>(tensor: Tensor<B, D>) -> f64 {
        let mean = tensor.clone().mean().into_scalar();
        let n = tensor.shape().num_elements() as f64;
        let x: f64 = (tensor.clone() - mean)
            .powf_scalar(2.0)
            .sum()
            .into_scalar()
            .to_f64();
        return x / n;
    }
    fn std<const D: usize>(tensor: Tensor<B, D>) -> f64 {
        return Self::var(tensor).sqrt();
    }
    fn ndarray2tensor1(arr: &nd::Array1<f64>, device: &B::Device) -> Tensor<B, 1> {
        let vec = arr.as_slice().unwrap().to_vec();
        let tensor_data = TensorData::new(vec, arr.shape());
        return Tensor::<B, 1>::from_data(tensor_data, device);
    }
    fn ndarray2tensor2<T: Element>(arr: &nd::Array2<T>, device: &B::Device) -> Tensor<B, 2> {
        let vec = arr.as_slice().unwrap().to_vec();
        let tensor_data = TensorData::new(vec, arr.shape());
        return Tensor::<B, 2>::from_data(tensor_data, device);
    }

    fn ndarray2tensor3(arr: &nd::Array3<f64>, device: &B::Device) -> Tensor<B, 3> {
        let vec = arr.as_slice().unwrap().to_vec();
        let tensor_data = TensorData::new(vec, arr.shape());
        return Tensor::<B, 3>::from_data(tensor_data, device);
    }

    fn tensor1ndarray1(tensor: &Tensor<B, 1>) -> nd::Array1<f32> {
        let t = tensor.to_data().into_vec::<f32>().unwrap();
        return nd::Array1::from_shape_vec::<[usize; 1]>(tensor.shape().dims(), t).unwrap();
    }
    fn tensor2ndarray2(tensor: &Tensor<B, 2>) -> nd::Array2<f32> {
        let t = tensor.to_data().into_vec::<f32>().unwrap();
        return nd::Array2::from_shape_vec::<[usize; 2]>(tensor.shape().dims(), t).unwrap();
    }
}
