use std::usize;

use burn::backend::ndarray::NdArrayDevice;
use burn::config::Config;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Tensor;
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
        return PolicyGradientAgent {
            actor: policy::MLPPolicyTrainer::new(
                device,
                actor_config,
                AdamConfig::new(),
                self.learning_rate,
            ),
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

impl<B: AutodiffBackend> PolicyGradientAgent<B> {
    pub fn update(
        &mut self,
        obs: &nd::Array3<f64>,       // (B, T, obs_dim)
        actions: &nd::Array3<f64>,   // (B, T, action_dim)
        rewards: &nd::Array2<f64>,   // (B, T)
        terminals: &nd::Array2<f64>, // (B, T)
    ) {
        let q_values = self.calculate_q_vals(&rewards);
        let batch_size = rewards.shape()[0];
        let traj_length = rewards.shape()[1];

        let new_batch_size = batch_size * traj_length;
        let obs = Self::ndarray2tensor3(obs, &self.device).flatten::<2>(0, 1);
        let actions = Self::ndarray2tensor3(actions, &self.device).flatten::<2>(0, 1);
        let rewards = Self::ndarray2tensor2(rewards, &self.device).flatten::<1>(0, 1);
        let terminals = Self::ndarray2tensor2(terminals, &self.device).flatten::<1>(0, 1);
        let q_values = Self::ndarray2tensor2(&q_values, &self.device).flatten::<1>(0, 1);

        let advantages = self.estimate_advantage(&obs, &actions, &q_values, &terminals);
        self.actor
            .train_update(obs.clone(), actions.clone(), advantages.clone());
        for i in 0..self.baseline_gradient_steps {
            self.critic.train_update(obs.clone(), q_values.clone());
        }
    }

    pub fn get_action(&self, obs: &nd::Array1<f64>) -> nd::Array1<f64> {
        let obs = Self::ndarray2tensor1(obs, &self.device).unsqueeze::<2>();
        let action = self.actor.mlp_policy.forward(obs).sample().flatten(0, 1);
        return Self::tensor1ndarray1(&action).mapv(|x| x as f64);
    }

    // return q_values (B, T)
    fn calculate_q_vals(&self, traj_reward_list: &nd::Array2<f64>) -> nd::Array2<f64> {
        let B = traj_reward_list.raw_dim()[0];
        let T = traj_reward_list.raw_dim()[1];
        let mut res = nd::Array2::<f64>::zeros((B, T));
        for b in 0..B {
            res[(b, T - 1)] = traj_reward_list[(b, T - 1)];
            for t in (0..T - 1).rev() {
                res[(b, t)] = traj_reward_list[(b, t)] + self.reward_gamma * res[(b, t + 1)];
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
        let values = self
            .critic
            .mlp_critic
            .forward(obs.clone())
            .flatten::<1>(1, 2); // (B * T)
        let advantages = q_values.clone() - values;
        return (advantages.clone() - advantages.clone().mean().into_scalar())
            / (advantages.var(0).sqrt().into_scalar().to_f64() + 1e-6);
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
        return Tensor::<B, 1>::from_floats(arr.as_slice().unwrap(), device);
    }
    fn ndarray2tensor2(arr: &nd::Array2<f64>, device: &B::Device) -> Tensor<B, 2> {
        return Tensor::<B, 2>::from_floats(arr.as_slice().unwrap(), device);
    }

    fn ndarray2tensor3(arr: &nd::Array3<f64>, device: &B::Device) -> Tensor<B, 3> {
        return Tensor::<B, 3>::from_floats(arr.as_slice().unwrap(), device);
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
