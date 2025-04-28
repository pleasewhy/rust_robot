use std::collections::HashMap;

use burn::{
    prelude::Backend,
    tensor::{self, Bool, Int, Tensor, TensorData},
};
use ndarray::{Array1, Array2, Array3};

use crate::burn_utils::randperm;

use super::rl_utils::{self, ndarray2tensor1, ndarray2tensor2, tensor2ndarray2};

fn generate_mask<B: Backend>(
    traj_length: &ndarray::Array1<i32>,
    max_traj_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch_size = traj_length.len();
    let mut mask_arr = Array2::<i32>::zeros([batch_size, max_traj_len]);
    for i in 0..batch_size {
        let traj_len = traj_length[i];
        for j in 0..traj_len {
            mask_arr[[i, j as usize]] = 1;
        }
    }

    return ndarray2tensor2(mask_arr, device);
}

pub struct Memory<B: Backend> {
    obs: Tensor<B, 3>,               // (batch_size, max_traj_len, obs_dim)
    next_obs: Tensor<B, 3>,          // (batch_size, max_traj_len, obs_dim)
    action: Tensor<B, 3>,            // (batch_size, max_traj_len, action_dim)
    reward: Tensor<B, 2>,            // (batch_size, max_traj_len)
    done: Tensor<B, 2, Bool>,        // (batch_size, max_traj_len)
    traj_length: Tensor<B, 1, Int>,  // 实际长度(batch_size)
    seq_mask: Tensor<B, 2>,          // (batch_size, max_traj_len)
    actor_net_mask: Tensor<B, 3>,    // (batch_size, max_traj_len, action_dim)
    baseline_net_mask: Tensor<B, 3>, // (batch_size, max_traj_len, 1)
    batch_size: usize,
    max_traj_len: usize,
    obs_dim: usize,
    action_dim: usize,
    device: B::Device,
}

impl<B: Backend> Memory<B> {
    // pub fn new_by_tensor(
    //     obs: Tensor<B, 3>,
    //     next_obs: Tensor<B, 3>,
    //     traj_length: Tensor<B, 1, Int>,
    //     mask: Tensor<B, 2, Int>,
    //     action: Tensor<B, 3>,
    //     reward: Tensor<B, 2>,
    //     done: Tensor<B, 2, Bool>,
    // ) -> Self {
    //     let dims = obs.shape().dims;
    //     let batch_size = dims[0];
    //     let max_traj_len = dims[1];
    //     let obs_dim = dims[2];
    //     let action_dim = action.shape().dims[2];
    //     let device = obs.device();
    //     Self {
    //         obs,
    //         next_obs,
    //         action,
    //         reward,
    //         traj_length,
    //         mask,
    //         done,
    //         batch_size,
    //         max_traj_len,
    //         obs_dim,
    //         action_dim,
    //         device,
    //     }
    // }
    pub fn new(
        obs_arr: Array3<f32>,
        next_obs_arr: Array3<f32>,
        action_arr: Array3<f32>,
        reward_arr: Array2<f32>,
        done_arr: Array2<bool>,
        traj_length_arr: Array1<i32>,
        device: &B::Device,
    ) -> Self {
        let start = std::time::SystemTime::now();
        let dims = obs_arr.shape();
        let batch_size = dims[0];
        let max_traj_len = dims[1];
        let obs_dim = dims[2];
        let action_dim = action_arr.shape()[2];
        let obs = rl_utils::ndarray2tensor3(obs_arr, device);

        let mut next_obs: Tensor<B, 3> = Tensor::zeros([1, 1, 1], device);
        if next_obs_arr.shape()[0] > 0 {
            next_obs = rl_utils::ndarray2tensor3(next_obs_arr, device);
        }
        let action = rl_utils::ndarray2tensor3(action_arr, device);
        let reward = rl_utils::ndarray2tensor2(reward_arr, device);
        let done = rl_utils::ndarray2tensor2(done_arr, device);

        let seq_mask = generate_mask(&traj_length_arr, max_traj_len, device).float();
        let traj_length = rl_utils::ndarray2tensor1(traj_length_arr, device);
        let actor_net_mask = seq_mask
            .clone()
            .repeat_dim(1, action_dim)
            .reshape([batch_size, action_dim, max_traj_len])
            .swap_dims(1, 2);
        let baseline_net_mask = seq_mask.clone().unsqueeze_dim(2);
        Self {
            obs,
            next_obs,
            action,
            reward,
            done,
            batch_size,
            max_traj_len,
            obs_dim,
            action_dim,
            traj_length,
            seq_mask,
            actor_net_mask,
            baseline_net_mask,
            device: device.clone(),
        }
    }
    pub fn merge(&mut self, other: &Self) {
        self.obs = Tensor::cat(vec![self.obs.clone(), other.obs.clone()], 0);
        self.next_obs = Tensor::cat(vec![self.next_obs.clone(), other.next_obs.clone()], 0);
        self.action = Tensor::cat(vec![self.action.clone(), other.action.clone()], 0);
        self.reward = Tensor::cat(vec![self.reward.clone(), other.reward.clone()], 0);
        self.done = Tensor::cat(vec![self.done.clone(), other.done.clone()], 0);
        self.batch_size += other.batch_size;
    }
    pub fn len(&self) -> usize {
        self.batch_size
    }
    pub fn obs(&self) -> &Tensor<B, 3> {
        &self.obs
    }
    pub fn next_obs(&self) -> &Tensor<B, 3> {
        &self.next_obs
    }
    pub fn action(&self) -> &Tensor<B, 3> {
        &self.action
    }
    pub fn reward(&self) -> &Tensor<B, 2> {
        &self.reward
    }
    pub fn traj_length(&self) -> &Tensor<B, 1, Int> {
        &self.traj_length
    }
    pub fn actor_net_mask(&self) -> &Tensor<B, 3> {
        &self.actor_net_mask
    }
    pub fn baseline_net_mask(&self) -> &Tensor<B, 3> {
        &self.baseline_net_mask
    }
    pub fn seq_mask(&self) -> &Tensor<B, 2> {
        &self.seq_mask
    }
    pub fn done(&self) -> &Tensor<B, 2, Bool> {
        &self.done
    }
    pub fn mini_batch_iter<'a>(
        &'a self,
        num_epoch: usize,
        num_mini_batches: usize,
        extra_mini_batch_2d_tensor: Option<HashMap<&'a str, Tensor<B, 2>>>,
        extra_mini_batch_3d_tensor: Option<HashMap<&'a str, Tensor<B, 3>>>,
    ) -> MiniBatchIter<'a, B> {
        let mini_batch_size = self.len() / num_mini_batches;
        let random_indices_tensor = randperm::<B>(num_mini_batches * mini_batch_size, &self.device);
        let random_indices_tensor: Vec<Tensor<B, 1, Int>> =
            random_indices_tensor.split(mini_batch_size, 0);
        MiniBatchIter {
            memory: self,
            current_step: 0,
            num_epoch,
            num_mini_batches,
            mini_batch_size,
            random_indices_tensor,
            extra_mini_batch_2d_tensor,
            extra_mini_batch_3d_tensor,
        }
    }
}

pub struct MiniBatchIter<'a, B: Backend> {
    memory: &'a Memory<B>,
    current_step: usize,
    num_epoch: usize,
    num_mini_batches: usize,
    mini_batch_size: usize,
    random_indices_tensor: Vec<Tensor<B, 1, Int>>,
    extra_mini_batch_2d_tensor: Option<HashMap<&'a str, Tensor<B, 2>>>,
    extra_mini_batch_3d_tensor: Option<HashMap<&'a str, Tensor<B, 3>>>,
}

impl<'a, B: Backend> Iterator for MiniBatchIter<'a, B> {
    type Item = (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 1, Int>,
        HashMap<&'a str, Tensor<B, 2>>,
        HashMap<&'a str, Tensor<B, 3>>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_step >= self.num_epoch * self.num_mini_batches {
            return None;
        }

        let i = self.current_step % self.num_mini_batches;
        if i == 0 && self.current_step > self.num_mini_batches {
            self.reset_random_indices_tensor();
        }
        let indices_tensor = self.random_indices_tensor[i].clone();

        let obs = self.memory.obs.clone().select(0, indices_tensor.clone());
        let action = self.memory.action.clone().select(0, indices_tensor.clone());
        let traj_length = self
            .memory
            .traj_length
            .clone()
            .select(0, indices_tensor.clone());
        let mut extra_tensor_map_2d = HashMap::new();
        let mut extra_tensor_map_3d = HashMap::new();

        if let Some(extra_mini_batch_2d_tensor) = self.extra_mini_batch_2d_tensor.as_ref() {
            extra_tensor_map_2d = extra_mini_batch_2d_tensor
                .iter()
                .map(|(k, v)| (*k, v.clone().select(0, indices_tensor.clone())))
                .collect();
        }

        if let Some(extra_mini_batch_3d_tensor) = self.extra_mini_batch_3d_tensor.as_ref() {
            extra_tensor_map_3d = extra_mini_batch_3d_tensor
                .iter()
                .map(|(k, v)| (*k, v.clone().select(0, indices_tensor.clone())))
                .collect();
        }

        self.current_step += 1;

        Some((
            obs,
            action,
            traj_length,
            extra_tensor_map_2d,
            extra_tensor_map_3d,
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_epoch * self.num_mini_batches - self.current_step;
        (remaining, Some(remaining))
    }
}

impl<'a, B: Backend> ExactSizeIterator for MiniBatchIter<'a, B> {
    fn len(&self) -> usize {
        self.num_epoch * self.num_mini_batches - self.current_step
    }
}

impl<'a, B: Backend> MiniBatchIter<'a, B> {
    pub fn reset_random_indices_tensor(&mut self) {
        let random_indices_tensor = randperm::<B>(
            self.num_mini_batches * self.mini_batch_size,
            &self.memory.device,
        );
        self.random_indices_tensor = random_indices_tensor.split(self.mini_batch_size, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Data, Int, Tensor};

    #[test]
    fn test_generate_mask() {
        let mut traj_length = ndarray::Array1::<i32>::from_elem(5, 1);
        traj_length[0] = 3;
        traj_length[1] = 3;
        let max_traj_len = 4;
        let device = &NdArrayDevice::Cpu;
        let mask = generate_mask::<NdArray>(&traj_length, max_traj_len, device);
        println!("mask={}", mask);
    }
}
