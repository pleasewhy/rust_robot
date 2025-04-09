use std::collections::HashMap;

use burn::{
    prelude::Backend,
    tensor::{Bool, Int, Tensor, TensorData},
};

use crate::burn_utils::randperm;

use super::rl_utils;

pub struct Memory<B: Backend> {
    obs: Tensor<B, 2>,
    action: Tensor<B, 2>,
    reward: Tensor<B, 1>,
    done: Tensor<B, 1, Bool>,
    len: usize,
    device: B::Device,
}

impl<B: Backend> Memory<B> {
    pub fn new(
        obs_vec: Vec<f32>,
        action_vec: Vec<f32>,
        reward_vec: Vec<f32>,
        done_vec: Vec<bool>,
        device: &B::Device,
    ) -> Self {
        let batch = done_vec.len();
        let obs_dim = obs_vec.len() / batch;
        let action_dim = action_vec.len() / batch;
        let obs = Tensor::from_data(TensorData::new(obs_vec, [batch, obs_dim]), device);
        let action = Tensor::from_data(TensorData::new(action_vec, [batch, action_dim]), device);
        let reward = Tensor::from_data(TensorData::new(reward_vec, [batch]), device);
        let done = Tensor::from_data(TensorData::new(done_vec, [batch]), device);
        Self {
            obs,
            action,
            reward,
            done,
            len: batch,
            device: device.clone(),
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn obs(&self) -> &Tensor<B, 2> {
        &self.obs
    }
    pub fn action(&self) -> &Tensor<B, 2> {
        &self.action
    }
    pub fn reward(&self) -> &Tensor<B, 1> {
        &self.reward
    }
    pub fn done(&self) -> &Tensor<B, 1, Bool> {
        &self.done
    }
    pub fn mini_batch_iter<'a>(
        &'a self,
        num_epoch: usize,
        num_mini_batches: usize,
        extra_mini_batch_tensor: Option<HashMap<&'a str, Tensor<B, 1>>>,
    ) -> MiniBatchIter<'a, B> {
        let mini_batch_size = self.len() / num_mini_batches;
        let random_indices_tensor = randperm::<B>(num_mini_batches * mini_batch_size, &self.device);
        let random_indices_tensor = random_indices_tensor.split(mini_batch_size, 0);
        MiniBatchIter {
            memory: self,
            current_step: 0,
            num_epoch,
            num_mini_batches,
            mini_batch_size,
            random_indices_tensor,
            extra_mini_batch_tensor,
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
    extra_mini_batch_tensor: Option<HashMap<&'a str, Tensor<B, 1>>>,
}

impl<'a, B: Backend> Iterator for MiniBatchIter<'a, B> {
    type Item = (Tensor<B, 2>, Tensor<B, 2>, HashMap<&'a str, Tensor<B, 1>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_step >= self.num_epoch * self.num_mini_batches {
            return None;
        }

        let i = self.current_step % self.num_mini_batches;

        let indices_tensor = self.random_indices_tensor[i].clone();

        let obs = self.memory.obs.clone().select(0, indices_tensor.clone());
        let action = self.memory.action.clone().select(0, indices_tensor.clone());
        let mut extra_tensor_map = HashMap::new();

        if let Some(extra_mini_batch_tensor) = self.extra_mini_batch_tensor.as_ref() {
            extra_tensor_map = extra_mini_batch_tensor
                .iter()
                .map(|(k, v)| (k.clone(), v.clone().select(0, indices_tensor.clone())))
                .collect();
        }

        self.current_step += 1;

        Some((obs, action, extra_tensor_map))
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use burn::backend::ndarray::{NdArray, NdArrayDevice};
//     use burn::tensor::{Data, Int, Tensor};

//     #[test]
//     fn test_memory_initialization() {
//         let device = Default::default();
//         let obs = vec![1.0, 2.0, 3.0, 4.0];
//         let actions = vec![0.0, 1.0];
//         let rewards = vec![1.0, 0.5];
//         let dones = vec![false, true];

//         let memory = Memory::<NdArray>::new(
//             obs.clone(),
//             actions.clone(),
//             rewards.clone(),
//             dones.clone(),
//             &device,
//         );

//         assert_eq!(memory.len(), 2);
//         assert_eq!(memory.obs().shape(), [2, 2].into());
//         assert_eq!(memory.action().shape(), [2, 1].into());
//         assert_eq!(memory.reward().shape(), [2].into());
//         assert_eq!(memory.done().shape(), [2].into());
//     }

//     #[test]
//     fn test_mini_batch_iterator() {
//         let device = Default::default();
//         let memory = Memory::<NdArray>::new(
//             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//             vec![false, false, true, false, false, true],
//             &device,
//         );

//         let t1 = Tensor::zeros([6], &NdArrayDevice::Cpu);
//         let t2 = Tensor::zeros([6], &NdArrayDevice::Cpu);
//         let mut iter = memory.mini_batch_iter(2, 3, t1, t2);
//         assert_eq!(iter.len(), 6);

//         let mut count = 0;
//         while let Some((obs, action, probs, values)) = iter.next() {
//             assert_eq!(obs.shape().dims[0], 2);
//             assert_eq!(action.shape().dims[0], 2);
//             assert_eq!(probs.shape().dims[0], 2);
//             assert_eq!(values.shape().dims[0], 2);
//             count += 1;
//         }
//         assert_eq!(count, 6);
//     }

//     // #[test]
//     // fn test_done_tensor_conversion() {
//     //     let device = Default::default();
//     //     let memory = Memory::<NdArray>::new(vec![1.0], vec![0.0], vec![1.0], vec![true], &device);

//     //     let t1 = Tensor::zeros([6], &NdArrayDevice::Cpu);
//     //     let t2 = Tensor::zeros([6], &NdArrayDevice::Cpu);
//     //     let mut iter = memory.mini_batch_iter(1, 1, t1, t2);
//     //     let (_, _, _, _) = iter.next().unwrap();

//     //     assert_eq!(done.to_data(), Data::from([true]));
//     // }
// }
