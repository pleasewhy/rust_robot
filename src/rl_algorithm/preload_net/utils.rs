use burn::{
    prelude::Backend,
    tensor::{self, Bool, Int, Tensor, TensorData},
};
use ndarray::{Array1, Array2, Array3};

use crate::rl_algorithm::base::rl_utils::{ndarray2tensor1, tensor2ndarray1};

pub fn generate_flatten_idx<B: Backend>(
    traj_length: Array1<i32>,
    max_traj_len: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let batch_size = traj_length.len();
    let mut mask_arr = Array1::<i32>::zeros(batch_size * max_traj_len);
    let vec = traj_length
        .iter()
        .enumerate()
        .map(|(i, len)| {
            let start = (i * max_traj_len) as i32;
            let end = start as i32 + *len;
            (start..end)
        })
        .flatten()
        .collect::<Vec<i32>>();
    let arr = Array1::from(vec);
    return ndarray2tensor1(arr, device);
}
