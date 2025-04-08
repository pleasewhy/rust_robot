use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::rng;
use rand::seq::SliceRandom;

pub fn randperm<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    // 生成 0..n 的序列
    let mut indices: Vec<i32> = (0..n as i32).collect();

    // 使用线程安全的随机数生成器打乱
    indices.shuffle(&mut rng());

    let shape = [indices.len()];
    let tensor_data = TensorData::new(indices, shape);
    // 转换为 Burn 张量
    Tensor::<B, 1, Int>::from_data(tensor_data, device)
}
