#![allow(warnings)]
#![warn(incomplete_features)]
// #![feature(generic_const_exprs)]

mod burn_utils;
mod mujoco;
mod pg_run;
mod ppo_run;
mod rl_algorithm;
mod rl_env;

use burn::{backend::LibTorch, prelude::*};
use rl_env::gym_humanoid_v4::HumanoidV4;
use rl_env::inverted_pendulum_v4::InvertedPendulumV4;
use rl_env::mobile_arm::MobileArm;
fn main() {
    // let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    // let mask: Tensor<LibTorch, 2, Int> =
    //     Tensor::from_ints([[1, 0, 0], [1, 1, 0], [1, 1, 0]], &device);
    // println!("mask={}", mask.clone().repeat_dim(1, 2).reshape([3, 2, 3]).swap_dims(1, 2));
    // println!(
    //     "mask={}",
    //     mask.repeat_dim(1, 2)
    //         .reshape::<3, [usize; 3]>([2, 3, 3])
    //         .swap_dims(0, 2)
    // );
    ppo_run::train_network::<HumanoidV4>();
    // pg_run::train_network::<HumanoidV4>();
}
