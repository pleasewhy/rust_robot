#![allow(warnings)]
#![warn(incomplete_features)]
// #![feature(generic_const_exprs)]

mod burn_utils;
mod mujoco;
mod pg_run;
mod ppo_run;
// mod rnn_ppo_run;
mod rl_algorithm;
mod rl_env;

use burn::prelude::*;
use log::LevelFilter;
use rl_env::gym_humanoid_v4::HumanoidV4;
use rl_env::inverted_pendulum_v4::InvertedPendulumV4;
use rl_env::mobile_arm::MobileArm;

type FType = half::f16;
fn main() {
    // 注意，env_logger 必须尽可能早的初始化
    env_logger::builder().filter_level(LevelFilter::Warn).init();

    ppo_run::train_network::<InvertedPendulumV4>();
}
