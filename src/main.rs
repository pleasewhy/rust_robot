// #![allow(warnings)]

mod burn_utils;
mod mujoco;
mod pg_run;
mod ppo_run;
mod rl_algorithm;
mod rl_env;

use rl_env::gym_humanoid_v4::HumanoidV4;
use rl_env::inverted_pendulum_v4::InvertedPendulumV4;

fn main() {

    ppo_run::train_network::<HumanoidV4>();
    // pg_run::train_network::<HumanoidV4>();
}
