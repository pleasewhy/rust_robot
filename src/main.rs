// #![allow(warnings)]

mod burn_utils;
mod mujoco;
mod ppo_run;
mod rl_algorithm;
mod rl_env;

use rl_env::gym_humanoid_v4::HumanoidV4;

#[derive(Debug)]
struct Test {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}
impl Default for Test {
    fn default() -> Self {
        Self {
            a: 1,
            b: 2,
            c: 3,
            d: 4,
        }
    }
}
fn main() {
    // let t = Test{
    //     a : 10,
    //     ..Default::default()
    // };
    // println!("t={:?}", t);

    ppo_run::train_network::<HumanoidV4>();
}
