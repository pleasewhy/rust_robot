#![allow(warnings)]
#![warn(incomplete_features)]
#![recursion_limit = "256"]

mod burn_utils;
mod mujoco;
mod pg_run;
mod ppo_run;
// mod rnn_ppo_run;
mod rl_algorithm;
mod rl_env;

use std::marker::PhantomData;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{self, Autodiff, Wgpu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, HalfPrecisionSettings, PrecisionSettings};
use burn::serde::{de::DeserializeOwned, Serialize};
use burn::tensor::TensorPrimitive;
use burn::tensor::{Element, FloatDType};
use log::LevelFilter;
use rl_algorithm::base::model::ActorModel;
use rl_algorithm::preload_net::normal_mlp_policy::NormalMLPPolicyConfig;
use rl_env::gym_humanoid_v4::HumanoidV4;
use rl_env::inverted_pendulum_v4::InvertedPendulumV4;

#[cfg(not(feature = "use_f32"))]
type FType = half::f16;

#[cfg(not(feature = "use_f32"))]
fn f32_to_ftype(v: f32) -> FType {
    FType::from_f32(v)
}

#[cfg(not(feature = "use_f32"))]
fn ftype_to_f32(v: FType) -> f32 {
    v.into()
}

// #[cfg(not(feature = "use_f32"))]
// type PrecisionSettings = HalfPrecisionSettings;

#[cfg(feature = "use_f32")]
type FType = f32;

#[cfg(feature = "use_f32")]
fn f32_to_ftype(v: f32) -> FType {
    v
}

#[cfg(feature = "use_f32")]
fn ftype_to_f32(v: FType) -> f32 {
    v.into()
}

// type MyBackend = LibTorch<crate::FType>;
// type F32Backend = LibTorch;
// type MyDevice = LibTorchDevice;
type MyBackend = Wgpu;
type F32Backend = Wgpu;
type MyDevice = WgpuDevice;

// pub type Rocm<F = f32, I = i32, B = u8> = ;

// type MyBackend = CubeBackend<HipRuntime, FType, i32, u8>;
// type F32Backend = LibTorch;
// type MyDevice = HipDevice;

type IType = <MyBackend as Backend>::IntElem;
type MyPrecisionSettings = CustomPrecisionSettings<FType, IType>;

#[derive(Debug, Default, Clone)]
pub struct CustomPrecisionSettings<
    F: Element + Serialize + DeserializeOwned,
    I: Element + Serialize + DeserializeOwned,
> {
    f: PhantomData<F>,
    i: PhantomData<I>,
}
impl<F: Element + Serialize + DeserializeOwned, I: Element + Serialize + DeserializeOwned>
    PrecisionSettings for CustomPrecisionSettings<F, I>
{
    type FloatElem = F;
    type IntElem = I;
}

fn main() {
    ppo_run::train_network::<HumanoidV4>();
}
