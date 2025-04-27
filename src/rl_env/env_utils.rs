use crate::mujoco;
use ndarray::{s, Array1, ArrayView2, Axis};
use std::sync::Arc;

pub fn mass_center(model: Arc<mujoco::model::Model>, xipos: ArrayView2<f64>) -> Array1<f64> {
    let body_mass = model.get_body_mass();
    let res = (&body_mass * &xipos).sum_axis(Axis(0)) / body_mass.sum();
    return res.slice(s![0..3]).to_owned();
}

pub fn velocity(
    dt: f64,
    model: Arc<mujoco::model::Model>,
    before_xipos: ArrayView2<f64>,
    after_xipos: ArrayView2<f64>,
) -> (f64, f64, f64) {
    let befor_pos = mass_center(model.clone(), before_xipos);
    let after_pos = mass_center(model, after_xipos);
    let xyz_velocity = (&after_pos - &befor_pos) / dt;
    return (xyz_velocity[0], xyz_velocity[1], xyz_velocity[2]);
}
