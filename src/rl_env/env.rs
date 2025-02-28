// use crate::mujoco;

// pub struct Env {
//     pub model: mujoco::model::Model,
//     pub data: mujoco::data::Data,
//     pub render: mujoco::render::Render,
// }

// impl Env {
//     fn get_obs(&self) -> [f64] {
//         let obs: &[i32] = &[];
//         position = std::slice::from_raw_parts(self.data.qpos.flat.copy(), len)
//         velocity = self.data.qvel.flat.copy()

//         com_inertia = self.data.cinert.flat.copy()
//         com_velocity = self.data.cvel.flat.copy()

//         actuator_forces = self.data.qfrc_actuator.flat.copy()
//         external_contact_forces = self.data.cfrc_ext.flat.copy()
//         // self.data.qvel.flat,
//         // self.data.cinert.flat,
//         // self.data.cvel.flat,
//         // self.data.qfrc_actuator.flat,
//         // self.data.cfrc_ext.flat,
//     }
// }
