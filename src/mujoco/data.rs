use super::model;
use std::{
    any::Any,
    ffi::CString,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use super::ffi::{self, mjData_, mjs_findBody};

#[macro_export]
macro_rules! data_field_by_body_def {
    ($func_name:ident, $field:ident, $type:ty, $num_field:ident, $size:expr) => {
        pub(crate) fn $func_name(&self, name: &str) -> [$type; $size] {
            let id = self.get_body_id(name);
            let field_ptr = self.get_ref().$field;
            let mut res: [$type; $size] = [0 as $type; $size];
            unsafe {
                if id < self.model.$num_field as usize {
                    res.copy_from_slice(std::slice::from_raw_parts(
                        field_ptr.wrapping_add(3 * id),
                        3,
                    ));
                }
            }
            return res;
        }
    };
}

#[macro_export]
macro_rules! get_data_filed {
    ($func_name:ident, $field:ident, $type:ty, $row_size_field:ident, $col_size:expr) => {
        pub(crate) fn $func_name(&self) -> ndarray::ArrayViewMut2<$type> {
            return unsafe {
                ndarray::ArrayViewMut2::<$type>::from_shape_ptr(
                    (self.model.$row_size_field as usize, $col_size),
                    self.$field,
                )
            };
        }
    };
}

struct MyData(*mut ffi::mjData);
unsafe impl Send for MyData {}
impl Deref for MyData {
    type Target = ffi::mjData;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl DerefMut for MyData {
    // type Target = ffi::mjData;

    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}
pub(crate) struct Data {
    pub(crate) model: Arc<model::Model>,
    pub(crate) mj_data: MyData,
}

impl Data {
    fn vec_i8_to_string(vec: Vec<i8>) -> Result<String, std::string::FromUtf8Error> {
        // 将 Vec<i8> 转换为 Vec<u8>
        let bytes = vec.iter().map(|&x| x as u8).collect();
        // 尝试将字节转换为字符串
        String::from_utf8(bytes)
    }
    pub(crate) fn new(model: Arc<model::Model>) -> Self {
        unsafe {
            let data = ffi::mj_makeData(model.get_ref());
            return Self {
                model: model,
                mj_data: MyData(data),
            };
        }
    }

    pub(crate) fn new_arc(model: Arc<model::Model>) -> Arc<Mutex<Self>> {
        unsafe {
            let data = ffi::mj_makeData(model.get_ref());
            return Arc::from(Mutex::new(Self {
                model: model,
                mj_data: MyData(data),
            }));
        }
    }
    pub(crate) fn get_ref(&self) -> &ffi::mjData {
        unsafe { &*self.mj_data }
    }
    pub(crate) fn get_mut(&mut self) -> &mut ffi::mjData {
        unsafe { &mut *self.mj_data }
    }
    pub(crate) fn get_body_id(&self, name: &str) -> usize {
        unsafe {
            return ffi::mj_name2id(
                self.model.get_ref(),
                ffi::mjtObj__mjOBJ_BODY as i32,
                CString::new(name).unwrap().as_ptr(),
            ) as usize;
        };
    }

    pub(crate) fn get_all_body_name(&self) -> Vec<String> {
        unsafe {
            let mut vec = vec![];
            for id in 0..self.model.nbody {
                let ptr = ffi::mj_id2name(self.model.get_ref(), ffi::mjtObj__mjOBJ_BODY as i32, id);
                let c_str = CString::from_raw(ptr as *mut i8);
                vec.push(c_str.clone().to_string_lossy().to_string());
                c_str.into_raw();
            }
            return vec;
        };
    }

    pub(crate) fn get_subtree_comxx(&self) -> ndarray::ArrayViewMut2<f64> {
        return unsafe {
            ndarray::ArrayViewMut2::<f64>::from_shape_ptr(
                (self.model.nbody as usize, 3),
                self.subtree_com,
            )
        };
    }
    data_field_by_body_def!(get_subtree_com_by_body, subtree_com, f64, nbody, 3);

    get_data_filed!(get_qpos, qpos, f64, nq, 1);
    get_data_filed!(get_qvel, qvel, f64, nv, 1);
    get_data_filed!(get_act, act, f64, na, 1);
    get_data_filed!(get_qacc_warmstart, qacc_warmstart, f64, nv, 1);
    get_data_filed!(get_plugin_state, plugin_state, f64, npluginstate, 1);
    get_data_filed!(get_ctrl, ctrl, f64, nu, 1);
    get_data_filed!(get_qfrc_applied, qfrc_applied, f64, nv, 1);
    get_data_filed!(get_xfrc_applied, xfrc_applied, f64, nbody, 6);
    get_data_filed!(get_mocap_pos, mocap_pos, f64, nmocap, 3);
    get_data_filed!(get_mocap_quat, mocap_quat, f64, nmocap, 4);
    get_data_filed!(get_qacc, qacc, f64, nv, 1);
    get_data_filed!(get_act_dot, act_dot, f64, na, 1);
    get_data_filed!(get_userdata, userdata, f64, nuserdata, 1);
    get_data_filed!(get_sensordata, sensordata, f64, nsensordata, 1);
    get_data_filed!(get_xpos, xpos, f64, nbody, 3);
    get_data_filed!(get_xquat, xquat, f64, nbody, 4);
    get_data_filed!(get_xmat, xmat, f64, nbody, 9);
    get_data_filed!(get_xipos, xipos, f64, nbody, 3);
    get_data_filed!(get_ximat, ximat, f64, nbody, 9);
    get_data_filed!(get_xanchor, xanchor, f64, njnt, 3);
    get_data_filed!(get_xaxis, xaxis, f64, njnt, 3);
    get_data_filed!(get_geom_xpos, geom_xpos, f64, ngeom, 3);
    get_data_filed!(get_geom_xmat, geom_xmat, f64, ngeom, 9);
    get_data_filed!(get_site_xpos, site_xpos, f64, nsite, 3);
    get_data_filed!(get_site_xmat, site_xmat, f64, nsite, 9);
    get_data_filed!(get_cam_xpos, cam_xpos, f64, ncam, 3);
    get_data_filed!(get_cam_xmat, cam_xmat, f64, ncam, 9);
    get_data_filed!(get_light_xpos, light_xpos, f64, nlight, 3);
    get_data_filed!(get_light_xdir, light_xdir, f64, nlight, 3);
    get_data_filed!(get_subtree_com, subtree_com, f64, nbody, 3);
    get_data_filed!(get_cdof, cdof, f64, nv, 6);
    get_data_filed!(get_cinert, cinert, f64, nbody, 10);
    get_data_filed!(get_flexvert_xpos, flexvert_xpos, f64, nflexvert, 3);
    get_data_filed!(get_flexelem_aabb, flexelem_aabb, f64, nflexelem, 6);
    // get_data_filed!(get_flexedge_J, flexedge_J, f64, nflexedge, nv);
    get_data_filed!(get_flexedge_length, flexedge_length, f64, nflexedge, 1);
    // get_data_filed!(get_ten_J, ten_J, f64, ntendon, nv);
    get_data_filed!(get_ten_length, ten_length, f64, ntendon, 1);
    get_data_filed!(get_wrap_xpos, wrap_xpos, f64, nwrap, 6);
    get_data_filed!(get_actuator_length, actuator_length, f64, nu, 1);
    get_data_filed!(get_actuator_moment, actuator_moment, f64, nJmom, 1);
    get_data_filed!(get_crb, crb, f64, nbody, 10);
    get_data_filed!(get_qM, qM, f64, nM, 1);
    get_data_filed!(get_qLD, qLD, f64, nM, 1);
    get_data_filed!(get_qLDiagInv, qLDiagInv, f64, nv, 1);
    get_data_filed!(get_bvh_aabb_dyn, bvh_aabb_dyn, f64, nbvhdynamic, 6);
    get_data_filed!(get_flexedge_velocity, flexedge_velocity, f64, nflexedge, 1);
    get_data_filed!(get_ten_velocity, ten_velocity, f64, ntendon, 1);
    get_data_filed!(get_actuator_velocity, actuator_velocity, f64, nu, 1);
    get_data_filed!(get_cvel, cvel, f64, nbody, 6);
    get_data_filed!(get_cdof_dot, cdof_dot, f64, nv, 6);
    get_data_filed!(get_qfrc_bias, qfrc_bias, f64, nv, 1);
    get_data_filed!(get_qfrc_spring, qfrc_spring, f64, nv, 1);
    get_data_filed!(get_qfrc_damper, qfrc_damper, f64, nv, 1);
    get_data_filed!(get_qfrc_gravcomp, qfrc_gravcomp, f64, nv, 1);
    get_data_filed!(get_qfrc_fluid, qfrc_fluid, f64, nv, 1);
    get_data_filed!(get_qfrc_passive, qfrc_passive, f64, nv, 1);
    get_data_filed!(get_subtree_linvel, subtree_linvel, f64, nbody, 3);
    get_data_filed!(get_subtree_angmom, subtree_angmom, f64, nbody, 3);
    get_data_filed!(get_qH, qH, f64, nM, 1);
    get_data_filed!(get_qHDiagInv, qHDiagInv, f64, nv, 1);
    get_data_filed!(get_qDeriv, qDeriv, f64, nD, 1);
    get_data_filed!(get_qLU, qLU, f64, nD, 1);
    get_data_filed!(get_actuator_force, actuator_force, f64, nu, 1);
    get_data_filed!(get_qfrc_actuator, qfrc_actuator, f64, nv, 1);
    get_data_filed!(get_qfrc_smooth, qfrc_smooth, f64, nv, 1);
    get_data_filed!(get_qacc_smooth, qacc_smooth, f64, nv, 1);
    get_data_filed!(get_qfrc_constraint, qfrc_constraint, f64, nv, 1);
    get_data_filed!(get_qfrc_inverse, qfrc_inverse, f64, nv, 1);
    get_data_filed!(get_cacc, cacc, f64, nbody, 6);
    get_data_filed!(get_cfrc_int, cfrc_int, f64, nbody, 6);
    get_data_filed!(get_cfrc_ext, cfrc_ext, f64, nbody, 6);
    // get_data_filed!(get_efc_J, efc_J, f64, nJ, 1);
    // get_data_filed!(get_efc_JT, efc_JT, f64, nJ, 1);
    // get_data_filed!(get_efc_pos, efc_pos, f64, nefc, 1);
    // get_data_filed!(get_efc_margin, efc_margin, f64, nefc, 1);
    // get_data_filed!(get_efc_frictionloss, efc_frictionloss, f64, nefc, 1);
    // get_data_filed!(get_efc_diagApprox, efc_diagApprox, f64, nefc, 1);
    // get_data_filed!(get_efc_KBIP, efc_KBIP, f64, nefc, 4);
    // get_data_filed!(get_efc_D, efc_D, f64, nefc, 1);
    // get_data_filed!(get_efc_R, efc_R, f64, nefc, 1);
    // get_data_filed!(get_efc_AR, efc_AR, f64, nA, 1);
    // get_data_filed!(get_efc_vel, efc_vel, f64, nefc, 1);
    // get_data_filed!(get_efc_aref, efc_aref, f64, nefc, 1);
    // get_data_filed!(get_efc_b, efc_b, f64, nefc, 1);
    // get_data_filed!(get_efc_force, efc_force, f64, nefc, 1);
}

impl Deref for Data {
    type Target = ffi::mjData;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mj_data }
    }
}

impl DerefMut for Data {
    // type Target = ffi::mjData;

    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mj_data }
    }
}

// 实现 Drop 以自动释放内存
impl Drop for Data {
    fn drop(&mut self) {
        unsafe {
            ffi::mj_deleteData(self.mj_data.0);
        }
    }
}
