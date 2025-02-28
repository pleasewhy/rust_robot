use super::model;
use std::{
    any::Any,
    ffi::CString,
    ops::{Deref, DerefMut},
    sync::Arc,
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
    ($func_name:ident, $field:ident, $type:ty, $num_field:ident, $size:expr) => {
        pub(crate) fn $func_name(&self) -> ndarray::Array2<$type> {
            let raw_slice = unsafe {
                std::slice::from_raw_parts_mut(self.$field, self.model.$num_field as usize * 3)
            };
            let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<$type>, ndarray::Dim<[usize; 2]>> =
                ndarray::Array2::<$type>::zeros((self.model.$num_field as usize, 3));
            res.as_slice_mut().unwrap().copy_from_slice(&raw_slice);
            return res;
        }
    };
}

pub(crate) struct Data {
    pub(crate) model: Arc<model::Model>,
    pub(crate) mj_data: *mut ffi::mjData,
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
                mj_data: data,
            };
        }
    }
    pub(crate) fn get_ref(&self) -> &ffi::mjData {
        unsafe { &mut *self.mj_data }
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

    // pub(crate) fn get_subtree_com(&self) -> ndarray::Array2<f64> {
    //     let raw_slice = unsafe {
    //         std::slice::from_raw_parts_mut(self.subtree_com, self.model.nbody as usize * 3)
    //     };
    //     let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
    //         ndarray::Array2::<f64>::zeros((self.model.nbody as usize, 3));
    //     res.as_slice_mut().unwrap().copy_from_slice(&raw_slice);
    //     return res;
    // }
    data_field_by_body_def!(get_subtree_com_by_body, subtree_com, f64, nbody, 3);
    get_data_filed!(get_subtree_com, subtree_com, f64, nbody, 3);
    get_data_filed!(get_qpos, qpos, f64, nq, 1);
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
            ffi::mj_deleteData(self.mj_data);
        }
    }
}
