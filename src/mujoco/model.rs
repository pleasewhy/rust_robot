use std::{
    any::Any,
    ffi::CString,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use super::ffi::{self, mjData_, mjs_findBody};

pub struct Model {
    pub mj_model: *mut ffi::mjModel,
    pub mj_spec: *mut ffi::mjSpec,
}

impl Model {
    fn vec_i8_to_string(vec: Vec<i8>) -> Result<String, std::string::FromUtf8Error> {
        // 将 Vec<i8> 转换为 Vec<u8>
        let bytes = vec.iter().map(|&x| x as u8).collect();
        // 尝试将字节转换为字符串
        String::from_utf8(bytes)
    }
    pub fn from_xml_file(filename: &str) -> Result<Arc<Model>, String> {
        unsafe {
            let mut error: Vec<i8> = Vec::new();
            error.resize(1000, 0);
            let mj_spec = ffi::mj_parseXML(
                CString::new(filename).unwrap().as_ptr(),
                std::ptr::null(),
                error.as_mut_ptr(),
                1000,
            );
            if mj_spec == std::ptr::null_mut() {
                let s = Self::vec_i8_to_string(error);
                return Result::Err(s.unwrap());
            }
            let mj_model = ffi::mj_compile(mj_spec, std::ptr::null());
            if mj_model == std::ptr::null_mut() {
                let s = Self::vec_i8_to_string(error);
                return Result::Err(s.unwrap());
            }
            return Result::Ok(Arc::from(Model {
                mj_model: mj_model,
                mj_spec: mj_spec,
            }));
        }
    }
    pub fn get_ref(&self) -> &ffi::mjModel {
        unsafe { &mut *self.mj_model }
    }
    pub fn get_mut(&mut self) -> &mut ffi::mjModel {
        unsafe { &mut *self.mj_model }
    }
    pub fn get_body_id(&self, name: &str) -> usize {
        unsafe {
            return ffi::mj_name2id(
                self.mj_model,
                ffi::mjtObj__mjOBJ_BODY as i32,
                CString::new(name).unwrap().as_ptr(),
            ) as usize;
        };
    }
}

impl Deref for Model {
    type Target = ffi::mjModel;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mj_model }
    }
}

impl DerefMut for Model {
    // type Target = ffi::mjModel;

    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mj_model }
    }
}

// impl Deref for Arc<Model> {
//     type Target = ffi::mjModel;

//     fn deref(&self) -> &Self::Target {
//         unsafe { &*self.mj_model }
//     }
// }

// impl DerefMut for Model {
//     // type Target = ffi::mjModel;

//     fn deref_mut(&mut self) -> &mut Self::Target {
//         unsafe { &mut *self.mj_model }
//     }
// }

// 实现 Drop 以自动释放内存
impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            ffi::mj_deleteModel(self.mj_model);
        }
    }
}
