use super::ffi;
use core::hash;
use glfw::{self, Context};
use glfw::{fail_on_errors, Glfw};
use image;
use std::option;

#[derive(Debug)]
pub struct Render {
    pub scene: ffi::mjvScene,
    pub render_context: ffi::mjrContext,
    pub camera: ffi::mjvCamera,
    pub option: ffi::mjvOption,
    pub viewport: ffi::mjrRect,
    _glfw: glfw::Glfw,
    _window: glfw::PWindow,
    _events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
}

impl Render {
    fn init_glfw() -> (
        glfw::Glfw,
        glfw::PWindow,
        glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    ) {
        let mut glfw = glfw::init(fail_on_errors!()).unwrap();
        glfw::WindowHint::Visible(false);
        glfw::WindowHint::DoubleBuffer(false);
        let (mut window, events) = glfw
            .create_window(300, 300, "mujoco_test", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window.");
        window.make_current();
        return (glfw, window, events);
    }
    pub fn new(model: &ffi::mjModel) -> Self {
        let (glfw, window, events) = Self::init_glfw();
        let mut x = Self {
            scene: Default::default(),
            render_context: Default::default(),
            camera: Default::default(),
            option: Default::default(),
            viewport: Default::default(),
            _glfw: glfw,
            _window: window,
            _events: events,
        };
        unsafe {
            ffi::mjv_defaultCamera(&mut x.camera);
            ffi::mjv_defaultOption(&mut x.option);
            ffi::mjv_defaultScene(&mut x.scene);
            ffi::mjr_defaultContext(&mut x.render_context);
            ffi::mjv_makeScene(model, &mut x.scene, 2000);
            ffi::mjr_makeContext(model, &mut x.render_context, 200);
            ffi::mjv_defaultFreeCamera(model, &mut x.camera);
            ffi::mjr_setBuffer(
                ffi::mjtFramebuffer__mjFB_OFFSCREEN as i32,
                &mut x.render_context,
            );
            x.viewport = ffi::mjr_maxViewport(&mut x.render_context);
        }
        return x;
    }
    pub fn update_scene(&mut self, model: &ffi::mjModel, data: &mut ffi::mjData) {
        unsafe {
            ffi::mjv_updateScene(
                model,
                data,
                &self.option,
                std::ptr::null(),
                &mut self.camera,
                ffi::mjtCatBit__mjCAT_ALL as i32,
                &mut self.scene,
            );
        }
    }

    pub fn render(&mut self) -> (Vec<u8>, Vec<f32>) {
        let mut rgb = Vec::new();
        let mut depth = Vec::new();
        let w = self.viewport.width;
        let h = self.viewport.height;
        rgb.resize((3 * w * h) as usize, 0);
        depth.resize((w * h) as usize, 0.0);
        let mut stamp = format!("{:.2}", 1.0).into_bytes();
        stamp.resize(50, 0);
        unsafe {
            ffi::mjr_render(self.viewport, &mut self.scene, &mut self.render_context);
            ffi::mjr_overlay(
                ffi::mjtFont__mjFONT_NORMAL as i32,
                ffi::mjtGridPos__mjGRID_TOPLEFT as i32,
                self.viewport,
                stamp.as_ptr() as *const i8,
                std::ptr::null(),
                &mut self.render_context,
            );
            ffi::mjr_readPixels(
                rgb.as_mut_ptr(),
                depth.as_mut_ptr(),
                self.viewport,
                &mut self.render_context,
            );
        }
        let flipped_rgb = Self::flip_vertical(&rgb, w as usize, h as usize);
        return (flipped_rgb, depth);
    }

    fn flip_vertical(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
        let row_size = width * 3;
        let mut flipped = Vec::with_capacity(rgb.len());
        for y in (0..height).rev() {
            let start = y * row_size;
            let end = start + row_size;
            flipped.extend_from_slice(&rgb[start..end]);
        }
        flipped
    }

    pub fn get_width(&self) -> usize {
        self.viewport.width as usize
    }
    pub fn get_height(&self) -> usize {
        self.viewport.height as usize
    }
}

impl Drop for Render {
    fn drop(&mut self) {
        unsafe {
            ffi::mjr_freeContext(&mut self.render_context);
            ffi::mjv_freeScene(&mut self.scene);
        }
    }
}
