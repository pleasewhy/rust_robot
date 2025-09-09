use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, OnceLock},
};

use lazy_static::lazy_static;
use ndarray::parallel::prelude::IntoParallelIterator;
use tensorboard_rs::summary_writer::SummaryWriter;
use video_rs::ffmpeg::decoder::new;

pub mod memory;
pub mod on_policy_runner;
pub mod rl_utils;
// pub mod model_base_runner;
pub mod config;
pub mod model;

// #[derive(Debug)]
pub struct EpochLogger {
    log_info: BTreeMap<(String, String), f32>,
    writer: Option<SummaryWriter>,
}

lazy_static! {
    pub static ref MyLogger: Arc<Mutex<EpochLogger>> = Arc::new(Mutex::new(EpochLogger {
        log_info: BTreeMap::<(String, String), f32>::new(),
        writer: None,
    }));
}

pub enum EpochLoggerAggMode {
    Sum,
    Mean,
    Max,
    Min,
    Replace,
}
impl EpochLogger {
    pub fn init_writer(logdir: String) {
        let mut this = MyLogger.lock().unwrap();
        this.writer = Some(SummaryWriter::new(logdir));
    }

    pub fn add_scalar(main_tag_sub_tag: (&str, &str), val: f32) {
        let mut this = MyLogger.lock().unwrap();
        let main_tag_sub_tag = (
            main_tag_sub_tag.0.to_string(),
            main_tag_sub_tag.1.to_string(),
        );
        this.log_info.insert(main_tag_sub_tag, val);
    }

    pub fn add_scalar_agg(
        main_tag_sub_tag: (&str, &str),
        mut val: f32,
        agg_mod: EpochLoggerAggMode,
    ) {
        let mut this = MyLogger.lock().unwrap();
        let main_tag_sub_tag = (
            main_tag_sub_tag.0.to_string(),
            main_tag_sub_tag.1.to_string(),
        );
        if let Some(old_val) = this.log_info.get(&main_tag_sub_tag) {
            val = match agg_mod {
                EpochLoggerAggMode::Sum => val + old_val,
                EpochLoggerAggMode::Mean => (val + old_val) / 2.0,
                EpochLoggerAggMode::Max => val.max(*old_val),
                EpochLoggerAggMode::Min => val.min(*old_val),
                EpochLoggerAggMode::Replace => *old_val,
            }
        }

        this.log_info.insert(main_tag_sub_tag, val);
    }

    pub fn write_scalar(&mut self, main_tag: &str, sub_tag: &str, scalar: f32, step: usize) {
        let mut map = HashMap::<String, f32>::new();
        map.insert(sub_tag.to_string(), scalar);
        self.writer
            .as_mut()
            .expect("not call init_writer func")
            .add_scalars(format!("{}/{}", main_tag, sub_tag).as_str(), &map, step);
    }

    pub fn log(step: usize) {
        let mut this = MyLogger.lock().unwrap();

        println!("************iter={}************", step);
        let log_info = std::mem::take(&mut this.log_info);
        for ((main_tag, sub_tag), scalar) in log_info {
            this.write_scalar(main_tag.as_str(), sub_tag.as_str(), scalar.into(), step);
            println!("{}-{}={}", main_tag, sub_tag, scalar);
        }
        println!("************iter={}************", step);
        println!();
        println!();
        this.log_info.clear();

        this.writer.as_mut().unwrap().flush();
    }
}
