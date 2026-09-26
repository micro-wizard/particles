#[cfg(not(target_arch = "wasm32"))]
mod bench;
mod compute_pipeline;
mod config;
mod controller;
mod ecs;
mod gpu_context;
mod gpu_timing;
mod materials;
mod model;
mod overlay;
mod pixel_font;
mod view;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    controller::run().await;
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn bench() {
    env_logger::init();
    bench::run().await;
}
