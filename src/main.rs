fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().any(|arg| arg == "--bench") {
        pollster::block_on(two_d_game_engine::bench());
        return;
    }
    pollster::block_on(two_d_game_engine::run());
}
