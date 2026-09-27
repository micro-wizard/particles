fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::args().any(|arg| arg == "--bench") {
        pollster::block_on(particles::bench());
        return;
    }
    pollster::block_on(particles::run());
}
