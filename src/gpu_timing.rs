use crate::{config, gpu_context::GpuContext};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Span {
    Scan,
    Scatter,
    Solve,
}

impl Span {
    pub const ALL: [Span; 3] = [Span::Scan, Span::Scatter, Span::Solve];

    pub fn label(self) -> &'static str {
        match self {
            Span::Scan => "scan_cells",
            Span::Scatter => "scatter_particles",
            Span::Solve => "solve",
        }
    }
}

const CAPACITY: u32 = config::MAX_SUBSTEPS * Span::ALL.len() as u32 * 2;
const SMOOTHING: f32 = 0.1;

pub struct PassTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    in_flight: bool,
    period_ns: f32,
    cursor: u32,
    pending: u32,
    averages: [f32; Span::ALL.len()],
}

impl PassTimer {
    pub fn new(gpu_context: &GpuContext<'_>) -> Option<Self> {
        if !gpu_context
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }

        let query_set = gpu_context
            .device
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Solver Timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: CAPACITY,
            });
        let bytes = CAPACITY as u64 * 8;
        let resolve = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            query_set,
            resolve,
            readback,
            ready: Arc::new(AtomicBool::new(false)),
            in_flight: false,
            period_ns: gpu_context.queue.get_timestamp_period(),
            cursor: 0,
            pending: 0,
            averages: [0.0; Span::ALL.len()],
        })
    }

    pub fn begin_frame(&mut self) {
        self.cursor = 0;
    }

    pub fn next_pair(&mut self) -> Option<(u32, u32)> {
        if self.cursor + 2 > CAPACITY {
            return None;
        }
        let pair = (self.cursor, self.cursor + 1);
        self.cursor += 2;
        Some(pair)
    }

    pub fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    pub fn is_idle(&self) -> bool {
        !self.in_flight
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.cursor == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..self.cursor, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, self.cursor as u64 * 8);
        self.pending = self.cursor;
        self.in_flight = true;
    }

    pub fn begin_readback(&self) {
        let ready = Arc::clone(&self.ready);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
    }

    pub fn poll(&mut self, gpu_context: &GpuContext<'_>) {
        if !self.in_flight {
            return;
        }
        gpu_context.device.poll(wgpu::Maintain::Poll);
        if !self.ready.swap(false, Ordering::Acquire) {
            return;
        }

        let mut frame = [0.0f32; Span::ALL.len()];
        {
            let view = self.readback.slice(..).get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&view[..self.pending as usize * 8]);
            for (pair, window) in ticks.chunks_exact(2).enumerate() {
                let elapsed = window[1].saturating_sub(window[0]);
                frame[pair % Span::ALL.len()] += elapsed as f32 * self.period_ns / 1.0e6;
            }
        }
        self.readback.unmap();
        self.in_flight = false;

        for (average, sample) in self.averages.iter_mut().zip(frame) {
            *average = if *average == 0.0 {
                sample
            } else {
                *average + SMOOTHING * (sample - *average)
            };
        }
    }

    pub fn averages(&self) -> [f32; Span::ALL.len()] {
        self.averages
    }
}
