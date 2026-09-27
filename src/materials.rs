use crate::config;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MaterialParams {
    pub colour: [f32; 4],
    pub freq_n: f32,
    pub freq_t: f32,
    pub zeta_n: f32,
    pub zeta_t: f32,
    pub mu: f32,
    pub density: f32,
    pub radius: f32,
    pub rest_packing: f32,
    pub pressure_k: f32,
    pub viscosity: f32,
    pub is_static: f32,
    pub spacing: f32,
    pub conductivity: f32,
    pub heat_capacity: f32,
    pub default_temperature: f32,
    pub thermal_expansion: f32,
    pub above_point: f32,
    pub becomes_above: u32,
    pub below_point: f32,
    pub becomes_below: u32,
    pub bond_freq: f32,
    pub heat_release: f32,
    pub growth_period: f32,
    pub sprouts: u32,
}

pub const NO_TRANSITION: u32 = u32::MAX;

unsafe impl bytemuck::Zeroable for MaterialParams {}
unsafe impl bytemuck::Pod for MaterialParams {}

#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub name: &'static str,
    /// Whether the brush menu offers it; growth and burning products are left
    /// to arise on their own.
    pub palette: bool,
    pub params: MaterialParams,
}

pub const MATERIAL_COUNT: u32 = MATERIALS.len() as u32;

pub const WATER: u32 = 3;
pub const ICE: u32 = 5;
pub const STEAM: u32 = 6;
pub const LAVA: u32 = 7;
pub const OBSIDIAN: u32 = 8;
pub const PLANT: u32 = 10;
pub const EMBER: u32 = 11;
pub const ASH: u32 = 12;

pub const MATERIALS: [Material; 13] = [
    Material {
        name: "Sand",
        palette: true,
        params: MaterialParams {
            colour: [0.722, 0.561, 0.322, 1.0],
            freq_n: 80.0,
            freq_t: 25.0,
            zeta_n: 0.60,
            zeta_t: 0.63,
            mu: 0.5,
            density: 1.0,
            radius: 1.0,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 150.0,
            heat_capacity: 1.2,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 0.0,
            becomes_above: NO_TRANSITION,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Gravel",
        palette: true,
        params: MaterialParams {
            colour: [0.470, 0.470, 0.500, 1.0],
            freq_n: 53.0,
            freq_t: 17.0,
            zeta_n: 0.33,
            zeta_t: 0.39,
            mu: 0.8,
            density: 1.6,
            radius: 1.4,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 120.0,
            heat_capacity: 1.5,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 900.0,
            becomes_above: LAVA,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Powder",
        palette: true,
        params: MaterialParams {
            colour: [0.780, 0.760, 0.700, 1.0],
            freq_n: 77.0,
            freq_t: 24.5,
            zeta_n: 0.97,
            zeta_t: 0.92,
            mu: 0.2,
            density: 0.5,
            radius: 0.8,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 60.0,
            heat_capacity: 1.0,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 0.0,
            becomes_above: NO_TRANSITION,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Water",
        palette: true,
        params: MaterialParams {
            colour: [0.157, 0.435, 0.776, 1.0],
            freq_n: 77.0,
            freq_t: 24.5,
            zeta_n: 1.0,
            zeta_t: 1.0,
            mu: 0.001,
            density: 0.8,
            radius: 0.7,
            rest_packing: 0.2373,
            pressure_k: 450_000.0,
            viscosity: 6.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 300.0,
            heat_capacity: 1.0,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 100.0,
            becomes_above: STEAM,
            below_point: 0.0,
            becomes_below: ICE,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Wall",
        palette: true,
        params: MaterialParams {
            colour: [0.180, 0.180, 0.220, 1.0],
            freq_n: 80.0,
            freq_t: 25.0,
            zeta_n: 0.60,
            zeta_t: 0.63,
            mu: 0.6,
            density: 10.0,
            radius: 0.7,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 1.0,
            spacing: 2.2,
            conductivity: 400.0,
            heat_capacity: 1.0,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 0.0,
            becomes_above: NO_TRANSITION,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Ice",
        palette: true,
        params: MaterialParams {
            colour: [0.750, 0.850, 0.950, 1.0],
            freq_n: 80.0,
            freq_t: 25.0,
            zeta_n: 0.60,
            zeta_t: 0.63,
            mu: 0.05,
            density: 0.72,
            radius: 0.7,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 400.0,
            heat_capacity: 1.0,
            default_temperature: -10.0,
            thermal_expansion: 0.0,
            above_point: 0.0,
            becomes_above: WATER,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Steam",
        palette: true,
        params: MaterialParams {
            colour: [0.850, 0.870, 0.900, 1.0],
            freq_n: 77.0,
            freq_t: 24.5,
            zeta_n: 1.0,
            zeta_t: 1.0,
            mu: 0.001,
            density: 0.05,
            radius: 0.7,
            rest_packing: 0.1284,
            pressure_k: 4_300.0,
            viscosity: 0.5,
            is_static: 0.0,
            spacing: 3.0,
            conductivity: 100.0,
            heat_capacity: 3.0,
            default_temperature: 120.0,

            thermal_expansion: 1.0,
            above_point: 0.0,
            becomes_above: NO_TRANSITION,

            below_point: 99.0,
            becomes_below: WATER,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Lava",
        palette: true,

        params: MaterialParams {
            colour: [0.950, 0.350, 0.080, 1.0],
            freq_n: 77.0,
            freq_t: 24.5,
            zeta_n: 1.0,
            zeta_t: 1.0,
            mu: 0.001,

            density: 1.5,
            radius: 0.9,

            rest_packing: 0.2428,

            pressure_k: 600_000.0,

            viscosity: 18.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 250.0,

            heat_capacity: 4.0,
            default_temperature: 1000.0,

            thermal_expansion: 0.0,
            above_point: 0.0,
            becomes_above: NO_TRANSITION,

            below_point: 700.0,
            becomes_below: OBSIDIAN,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Obsidian",
        palette: true,

        params: MaterialParams {
            colour: [0.260, 0.220, 0.340, 1.0],
            freq_n: 80.0,
            freq_t: 25.0,
            zeta_n: 0.60,
            zeta_t: 0.63,

            mu: 0.7,

            density: 1.7,

            radius: 0.9,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,

            conductivity: 80.0,
            heat_capacity: 1.5,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,

            above_point: 900.0,
            becomes_above: LAVA,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Seed",
        palette: true,
        params: MaterialParams {
            colour: [0.470, 0.330, 0.170, 1.0],
            freq_n: 80.0,
            freq_t: 25.0,
            zeta_n: 0.30,
            zeta_t: 0.63,
            mu: 0.8,
            density: 3.0,
            radius: 1.0,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 60.0,
            heat_capacity: 1.5,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 250.0,
            becomes_above: EMBER,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 60.0,
            heat_release: 0.0,
            growth_period: 0.2,
            sprouts: PLANT,
        },
    },
    Material {
        name: "Plant",
        palette: false,
        params: MaterialParams {
            colour: [0.300, 0.620, 0.250, 1.0],
            freq_n: 60.0,
            freq_t: 20.0,
            zeta_n: 0.30,
            zeta_t: 0.80,
            mu: 0.4,
            density: 0.65,
            radius: 0.7,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 150.0,
            heat_capacity: 1.5,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 250.0,
            becomes_above: EMBER,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 60.0,
            heat_release: 0.0,
            growth_period: 0.2,
            sprouts: PLANT,
        },
    },
    Material {
        name: "Ember",
        palette: false,
        params: MaterialParams {
            colour: [1.000, 0.550, 0.120, 1.0],
            freq_n: 60.0,
            freq_t: 20.0,
            zeta_n: 0.30,
            zeta_t: 0.80,
            mu: 0.4,
            density: 0.65,
            radius: 0.7,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 150.0,
            heat_capacity: 1.5,
            default_temperature: 400.0,
            thermal_expansion: 0.0,
            above_point: 700.0,
            becomes_above: ASH,
            below_point: 150.0,
            becomes_below: PLANT,
            bond_freq: 60.0,
            heat_release: 450.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
    Material {
        name: "Ash",
        palette: false,
        params: MaterialParams {
            colour: [0.360, 0.350, 0.340, 1.0],
            freq_n: 77.0,
            freq_t: 24.5,
            zeta_n: 0.97,
            zeta_t: 0.92,
            mu: 0.3,
            density: 0.4,
            radius: 0.7,
            rest_packing: 0.0,
            pressure_k: 0.0,
            viscosity: 0.0,
            is_static: 0.0,
            spacing: 2.2,
            conductivity: 60.0,
            heat_capacity: 1.0,
            default_temperature: config::AMBIENT_TEMPERATURE,
            thermal_expansion: 0.0,
            above_point: 0.0,
            becomes_above: NO_TRANSITION,
            below_point: 0.0,
            becomes_below: NO_TRANSITION,
            bond_freq: 0.0,
            heat_release: 0.0,
            growth_period: 0.0,
            sprouts: NO_TRANSITION,
        },
    },
];

impl Material {
    pub fn is_static(&self) -> bool {
        self.params.is_static > 0.5
    }

    pub fn is_fluid(&self) -> bool {
        self.params.pressure_k > 0.0
    }

    // not actually dead, just used in test
    #[allow(dead_code)]
    pub fn is_gas(&self) -> bool {
        self.params.thermal_expansion > 0.0
    }
}

impl MaterialParams {
    pub fn rest_spacing(&self) -> f32 {
        self.radius * self.spacing
    }

    pub fn defaults() -> [MaterialParams; MATERIAL_COUNT as usize] {
        MATERIALS.map(|m| m.params)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Globals {
    pub gravity: f32,
    pub max_speed: f32,
    pub render_hysteresis: f32,

    pub ambient_density: f32,

    pub wind: f32,

    pub air_drag: f32,

    pub rest_temperature: f32,
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            gravity: 150.0,

            max_speed: 1500.0,
            render_hysteresis: 0.25,
            ambient_density: 0.08,
            wind: 0.0,

            air_drag: 0.01,

            rest_temperature: config::AMBIENT_TEMPERATURE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    fn w_poly6(r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        let d = h * h - r * r;
        4.0 / (std::f32::consts::PI * h.powi(8)) * d * d * d
    }

    fn lap_viscosity(r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        40.0 / (std::f32::consts::PI * h.powi(5)) * (h - r)
    }

    fn neighbour_distances(m: &MaterialParams) -> Vec<f32> {
        let spacing = m.rest_spacing();
        let reach = (config::SMOOTHING_RADIUS / spacing).ceil() as i32 + 1;
        let mut distances = Vec::new();
        for j in -reach..=reach {
            for i in -reach..=reach {
                if i == 0 && j == 0 {
                    continue;
                }
                let x = (i as f32 + 0.5 * j as f32) * spacing;
                let y = j as f32 * spacing * 0.866_025_4;
                distances.push(x.hypot(y));
            }
        }
        distances
    }

    fn measured_packing(m: &MaterialParams) -> f32 {
        let h = config::SMOOTHING_RADIUS;
        let volume = m.radius * m.radius;
        let neighbours: f32 = neighbour_distances(m)
            .into_iter()
            .map(|r| volume * w_poly6(r, h))
            .sum();
        volume * w_poly6(0.0, h) + neighbours
    }

    fn conduction_reach(m: &MaterialParams) -> f32 {
        let h = config::SMOOTHING_RADIUS;
        let volume = m.radius * m.radius;
        neighbour_distances(m)
            .into_iter()
            .map(|r| volume * lap_viscosity(r, h))
            .sum()
    }

    #[test]
    fn fluid_rest_packing_matches_its_radius() {
        for m in MATERIALS.iter().filter(|m| m.params.pressure_k > 0.0) {
            let measured = measured_packing(&m.params);
            let error = (measured - m.params.rest_packing).abs() / measured;
            assert!(
                error < 0.02,
                "{}: rest_packing is {} but at radius {} the kernel measures \
                 {measured:.4} — {:.0}% off",
                m.name,
                m.params.rest_packing,
                m.params.radius,
                error * 100.0,
            );
        }
    }

    #[test]
    fn every_material_respects_the_radius_range() {
        for m in MATERIALS.iter() {
            assert!(
                (config::RADIUS_FLOOR..=config::RADIUS_LIMIT).contains(&m.params.radius),
                "{}: radius {} is outside {}..={}",
                m.name,
                m.params.radius,
                config::RADIUS_FLOOR,
                config::RADIUS_LIMIT,
            );
        }
    }

    #[test]
    fn ambient_density_floats_gases_and_sinks_everything_else() {
        let ambient = Globals::default().ambient_density;
        for m in MATERIALS.iter().filter(|m| !m.is_static()) {
            if m.is_gas() {
                assert!(
                    m.params.density < ambient,
                    "{} is a gas at density {} but the air is only {ambient} — it \
                     would sink",
                    m.name,
                    m.params.density,
                );
            } else {
                assert!(
                    m.params.density > ambient,
                    "{} at density {} is lighter than the {ambient} air and would \
                     float away",
                    m.name,
                    m.params.density,
                );
            }
        }
    }

    #[test]
    fn shelter_reaches_the_first_ring_of_neighbours_only() {
        assert!(
            config::SHELTER_REACH < 3f32.sqrt(),
            "SHELTER_REACH {} reaches the second ring of a packing in contact, at {:.3}",
            config::SHELTER_REACH,
            3f32.sqrt(),
        );
        for m in MATERIALS.iter().filter(|m| !m.is_static() && !m.is_gas()) {
            let first_ring = m.params.rest_spacing() / (2.0 * m.params.radius);
            assert!(
                first_ring < config::SHELTER_REACH,
                "{} settles {first_ring} contact distances apart, beyond the {} \
                 SHELTER_REACH, so none of it would shelter any of the rest",
                m.name,
                config::SHELTER_REACH,
            );
        }
    }

    #[test]
    fn fluid_pressure_waves_stay_inside_the_substep() {
        for m in MATERIALS.iter().filter(|m| m.params.pressure_k > 0.0) {
            let p = &m.params;

            let hottest = 1.0
                + p.thermal_expansion * (config::MAX_TEMPERATURE - config::TEMPERATURE_REFERENCE)
                    / config::TEMPERATURE_REFERENCE;
            let sound_speed = (p.pressure_k * hottest * p.rest_packing / p.density).sqrt();
            let dt_max = 0.25 * config::SMOOTHING_RADIUS / sound_speed;
            assert!(
                dt_max > config::SUBSTEP,
                "{}: at {}° pressure waves travel at {sound_speed:.0} u/s, needing a \
                 substep under {dt_max:.5}s, but SUBSTEP is {}",
                m.name,
                config::MAX_TEMPERATURE,
                config::SUBSTEP,
            );
        }
    }

    #[test]
    fn heat_conduction_stays_inside_the_substep() {
        for m in MATERIALS.iter().filter(|m| !m.is_static()) {
            let thermal_mass = m.params.density * m.params.heat_capacity;
            for n in MATERIALS.iter() {
                let conductivity = 0.5 * (m.params.conductivity + n.params.conductivity);
                let step =
                    config::SUBSTEP * conductivity * conduction_reach(&n.params) / thermal_mass;
                assert!(
                    step <= 1.0,
                    "{} surrounded by {}: conduction moves it {step:.2} of the way to its \
                     neighbours' temperature in one substep, so it overshoots. Raise its \
                     heat_capacity or shorten SUBSTEP",
                    m.name,
                    n.name,
                );
            }
        }
    }

    #[test]
    fn contact_springs_stay_inside_the_substep() {
        for m in MATERIALS.iter() {
            assert!(
                m.params.freq_n <= config::MAX_FREQ_N
                    && m.params.freq_t <= config::MAX_FREQ_T
                    && m.params.bond_freq <= config::MAX_FREQ_N,
                "{}: springs at {} Hz normal, {} Hz tangential and {} Hz bonded, but \
                 SUBSTEP {} only holds {} and {}",
                m.name,
                m.params.freq_n,
                m.params.freq_t,
                m.params.bond_freq,
                config::SUBSTEP,
                config::MAX_FREQ_N,
                config::MAX_FREQ_T,
            );
        }
    }

    #[test]
    fn substep_budget_covers_the_longest_frame() {
        let reach = config::SUBSTEP * config::MAX_SUBSTEPS as f32;
        assert!(
            reach >= config::MAX_FRAME_TIME,
            "{} substeps of {} s buy only {reach} s a frame",
            config::MAX_SUBSTEPS,
            config::SUBSTEP,
        );
    }

    #[test]
    fn named_ids_match_the_table() {
        for (id, name) in [
            (WATER, "Water"),
            (ICE, "Ice"),
            (STEAM, "Steam"),
            (LAVA, "Lava"),
            (OBSIDIAN, "Obsidian"),
            (PLANT, "Plant"),
            (EMBER, "Ember"),
            (ASH, "Ash"),
        ] {
            assert_eq!(
                MATERIALS[id as usize].name, name,
                "id {id} is {}, not {name}",
                MATERIALS[id as usize].name,
            );
        }
    }

    #[test]
    fn transitions_point_at_real_materials() {
        for m in MATERIALS.iter() {
            for (target, edge) in [
                (m.params.becomes_above, "above"),
                (m.params.becomes_below, "below"),
                (m.params.sprouts, "sprout"),
            ] {
                assert!(
                    target == NO_TRANSITION || target < MATERIAL_COUNT,
                    "{}: {edge} transition targets id {target}, but there are only \
                     {MATERIAL_COUNT} materials",
                    m.name,
                );
            }
        }
    }

    #[test]
    fn phase_round_trips_cannot_oscillate() {
        for (id, m) in MATERIALS.iter().enumerate() {
            let me = &m.params;
            if me.becomes_above == NO_TRANSITION {
                continue;
            }

            let other = &MATERIALS[me.becomes_above as usize].params;
            if other.becomes_below == id as u32 {
                assert!(
                    other.below_point <= me.above_point,
                    "{} converts upward at {} but {} converts back at {} — a grain \
                     between them would flip every substep",
                    m.name,
                    me.above_point,
                    MATERIALS[me.becomes_above as usize].name,
                    other.below_point,
                );
            }
        }
    }
}

#[cfg(test)]
mod layout {
    use super::*;

    #[test]
    fn material_params_matches_the_wgsl_layout() {
        assert_eq!(std::mem::size_of::<MaterialParams>(), 112, "size");
        let expected: [(&str, usize); 25] = [
            ("colour", 0),
            ("freq_n", 16),
            ("freq_t", 20),
            ("zeta_n", 24),
            ("zeta_t", 28),
            ("mu", 32),
            ("density", 36),
            ("radius", 40),
            ("rest_packing", 44),
            ("pressure_k", 48),
            ("viscosity", 52),
            ("is_static", 56),
            ("spacing", 60),
            ("conductivity", 64),
            ("heat_capacity", 68),
            ("default_temperature", 72),
            ("thermal_expansion", 76),
            ("above_point", 80),
            ("becomes_above", 84),
            ("below_point", 88),
            ("becomes_below", 92),
            ("bond_freq", 96),
            ("heat_release", 100),
            ("growth_period", 104),
            ("sprouts", 108),
        ];
        let actual = [
            ("colour", std::mem::offset_of!(MaterialParams, colour)),
            ("freq_n", std::mem::offset_of!(MaterialParams, freq_n)),
            ("freq_t", std::mem::offset_of!(MaterialParams, freq_t)),
            ("zeta_n", std::mem::offset_of!(MaterialParams, zeta_n)),
            ("zeta_t", std::mem::offset_of!(MaterialParams, zeta_t)),
            ("mu", std::mem::offset_of!(MaterialParams, mu)),
            ("density", std::mem::offset_of!(MaterialParams, density)),
            ("radius", std::mem::offset_of!(MaterialParams, radius)),
            (
                "rest_packing",
                std::mem::offset_of!(MaterialParams, rest_packing),
            ),
            (
                "pressure_k",
                std::mem::offset_of!(MaterialParams, pressure_k),
            ),
            ("viscosity", std::mem::offset_of!(MaterialParams, viscosity)),
            ("is_static", std::mem::offset_of!(MaterialParams, is_static)),
            ("spacing", std::mem::offset_of!(MaterialParams, spacing)),
            (
                "conductivity",
                std::mem::offset_of!(MaterialParams, conductivity),
            ),
            (
                "heat_capacity",
                std::mem::offset_of!(MaterialParams, heat_capacity),
            ),
            (
                "default_temperature",
                std::mem::offset_of!(MaterialParams, default_temperature),
            ),
            (
                "thermal_expansion",
                std::mem::offset_of!(MaterialParams, thermal_expansion),
            ),
            (
                "above_point",
                std::mem::offset_of!(MaterialParams, above_point),
            ),
            (
                "becomes_above",
                std::mem::offset_of!(MaterialParams, becomes_above),
            ),
            (
                "below_point",
                std::mem::offset_of!(MaterialParams, below_point),
            ),
            (
                "becomes_below",
                std::mem::offset_of!(MaterialParams, becomes_below),
            ),
            ("bond_freq", std::mem::offset_of!(MaterialParams, bond_freq)),
            (
                "heat_release",
                std::mem::offset_of!(MaterialParams, heat_release),
            ),
            (
                "growth_period",
                std::mem::offset_of!(MaterialParams, growth_period),
            ),
            ("sprouts", std::mem::offset_of!(MaterialParams, sprouts)),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn lava_conductivity_is_what_the_table_says() {
        assert_eq!(MATERIALS[LAVA as usize].params.conductivity, 250.0);
    }
}
