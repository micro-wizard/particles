pub const HEIGHT: i32 = 7;

const SPACING: i32 = 1;

pub struct Glyph {
    pub width: i32,
    pub rows: [u8; HEIGHT as usize],
}

const fn glyph(width: i32, rows: [u8; HEIGHT as usize]) -> Glyph {
    Glyph { width, rows }
}

const SPACE: Glyph = glyph(3, [0; 7]);

fn glyph_for(c: char) -> &'static Glyph {
    const A: Glyph = glyph(
        5,
        [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    );
    const B: Glyph = glyph(
        5,
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
    );
    const C: Glyph = glyph(
        5,
        [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
    );
    const D: Glyph = glyph(
        5,
        [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
    );
    const E: Glyph = glyph(
        5,
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
    );
    const F: Glyph = glyph(
        5,
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    );
    const G: Glyph = glyph(
        5,
        [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
    );
    const H: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    );
    const I: Glyph = glyph(3, [0b111, 0b010, 0b010, 0b010, 0b010, 0b010, 0b111]);
    const J: Glyph = glyph(
        5,
        [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
    );
    const K: Glyph = glyph(
        5,
        [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
    );
    const L: Glyph = glyph(
        5,
        [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
    );
    const M: Glyph = glyph(
        5,
        [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
    );
    const N: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001,
        ],
    );
    const O: Glyph = glyph(
        5,
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    );
    const P: Glyph = glyph(
        5,
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    );
    const Q: Glyph = glyph(
        5,
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
    );
    const R: Glyph = glyph(
        5,
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
    );
    const S: Glyph = glyph(
        5,
        [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    );
    const T: Glyph = glyph(
        5,
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    );
    const U: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    );
    const V: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
    );
    const W: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
    );
    const X: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
    );
    const Y: Glyph = glyph(
        5,
        [
            0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100,
        ],
    );
    const Z: Glyph = glyph(
        5,
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
    );
    match c.to_ascii_uppercase() {
        'A' => &A,
        'B' => &B,
        'C' => &C,
        'D' => &D,
        'E' => &E,
        'F' => &F,
        'G' => &G,
        'H' => &H,
        'I' => &I,
        'J' => &J,
        'K' => &K,
        'L' => &L,
        'M' => &M,
        'N' => &N,
        'O' => &O,
        'P' => &P,
        'Q' => &Q,
        'R' => &R,
        'S' => &S,
        'T' => &T,
        'U' => &U,
        'V' => &V,
        'W' => &W,
        'X' => &X,
        'Y' => &Y,
        'Z' => &Z,
        _ => &SPACE,
    }
}

#[cfg(test)]
pub fn has_glyph(c: char) -> bool {
    !std::ptr::eq(glyph_for(c), &SPACE)
}

pub fn width(text: &str) -> i32 {
    let glyphs: i32 = text.chars().map(|c| glyph_for(c).width).sum();
    glyphs + SPACING * (text.chars().count() as i32 - 1).max(0)
}

pub fn pixel_size(target: f32, pixels_per_point: f32) -> f32 {
    (target * pixels_per_point).round().max(1.0) / pixels_per_point
}

#[derive(Copy, Clone)]
pub struct Grid {
    origin: egui::Pos2,
    pixel: f32,
}

impl Grid {
    pub fn new(near: egui::Pos2, target: f32, pixels_per_point: f32) -> Self {
        let snap = |v: f32| (v * pixels_per_point).round() / pixels_per_point;
        Self {
            origin: egui::pos2(snap(near.x), snap(near.y)),
            pixel: pixel_size(target, pixels_per_point),
        }
    }

    pub fn cell(&self, point: egui::Pos2) -> [i32; 2] {
        let local = (point - self.origin) / self.pixel;
        [local.x.floor() as i32, local.y.floor() as i32]
    }

    pub fn rect(&self, x: i32, y: i32, w: i32, h: i32) -> egui::Rect {
        egui::Rect::from_min_size(
            self.origin + egui::vec2(x as f32, y as f32) * self.pixel,
            egui::vec2(w as f32, h as f32) * self.pixel,
        )
    }

    pub fn fill(
        &self,
        painter: &egui::Painter,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        colour: egui::Color32,
    ) {
        painter.rect_filled(self.rect(x, y, w, h), 0.0, colour);
    }

    pub fn bitmap(
        &self,
        painter: &egui::Painter,
        x: i32,
        y: i32,
        glyph: &Glyph,
        colour: egui::Color32,
    ) {
        for (row, bits) in glyph.rows.iter().enumerate() {
            let lit = |col: i32| bits >> (glyph.width - 1 - col) & 1 == 1;
            let mut col = 0;
            while col < glyph.width {
                if lit(col) {
                    let start = col;
                    while col < glyph.width && lit(col) {
                        col += 1;
                    }
                    self.fill(painter, x + start, y + row as i32, col - start, 1, colour);
                } else {
                    col += 1;
                }
            }
        }
    }

    pub fn text(&self, painter: &egui::Painter, x: i32, y: i32, text: &str, colour: egui::Color32) {
        let mut left = x;
        for c in text.chars() {
            let glyph = glyph_for(c);
            self.bitmap(painter, left, y, glyph, colour);
            left += glyph.width + SPACING;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_glyph_fits_its_width() {
        for c in 'A'..='Z' {
            let glyph = glyph_for(c);
            assert!(glyph.rows.iter().all(|row| row >> glyph.width == 0), "{c}");
        }
    }

    #[test]
    fn a_grid_pixel_is_whole_screen_pixels() {
        for ppp in [0.75, 1.0, 1.25, 1.5, 2.0, 3.0] {
            let screen = pixel_size(1.75, ppp) * ppp;
            assert_eq!(screen, screen.round(), "{ppp}");
            assert!(screen >= 1.0);
        }
    }
}
