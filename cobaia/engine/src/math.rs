use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }

    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            (self.y * rhs.z) - (self.z * rhs.y),
            (self.z * rhs.x) - (self.x * rhs.z),
            (self.x * rhs.y) - (self.y * rhs.x),
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            return self;
        }
        self / len
    }

    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    pub fn rem_euclid(self, rhs: f32) -> Self {
        Self::new(
            self.x.rem_euclid(rhs),
            self.y.rem_euclid(rhs),
            self.z.rem_euclid(rhs),
        )
    }

    pub fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    pub fn clamp01(self) -> Self {
        Self::new(
            self.x.clamp(0.0, 1.0),
            self.y.clamp(0.0, 1.0),
            self.z.clamp(0.0, 1.0),
        )
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + (self.direction * t)
    }
}

pub fn reflect(direction: Vec3, normal: Vec3) -> Vec3 {
    direction - (normal * (2.0 * direction.dot(normal)))
}

pub fn refract(direction: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = (-direction).dot(normal).clamp(-1.0, 1.0);
    let k = 1.0 - (eta * eta * (1.0 - (cos_i * cos_i)));
    if k < 0.0 {
        None
    } else {
        Some((direction * eta) + (normal * ((eta * cos_i) - k.sqrt())))
    }
}

pub fn schlick(cosine: f32, eta_i: f32, eta_t: f32) -> f32 {
    let r0 = ((eta_i - eta_t) / (eta_i + eta_t)).powi(2);
    r0 + ((1.0 - r0) * (1.0 - cosine).powf(5.0))
}

fn fract(v: f32) -> f32 {
    v - v.floor()
}

fn hash31(p: Vec3) -> f32 {
    let n = p.dot(Vec3::new(127.1, 311.7, 74.7));
    fract(n.sin() * 43758.5453)
}

pub fn frosted_offset(p: Vec3, seed: f32) -> Vec3 {
    let s = seed * 13.37;
    let j = Vec3::new(
        (hash31(p + Vec3::new(0.17 + s, 4.7, 9.2)) * 2.0) - 1.0,
        (hash31(p + Vec3::new(5.31, 1.2 + s, 3.4)) * 2.0) - 1.0,
        (hash31(p + Vec3::new(7.83, 8.1, 2.6 + s)) * 2.0) - 1.0,
    );
    j.normalize()
}

pub fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

pub fn random01(seed: u32) -> f32 {
    hash_u32(seed) as f32 / u32::MAX as f32
}

pub fn sample_jitter(x: u32, y: u32, sample: u32, axis: u32) -> f32 {
    let seed = x
        .wrapping_mul(1973)
        .wrapping_add(y.wrapping_mul(9277))
        .wrapping_add(sample.wrapping_mul(26699))
        .wrapping_add(axis.wrapping_mul(104_729))
        ^ 0x68bc_21eb;
    random01(seed)
}
