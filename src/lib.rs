use std::collections::BTreeMap;

#[derive(Debug)]
pub struct Bimef {
    /// Enhancement ratio.
    mu: f64,

    /// Camera response model parameter.
    _a: f64,

    /// Camera response model parameter.
    _b: f64,
}

impl Bimef {
    pub fn new() -> Self {
        Self {
            mu: 0.5,
            _a: -0.3293,
            _b: 1.1258,
        }
    }

    pub fn enhance(&self, image: Image<Rgb<f64>>) -> Image<Rgb<u8>> {
        let start = std::time::Instant::now();

        let im = IlluminationMap::new(&image);
        let t_our = im.iter_f2().collect::<Vec<_>>();
        let is_bad = t_our.iter().map(|&v| v < 0.5).collect::<Vec<_>>();
        let bads = is_bad.iter().copied().filter(|&b| b).count();
        let goods = is_bad.len() - bads;
        let j = self.max_entropy_enhance(&image, &is_bad);

        // Weight matrix.
        let image2 = image.map(|i, rgb| {
            let w = t_our[i].powf(self.mu);
            Rgb {
                r: rgb.r * w,
                g: rgb.g * w,
                b: rgb.b * w,
            }
        });
        let j2 = j.map(|i, rgb| {
            let w = t_our[i].powf(self.mu);
            Rgb {
                r: rgb.r * (1.0 - w),
                g: rgb.g * (1.0 - w),
                b: rgb.b * (1.0 - w),
            }
        });

        let fused = image2.map(|i, a| {
            let b = j2.pixels[i];
            fn to_u8(v: f64) -> u8 {
                (v.max(0.0).min(1.0) * 255.0).round() as u8
            }
            Rgb {
                r: to_u8(a.r + b.r),
                g: to_u8(a.g + b.g),
                b: to_u8(a.b + b.b),
            }
        });
        println!("Elapsed: {:?}", start.elapsed());
        fused
    }

    fn max_entropy_enhance(&self, image: &Image<Rgb<f64>>, is_bad: &[bool]) -> Image<Rgb<f64>> {
        // TODO: resize 50x50
        let y = image.to_gray();
        let n = 50 * 50;
        let m = std::cmp::max(1, y.pixels.len() / n);
        let y = y
            .pixels
            .iter()
            .copied()
            .zip(is_bad.iter().copied())
            .filter(|x| x.1)
            .map(|x| x.0)
            .enumerate()
            .filter(|x| x.0 % m == 0)
            .map(|x| x.1)
            .collect::<Vec<_>>();

        struct FindNegativeEntropy {
            y: Vec<f64>,
        }

        impl FindNegativeEntropy {
            fn apply(&self, k: f64) -> f64 {
                let applied_k = apply_k(&self.y, k);
                let int_applied_k = applied_k
                    .into_iter()
                    .map(|v| (v.max(0.0).min(1.0) * 255.0).round() as u8)
                    .collect::<Vec<_>>();
                let mut hist = BTreeMap::<u8, f64>::new();
                for v in int_applied_k {
                    *hist.entry(v).or_default() += 1.0;
                }
                for v in hist.values_mut() {
                    *v /= self.y.len() as f64;
                }
                let negative_entropy = hist.values().map(|&v| v * v.log2()).sum::<f64>();
                negative_entropy
            }
        }

        let mut optim =
            tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(1.0, 7.9).unwrap());

        let mut best_value = std::f64::INFINITY;
        let mut best_k = 1.0;
        let mut rng = rand::thread_rng();
        let problem = FindNegativeEntropy { y };
        let start = std::time::Instant::now();
        for i in 0..500 {
            let k = optim.ask(&mut rng).unwrap();
            let v = problem.apply(k);
            optim.tell(k, v).unwrap();
            let do_break = i > 50 && (best_value.abs() - v.abs()) < 1.0e-5;
            if v < best_value {
                best_value = v;
                best_k = k;
            }
            if do_break {
                println!("break: {}", i);
                break;
            }
        }
        println!("Optimized: {:?}", start.elapsed());

        image.apply_k(best_k)
    }
}

#[derive(Debug)]
pub struct IlluminationMap {
    values: Vec<Vec<f64>>,
}

impl IlluminationMap {
    pub fn print(&self) {
        for v in &self.values {
            println!("{:?}", v);
        }
    }

    pub fn new(image: &Image<Rgb<f64>>) -> Self {
        let mut values = vec![vec![0.0; image.width as usize]; image.height as usize];
        for y in 0..image.height as usize {
            for x in 0..image.width as usize {
                values[y][x] = image.pixels[y * image.width as usize + x].max_value();
            }
        }
        Self { values }
    }

    pub fn iter_f(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| (0..y_len).map(move |y| self.values[y][x]))
    }

    pub fn iter_f2(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..y_len).flat_map(move |y| (0..x_len).map(move |x| self.values[y][x]))
    }

    pub fn iter_2x(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| {
            (0..y_len).map(move |y| self.values[y][x.checked_sub(1).unwrap_or(x_len - 1)])
        })
    }

    pub fn iter_2y(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| {
            (0..y_len).map(move |y| self.values[y.checked_sub(1).unwrap_or(y_len - 1)][x])
        })
    }

    pub fn iter_3x(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| {
            (0..y_len).map(move |y| {
                if x == 0 {
                    self.values[y][x_len - 1]
                } else {
                    0.0
                }
            })
        })
    }

    pub fn iter_3y(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| {
            (0..y_len).map(move |y| {
                if y == 0 {
                    self.values[y_len - 1][x]
                } else {
                    0.0
                }
            })
        })
    }

    pub fn iter_4x(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| {
            (0..y_len).map(move |y| {
                if x == x_len - 1 {
                    0.0
                } else {
                    self.values[y][x]
                }
            })
        })
    }

    pub fn iter_4y(&self) -> impl '_ + Iterator<Item = f64> {
        let x_len = self.x_len();
        let y_len = self.y_len();
        (0..x_len).flat_map(move |x| {
            (0..y_len).map(move |y| {
                if y == y_len - 1 {
                    0.0
                } else {
                    self.values[y][x]
                }
            })
        })
    }

    pub fn calc_weights(&mut self, other: &Self, sharpness: f64) {
        for y in 0..self.y_len() {
            for x in 0..self.x_len() {
                let w = 1.0 / (self.values[y][x].abs() * other.values[y][x].abs() + sharpness);
                self.values[y][x] = w;
            }
        }
    }

    pub fn convolve_h(&self, sigma: usize) -> Self {
        let mut values = vec![vec![0.0; self.values[0].len()]; self.values.len()];
        for y in 0..self.y_len() {
            for x in 0..self.x_len() {
                values[y][x] = (0..sigma)
                    .map(|i| {
                        (x + i)
                            .checked_sub(sigma / 2)
                            .and_then(|i| self.values[y].get(i).copied())
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>();
            }
        }
        Self { values }
    }

    pub fn convolve_v(&self, sigma: usize) -> Self {
        let mut values = vec![vec![0.0; self.values[0].len()]; self.values.len()];
        for y in 0..self.y_len() {
            for x in 0..self.x_len() {
                values[y][x] = (0..sigma)
                    .map(|i| {
                        (y + i)
                            .checked_sub(sigma / 2)
                            .and_then(|i| self.values.get(i).map(|row| row[x]))
                            .unwrap_or(0.0)
                    })
                    .sum::<f64>();
            }
        }
        Self { values }
    }

    pub fn x_len(&self) -> usize {
        self.values[0].len()
    }

    pub fn y_len(&self) -> usize {
        self.values.len()
    }

    pub fn diff_h(&self) -> Self {
        let mut values = Vec::new();
        for row in &self.values {
            values.push(
                row.iter()
                    .copied()
                    .zip(
                        row.iter()
                            .copied()
                            .skip(1)
                            .chain(row.iter().copied().take(1)),
                    )
                    .map(|(a, b)| b - a)
                    .collect::<Vec<_>>(),
            );
        }
        Self { values }
    }

    pub fn diff_v(&self) -> Self {
        let mut values = Vec::new();
        for (row0, row1) in self
            .values
            .iter()
            .zip(self.values.iter().skip(1).chain(self.values.iter().take(1)))
        {
            values.push(
                row0.iter()
                    .copied()
                    .zip(row1.iter().copied())
                    .map(|(a, b)| b - a)
                    .collect::<Vec<_>>(),
            );
        }
        Self { values }
    }
}

pub fn apply_k(xs: &[f64], k: f64) -> Vec<f64> {
    let a = -0.3293;
    let b = 1.1258;
    let beta = ((1.0 - k.powf(a)) * b).exp();
    let gamma = k.powf(a);
    xs.iter().map(|p| p.powf(gamma) * beta).collect()
}

#[derive(Debug)]
pub struct Image<T> {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<T>,
}

impl<T> Image<T> {
    pub fn map<F, U>(&self, f: F) -> Image<U>
    where
        F: Fn(usize, &T) -> U,
    {
        Image {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .enumerate()
                .map(|(i, x)| f(i, x))
                .collect(),
        }
    }
}

impl Image<Rgb<u8>> {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity((self.width * self.height * 3) as usize);
        for p in &self.pixels {
            buf.push(p.r);
            buf.push(p.g);
            buf.push(p.b);
        }
        buf
    }

    pub fn from_bytes(width: u32, height: u32, bytes: &[u8]) -> Self {
        let n = (width * height) as usize;
        assert_eq!(n * 3, bytes.len());

        let mut pixels = Vec::with_capacity(n);
        for i in 0..n {
            pixels.push(Rgb {
                r: bytes[i * 3],
                g: bytes[i * 3 + 1],
                b: bytes[i * 3 + 2],
            });
        }

        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn to_rgb_f64(&self) -> Image<Rgb<f64>> {
        let pixels = self.pixels.iter().map(|p| p.to_rgb_f64()).collect();
        Image {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

impl Image<Rgb<f64>> {
    // brightness component.
    pub fn to_gray(&self) -> Image<f64> {
        Image {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .map(|p| (p.r * p.g * p.b).powf(1.0 / 3.0))
                .collect(),
        }
    }

    pub fn apply_k(&self, k: f64) -> Self {
        let a = -0.3293;
        let b = 1.1258;
        let beta = ((1.0 - k.powf(a)) * b).exp();
        let gamma = k.powf(a);
        self.map(|_, rgb| Rgb {
            r: rgb.r.powf(gamma) * beta - 0.01,
            g: rgb.g.powf(gamma) * beta - 0.01,
            b: rgb.b.powf(gamma) * beta - 0.01,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rgb<T = u8> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl Rgb<u8> {
    pub fn to_rgb_f64(self) -> Rgb<f64> {
        let n = u8::MAX as f64;
        Rgb {
            r: self.r as f64 / n,
            g: self.g as f64 / n,
            b: self.b as f64 / n,
        }
    }
}

impl Rgb<f64> {
    pub fn max_value(&self) -> f64 {
        self.r.max(self.g.max(self.b))
    }
}
