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

        let illumination_map = image.get_illumination_map();
        println!("Elapsed(0): {:?}", start.elapsed());

        let j = self.max_entropy_enhance(&image, &illumination_map);
        println!("Elapsed(2): {:?}", start.elapsed());

        // Weight matrix.
        let fused = image.map(|i, p0| {
            let w = illumination_map[i].powf(self.mu);
            let r0 = p0.r * w;
            let g0 = p0.g * w;
            let b0 = p0.b * w;

            let p1 = j.pixels[i];
            let r1 = p1.r * (1.0 - w);
            let g1 = p1.g * (1.0 - w);
            let b1 = p1.b * (1.0 - w);

            Rgb {
                r: to_u8(r0 + r1),
                g: to_u8(g0 + g1),
                b: to_u8(b0 + b1),
            }
        });

        fused
    }

    fn max_entropy_enhance(
        &self,
        image: &Image<Rgb<f64>>,
        illumination_map: &[f64],
    ) -> Image<Rgb<f64>> {
        // TODO: resize 50x50
        let y = image.to_gray();
        let n = 50 * 50;
        let m = std::cmp::max(1, y.pixels.len() / n);
        let y = y
            .pixels
            .iter()
            .copied()
            .zip(illumination_map.iter().copied())
            .filter(|x| x.1 < 0.5)
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

    pub fn get_illumination_map(&self) -> Vec<f64> {
        // TODO: Vec<u8>
        self.pixels.iter().map(|p| p.max_value()).collect()
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

fn to_u8(v: f64) -> u8 {
    (v.max(0.0).min(1.0) * 255.0).round() as u8
}
