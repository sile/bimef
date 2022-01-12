#[derive(Debug)]
pub struct Bimef {
    /// Enhancement ratio.
    mu: f32,

    /// Camera response model parameter.
    _a: f32,

    /// Camera response model parameter.
    _b: f32,
}

impl Bimef {
    pub fn new() -> Self {
        Self {
            mu: 0.5,
            _a: -0.3293,
            _b: 1.1258,
        }
    }

    pub fn enhance(&self, image: Image<Rgb<f32>>) -> Image<Rgb<u8>> {
        let start = std::time::Instant::now();

        let k = self.max_entropy_enhance(&image);
        println!("Elapsed(2): {:?}", start.elapsed());

        let camera = CameraResponseModel::new(k);

        let fused = image.map(|_, p0| {
            let w = p0.illumination().powf(self.mu);
            let r0 = p0.r * w;
            let g0 = p0.g * w;
            let b0 = p0.b * w;

            let p1 = camera.apply(&p0);
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

    fn max_entropy_enhance(&self, image: &Image<Rgb<f32>>) -> f32 {
        let n = 50 * 50;
        let m = std::cmp::max(1, image.pixels.len() / n);
        let y = image
            .pixels
            .iter()
            .filter(|x| x.illumination() < 0.5)
            .enumerate()
            .filter(|x| x.0 % m == 0) // TODO: Move before the first filter()
            .map(|x| x.1.gray())
            .collect::<Vec<_>>();

        struct FindNegativeEntropy {
            y: Vec<f32>,
        }

        impl FindNegativeEntropy {
            fn apply(&self, k: f32) -> f32 {
                let applied_k = apply_k(&self.y, k).map(|v| to_u8(v) as usize);
                let mut hist = [0; 256];
                for v in applied_k {
                    hist[v] += 1;
                }
                let n = self.y.len();
                let negative_entropy = hist
                    .into_iter()
                    .filter(|&v| v != 0)
                    .map(|v| v as f32 / n as f32)
                    .map(|v| v * v.log2())
                    .sum::<f32>();
                negative_entropy
            }
        }

        let mut optim =
            tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(1.0, 7.9).unwrap());

        let mut best_value = std::f32::INFINITY;
        let mut best_k = 1.0;
        let mut rng = rand::thread_rng();
        let problem = FindNegativeEntropy { y };
        let start = std::time::Instant::now();

        let mut low_param = 1.0;
        let mut high_param = 7.9;
        let mut middle_param = (high_param - low_param) / 2.0 + low_param;

        let mut low_value = problem.apply(low_param);
        let mut best_k = low_param;
        let mut best_value = low_value;

        let mut middle_value = problem.apply(middle_param);
        if middle_value < best_value {
            best_value = middle_value;
            best_k = middle_param
        }

        let mut high_value = problem.apply(high_param);
        if high_value < best_value {
            best_value = high_value;
            best_k = high_param;
        }

        for k in (0..50).map(|i| 2.0 + (0.1 * i as f32)) {
            println!("[{}] {}", k, problem.apply(k));
        }

        for i in 0..0 {
            println!(
                "# low={}({}), middle={}({}), high={}({})",
                low_value, low_param, middle_value, middle_param, high_value, high_param
            );

            let (k, value) = if low_value <= middle_value && middle_value <= high_value {
                let param = (middle_param - low_param) / 2.0 + low_param;
                let value = problem.apply(param);
                (param, value)
            } else if low_value >= middle_value && middle_value >= high_value {
                let param = (high_param - middle_param) / 2.0 + middle_param;
                let value = problem.apply(param);
                (param, value)
            } else {
                assert!(
                    low_value >= middle_value && middle_value <= high_value,
                    "low={}, middle={}, high={}",
                    low_value,
                    middle_value,
                    high_value
                );
                if middle_param - low_param > high_param - middle_param {
                    let param = (middle_param - low_param) / 2.0 + low_param;
                    let value = problem.apply(param);
                    (param, value)
                } else {
                    let param = (high_param - middle_param) / 2.0 + middle_param;
                    let value = problem.apply(param);
                    (param, value)
                }
            };

            if value < best_value {
                best_value = value;
                best_k = k;
            } else if (value - best_value) < 1.0e-5 {
                println!("break: {}", i);
                break;
            }

            println!("=> {}, {} ({})", k, value, best_value);
            if k <= middle_param {
                if value <= low_value && value <= middle_value {
                    high_param = middle_param;
                    high_value = middle_value;
                    middle_param = k;
                    middle_value = value;
                } else if value <= low_value {
                    low_param = k;
                    low_value = value;
                } else {
                    assert!(low_value <= value && value <= middle_value);
                    high_param = middle_param;
                    high_value = middle_value;
                    middle_param = k;
                    middle_value = value;
                }
            } else {
                if value <= middle_value && value <= high_value {
                    low_param = middle_param;
                    low_value = middle_value;
                    middle_param = k;
                    middle_value = value;
                } else if value <= high_value {
                    high_param = k;
                    high_value = value;
                } else {
                    assert!(middle_value <= value && value <= high_value);
                    low_param = middle_param;
                    low_value = middle_value;
                    middle_param = k;
                    middle_value = value;
                }
            }
        }
        println!(
            "Optimized(k={},v={}): {:?}",
            best_k,
            best_value,
            start.elapsed()
        );

        for i in 0..50 {
            let k = optim.ask(&mut rng).unwrap();
            let v = problem.apply(k as f32);
            optim.tell(k, v as f64).unwrap();
            let do_break = i > 50 && (best_value.abs() - v.abs()) < 1.0e-5;
            if v < best_value {
                best_value = v;
                best_k = k as f32;
            }
            if do_break {
                println!("break: {}", i);
                break;
            }
        }
        println!(
            "Optimized(k={},v={}): {:?}",
            best_k,
            best_value,
            start.elapsed()
        );

        best_k as f32
    }
}

pub fn apply_k(xs: &[f32], k: f32) -> impl '_ + Iterator<Item = f32> {
    let a = -0.3293;
    let b = 1.1258;
    let beta = ((1.0 - k.powf(a)) * b).exp();
    let gamma = k.powf(a);
    xs.iter().map(move |p| p.powf(gamma) * beta)
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

    pub fn to_rgb_f32(&self) -> Image<Rgb<f32>> {
        let pixels = self.pixels.iter().map(|p| p.to_rgb_f32()).collect();
        Image {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

impl Image<Rgb<f32>> {
    // brightness component.
    pub fn to_gray(&self) -> Image<f32> {
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

    pub fn get_illumination_map(&self) -> Vec<f32> {
        // TODO: Vec<u8>
        self.pixels.iter().map(|p| p.max_value()).collect()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rgb<T = u8> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl Rgb<u8> {
    pub fn to_rgb_f32(self) -> Rgb<f32> {
        let n = u8::MAX as f32;
        Rgb {
            r: self.r as f32 / n,
            g: self.g as f32 / n,
            b: self.b as f32 / n,
        }
    }
}

impl Rgb<f32> {
    pub fn max_value(&self) -> f32 {
        self.r.max(self.g.max(self.b))
    }

    pub fn illumination(&self) -> f32 {
        self.max_value()
    }

    pub fn gray(&self) -> f32 {
        (self.r * self.g * self.b).powf(1.0 / 3.0)
    }
}

fn to_u8(v: f32) -> u8 {
    (v.max(0.0).min(1.0) * 255.0).round() as u8
}

#[derive(Debug)]
pub struct CameraResponseModel {
    beta: f32,
    gamma: f32,
}

impl CameraResponseModel {
    fn new(k: f32) -> Self {
        let a = -0.3293;
        let b = 1.1258;
        let beta = ((1.0 - k.powf(a)) * b).exp();
        let gamma = k.powf(a);

        Self { beta, gamma }
    }

    fn apply(&self, rgb: &Rgb<f32>) -> Rgb<f32> {
        let gamma = self.gamma;
        let beta = self.beta;
        Rgb {
            r: rgb.r.powf(gamma) * beta - 0.01,
            g: rgb.g.powf(gamma) * beta - 0.01,
            b: rgb.b.powf(gamma) * beta - 0.01,
        }
    }
}
