use std::collections::BTreeMap;

#[derive(Debug)]
pub struct Bimef {
    /// Enhancement ratio.
    mu: f64,

    /// Exposure ratio.
    k: Option<Vec<f32>>,

    /// Camera response model parameter.
    _a: f64,

    /// Camera response model parameter.
    _b: f64,
}

impl Bimef {
    pub fn new() -> Self {
        Self {
            mu: 0.5,
            k: None,
            _a: -0.3293,
            _b: 1.1258,
        }
    }

    pub fn enhance(&self, image: Image<Rgb<f64>>) -> Image<Rgb<u8>> {
        let lamb = 0.5;
        let sigma = 5.0;

        let im = IlluminationMap::new(&image);

        // TODO: resize 0.5 => tsmooth => resize 2
        let t_our = self.tsmooth(&im, lamb, sigma);

        let j = if self.k.is_none() {
            let is_bad = t_our.iter().map(|&v| v < 0.5).collect::<Vec<_>>();
            let bads = is_bad.iter().copied().filter(|&b| b).count();
            let goods = is_bad.len() - bads;
            dbg!((goods, bads));
            self.max_entropy_enhance(&image, &is_bad)
        } else {
            todo!()
        };

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

        fused
    }

    fn max_entropy_enhance(&self, image: &Image<Rgb<f64>>, is_bad: &[bool]) -> Image<Rgb<f64>> {
        // TODO: resize 50x50
        let y = image.to_gray();
        let y = y
            .pixels
            .iter()
            .copied()
            .zip(is_bad.iter().copied())
            .filter(|x| x.1)
            .map(|x| x.0)
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

    fn tsmooth(&self, im: &IlluminationMap, lamb: f64, sigma: f64) -> Vec<f64> {
        let sharpness = 0.001;
        let (wx, wy) = self.compute_texture_weights(im, sigma, sharpness);
        self.solve_linear_equation(im, &wx, &wy, lamb)
    }

    fn solve_linear_equation(
        &self,
        im: &IlluminationMap,
        wx: &IlluminationMap,
        wy: &IlluminationMap,
        lamb: f64,
    ) -> Vec<f64> {
        let r = im.y_len() as isize;
        let c = im.x_len() as isize;
        let k = r * c;
        let dx = wx.iter_f().map(|v| -lamb * v).collect::<Vec<_>>();
        let dy = wy.iter_f().map(|v| -lamb * v).collect::<Vec<_>>();
        let dxa = wx.iter_2x().map(|v| -lamb * v).collect::<Vec<_>>();
        let dya = wy.iter_2y().map(|v| -lamb * v).collect::<Vec<_>>();
        let dxd1 = wx.iter_3x().map(|v| -lamb * v).collect::<Vec<_>>();
        let dyd1 = wy.iter_3y().map(|v| -lamb * v).collect::<Vec<_>>();
        let dxd2 = wx.iter_4x().map(|v| -lamb * v).collect::<Vec<_>>();
        let dyd2 = wy.iter_4y().map(|v| -lamb * v).collect::<Vec<_>>();

        let ax = SparseMatrix::from_diags([(dxd1, -k + r), (dxd2, -r)].into_iter(), k as usize);
        let ay = SparseMatrix::from_diags([(dyd1, -r + 1), (dyd2, -1)].into_iter(), k as usize);

        let mut d = Vec::with_capacity(dx.len());
        for i in 0..dx.len() {
            d.push(1.0 - (dx[i] + dy[i] + dxa[i] + dya[i]));
        }

        let axy = ax.add(ay);
        // let axy_t = axy.t();
        // let a = axy.add(axy_t.add(SparseMatrix::from_diag(d)));
        let a = axy.add(SparseMatrix::from_diag(d));

        let tin = im.iter_f().collect::<Vec<_>>();

        let start = std::time::Instant::now();
        let factor = a.cholesky();
        println!("Cholesky: {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let tout = factor.solve(&a, &tin);
        println!("Solve: {:?}", start.elapsed());

        tout
    }

    fn compute_texture_weights(
        &self,
        im: &IlluminationMap,
        sigma: f64,
        sharpness: f64,
    ) -> (IlluminationMap, IlluminationMap) {
        let dt0_v = im.diff_v();
        let dt0_h = im.diff_h();

        let mut gauker_v = dt0_v.convolve_v(sigma as usize);
        let mut gauker_h = dt0_h.convolve_h(sigma as usize);

        gauker_v.calc_weights(&dt0_v, sharpness);
        gauker_h.calc_weights(&dt0_h, sharpness);
        (gauker_h, gauker_v)
    }
}

#[derive(Debug, Clone)]
pub struct Factor {
    d: Vec<f64>,
    l: BTreeMap<(usize, usize), f64>,
}

impl Factor {
    // ICCGSolver
    pub fn solve(&self, a: &SparseMatrix, b: &[f64]) -> Vec<f64> {
        let n = self.d.len();
        let mut x = vec![0.0; n];
        let mut r = b.to_owned();

        let lu_u = self.lu_u();
        let mut p = vec![0.0; n];
        self.icres(&lu_u, &r, &mut p);

        fn dot(a: &[f64], b: &[f64]) -> f64 {
            assert_eq!(a.len(), b.len());
            a.iter()
                .copied()
                .zip(b.iter().copied())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        }

        let mut r2 = vec![0.0; n];
        let mut rr0 = dot(&r, &p);
        let max_iter = 20;
        let eps = 0.001;
        let mut y = vec![0.0; n];

        for k in 0..max_iter {
            // y = AP
            for i in 0..n {
                y[i] = a.dot(i, &p);
            }

            let alpha = rr0 / dot(&p, &y);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * y[i];
            }

            self.icres(&lu_u, &r, &mut r2);
            let rr1 = dot(&r, &r2);

            let e = rr1.sqrt();
            dbg!((rr1, e));
            if e < eps {
                dbg!((k, e));
                break;
            }

            let beta = rr1 / rr0;
            for i in 0..n {
                p[i] = r2[i] + beta * p[i];
            }

            rr0 = rr1;
        }

        x
    }

    fn icres(&self, lu_u: &BTreeMap<(usize, usize), f64>, r: &[f64], u: &mut [f64]) {
        let n = self.d.len();
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let mut rly = r[i];
            for (&(_, j), &v) in self.l.range((i, 0)..(i, i)) {
                rly -= v * y[j];
            }
            y.push(rly / self.l[&(i, i)]);
        }

        for i in (0..n).rev() {
            let mut lu = 0.0;
            for (&(_, j), &v) in lu_u.range((i, i + 1)..(i, n)) {
                lu += v * u[j];
            }
            u[i] = y[i] - self.d[i] * lu;
        }
    }

    fn lu_u(&self) -> BTreeMap<(usize, usize), f64> {
        self.l.iter().map(|(&(y, x), &v)| ((x, y), v)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    size: usize,
    matrix: BTreeMap<(usize, usize), f64>,
}

impl SparseMatrix {
    pub fn dot(&self, y: usize, rhs: &[f64]) -> f64 {
        self.matrix
            .range((y, 0)..(y, self.size))
            .map(|(&(_, x), &v)| v * rhs[x])
            .sum::<f64>()
    }

    pub fn print(&self) {
        println!("[");
        for y in 0..self.size {
            print!("[");
            for x in 0..self.size {
                print!("{},", self.matrix.get(&(y, x)).copied().unwrap_or_default());
            }
            println!("],");
        }
        println!("]");
    }

    pub fn from_diags(diags: impl Iterator<Item = (Vec<f64>, isize)>, size: usize) -> Self {
        let mut matrix = BTreeMap::new();
        for (vs, offset) in diags {
            for (x, v) in vs.iter().copied().enumerate() {
                let y = x as isize - offset;
                if y >= 0 && (y as usize) < size {
                    if v != 0.0 {
                        matrix.insert((y as usize, x), v);
                    }
                } else {
                    // TODO: break if possible
                }
            }
        }
        Self { size, matrix }
    }

    pub fn from_diag(diag: Vec<f64>) -> Self {
        let size = diag.len();
        let mut matrix = BTreeMap::new();
        for (i, v) in diag.into_iter().enumerate() {
            matrix.insert((i, i), v);
        }
        Self { size, matrix }
    }

    pub fn t(&self) -> Self {
        Self {
            size: self.size,
            matrix: self
                .matrix
                .iter()
                .map(|(&(y, x), &v)| ((x, y), v))
                .collect(),
        }
    }

    // incomplete cholesky decomposition
    pub fn cholesky(&self) -> Factor {
        let mut d = Vec::new();
        let mut l = BTreeMap::<(usize, usize), f64>::new();

        for (&(i, j), &v) in &self.matrix {
            let mut lld = v;
            let mut vs_i = l.range((i, 0)..(i, j)).peekable();
            let mut vs_j = l.range((j, 0)..(j, j)).peekable();
            while vs_i.peek().is_some() && vs_j.peek().is_some() {
                let k_i = vs_i.peek().unwrap().0 .1;
                let k_j = vs_j.peek().unwrap().0 .1;
                if k_i == k_j {
                    lld -= vs_i.next().unwrap().1 * vs_j.next().unwrap().1 * d[k_i];
                } else if k_i < k_j {
                    vs_i.next();
                } else {
                    vs_j.next();
                }
            }
            l.insert((i, j), lld);

            if i == j {
                d.push(1.0 / l[&(i, i)]);
            }
        }
        assert_eq!(d.len(), self.size);

        Factor { d, l }
    }

    // pub fn cholesky(&self) -> Factor {
    //     dbg!(self.matrix.len());
    //     let mut d = Vec::new();
    //     let mut l = BTreeMap::new();

    //     for (&(i, j), &v) in &self.matrix {
    //         if i == j {
    //             let mut ld = v;
    //             for (&(_, k), &v) in l.range((i, 0)..(i, i)) {
    //                 ld -= v * v * d[k];
    //             }
    //             d.push(ld);
    //             l.insert((i, i), 1.0);
    //         } else {
    //             let mut lld = v;
    //             let mut vs_i = l.range((i, 0)..(i, j)).peekable();
    //             let mut vs_j = l.range((j, 0)..(j, j)).peekable();
    //             while vs_i.peek().is_some() && vs_j.peek().is_some() {
    //                 let k_i = vs_i.peek().unwrap().0 .1;
    //                 let k_j = vs_j.peek().unwrap().0 .1;
    //                 if k_i == k_j {
    //                     lld -= vs_i.next().unwrap().1 * vs_j.next().unwrap().1 * d[k_i];
    //                 } else if k_i < k_j {
    //                     vs_i.next();
    //                 } else {
    //                     vs_j.next();
    //                 }
    //             }

    //             l.insert((i, j), 1.0 / d[j] * lld);
    //         }
    //     }

    //     Factor { d, l }
    // }

    pub fn add(mut self, rhs: Self) -> Self {
        for (k, v) in rhs.matrix.into_iter() {
            *self.matrix.entry(k).or_default() += v;
        }
        self
    }

    pub fn get(&self, y: usize, x: usize) -> f64 {
        self.matrix.get(&(y, x)).copied().unwrap_or(0.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_works() {
        let a = [[4., 0.0, 0.0], [12., 37., 0.0], [-16., -43., 98.]];
        let mut m = SparseMatrix {
            size: 3,
            matrix: Default::default(),
        };
        for y in 0..3 {
            for x in 0..3 {
                if a[y][x] != 0.0 {
                    m.matrix.insert((y, x), a[y][x]);
                }
            }
        }

        let f = m.cholesky();
        println!("{:?}", f.solve(&m, &[4.0, 13.0, -11.0]));
        assert!(false);
    }
}
