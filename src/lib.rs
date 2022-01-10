use std::collections::HashMap;

#[derive(Debug)]
pub struct Bimef {
    /// Enhancement ratio.
    mu: f32,

    /// Exposure ratio.
    k: Option<Vec<f32>>,

    /// Camera response model parameter.
    a: f32,

    /// Camera response model parameter.
    b: f32,
}

impl Bimef {
    pub fn new() -> Self {
        Self {
            mu: 0.5,
            k: None,
            a: -0.3293,
            b: 1.1258,
        }
    }

    pub fn enhance(&self, image: Image<Rgb<f64>>) -> Image<Rgb<f64>> {
        let lamb = 0.5;
        let sigma = 5.0;

        let im = IlluminationMap::new(&image);

        // TODO: resize
        self.tsmooth(&im, lamb, sigma);

        image
    }

    fn tsmooth(&self, im: &IlluminationMap, lamb: f64, sigma: f64) {
        let sharpness = 0.001;
        let (wx, wy) = self.compute_texture_weights(im, sigma, sharpness);
        self.solve_linear_equation(im, &wx, &wy, lamb);
    }

    fn solve_linear_equation(
        &self,
        im: &IlluminationMap,
        wx: &IlluminationMap,
        wy: &IlluminationMap,
        lamb: f64,
    ) {
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
        let axy_t = axy.t();
        let a = axy.add(axy_t.add(SparseMatrix::from_diag(d)));
        a.print();

        let tin = im.iter_f().collect::<Vec<_>>();
        let factor = a.cholesky(false);
        println!("TIN: {:?}", tin);
        let tout = factor.solve(&tin);
        println!("{:?}", tout);
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
    l: HashMap<(usize, usize), f64>,
}

impl Factor {
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let l = self.lu_l();
        let u = self.lu_u();

        fn get(m: &HashMap<(usize, usize), f64>, y: usize, x: usize) -> f64 {
            m.get(&(y, x)).copied().unwrap_or(0.0)
        }

        let mut x = Vec::new();
        let n = self.d.len();

        // Forward substitution.
        for i in 0..n {
            let mut bly = b[i];
            for j in 0..i {
                bly -= get(&l, i, j) * x[j];
            }
            x.push(bly / get(&l, i, i));
        }

        // Backword substitution.
        for i in (0..n).rev() {
            let mut yux = x[i];
            for j in i + 1..n {
                yux -= get(&u, i, j) * x[j];
            }
            x[i] = yux;
        }

        x
    }

    fn lu_l(&self) -> HashMap<(usize, usize), f64> {
        self.l
            .iter()
            .map(|(&(y, x), &v)| ((y, x), v * self.d[x]))
            .collect()
    }

    fn lu_u(&self) -> HashMap<(usize, usize), f64> {
        self.l.iter().map(|(&(y, x), &v)| ((x, y), v)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    size: usize,
    matrix: HashMap<(usize, usize), f64>,
}

impl SparseMatrix {
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
        let mut matrix = HashMap::new();
        for (vs, offset) in diags {
            for (x, v) in vs.iter().copied().enumerate() {
                let y = x as isize - offset;
                if y >= 0 && (y as usize) < size {
                    matrix.insert((y as usize, x), v);
                } else {
                    // TODO: break if possible
                }
            }
        }
        Self { size, matrix }
    }

    pub fn from_diag(diag: Vec<f64>) -> Self {
        let size = diag.len();
        let mut matrix = HashMap::new();
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

    pub fn cholesky(self, incomplete: bool) -> Factor {
        let n = self.size;
        let mut d = vec![self.get(0, 0)];
        let mut l = HashMap::new();

        fn get(m: &HashMap<(usize, usize), f64>, y: usize, x: usize) -> f64 {
            m.get(&(y, x)).copied().unwrap_or(0.0)
        }

        l.insert((0, 0), 1.0);

        for i in 1..n {
            // i < k
            for j in 0..i {
                if incomplete && self.get(i, j).abs() < 1.0e-10 {
                    continue;
                }

                let mut lld = self.get(i, j);
                for k in 0..j {
                    lld -= get(&l, i, k) * get(&l, j, k) * d[k];
                }

                l.insert((i, j), 1.0 / d[j] * lld);
            }

            // i == k
            let mut ld = self.get(i, i);
            for k in 0..i {
                ld -= get(&l, i, k) * get(&l, i, k) * d[k];
            }
            d.push(ld);
            l.insert((i, i), 1.0);
        }

        Factor { d, l }
    }

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

#[derive(Debug)]
pub struct Image<T> {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<T>,
}

impl Image<Rgb<u8>> {
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
        let a = [[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]];
        let mut m = SparseMatrix {
            size: 3,
            matrix: Default::default(),
        };
        for y in 0..3 {
            for x in 0..3 {
                m.matrix.insert((y, x), a[y][x]);
            }
        }

        let f = m.cholesky();
        let l = f.lu_l();
        for y in 0..3 {
            for x in 0..3 {
                print!("{} ", l.get(&(y, x)).copied().unwrap_or(0.0));
            }
            println!();
        }

        println!("---------");
        println!("{:?}", f.solve(&[4.0, 13.0, -11.0]));
        // assert!(false);
    }

    #[test]
    fn ndarray_linalg() {
        use ndarray_linalg::cholesky::*;

        let a: ndarray::Array2<f64> = ndarray::array![
            [
                1003.0038505363246,
                -500.0,
                0.0,
                -500.0,
                -1.0019252681623512,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0019252681623512,
                0.0,
                0.0,
                0.0,
            ],
            [
                -500.0,
                1003.0038505363246,
                -500.0,
                0.0,
                0.0,
                -1.0019252681623512,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0019252681623512,
                0.0,
                0.0,
            ],
            [
                0.0, -500.0, 2001.0, -500.0, 0.0, 0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                -500.0, 0.0,
            ],
            [
                -500.0, 0.0, -500.0, 2001.0, 0.0, 0.0, 0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -500.0,
            ],
            [
                -1.0019252681623512,
                0.0,
                0.0,
                0.0,
                1502.0019252681623,
                -500.0,
                0.0,
                -500.0,
                -500.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                -1.0019252681623512,
                0.0,
                0.0,
                -500.0,
                1502.0019252681623,
                -500.0,
                0.0,
                0.0,
                -500.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0, 0.0, -500.0, 0.0, 0.0, -500.0, 2001.0, -500.0, 0.0, 0.0, -500.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, -500.0, -500.0, 0.0, -500.0, 2001.0, 0.0, 0.0, 0.0, -500.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, -500.0, 0.0, 0.0, 0.0, 2001.0, -500.0, 0.0, -500.0, -500.0,
                0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, 0.0, -500.0, 2001.0, -500.0, 0.0, 0.0,
                -500.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, 0.0, -500.0, 2001.0, -500.0, 0.0, 0.0,
                -500.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -500.0, -500.0, 0.0, -500.0, 2001.0, 0.0, 0.0,
                0.0, -500.0,
            ],
            [
                -1.0019252681623512,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -500.0,
                0.0,
                0.0,
                0.0,
                1502.0019252681623,
                -500.0,
                0.0,
                -500.0,
            ],
            [
                0.0,
                -1.0019252681623512,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -500.0,
                0.0,
                0.0,
                -500.0,
                1502.0019252681623,
                -500.0,
                0.0,
            ],
            [
                0.0, 0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, 0.0, -500.0,
                2001.0, -500.0,
            ],
            [
                0.0, 0.0, 0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -500.0, -500.0, 0.0,
                -500.0, 2001.0,
            ],
        ];

        let lower = a.cholesky(ndarray_linalg::cholesky::UPLO::Lower).unwrap();
        println!("{:?}", lower);
        println!();

        let b = ndarray::array![
            0.0,
            0.0,
            0.6941176470588235,
            0.9019607843137255,
            0.4980392156862745,
            0.4980392156862745,
            0.6941176470588235,
            0.9019607843137255,
            0.5333333333333333,
            0.5333333333333333,
            0.6941176470588235,
            0.9019607843137255,
            1.0,
            1.0,
            0.6941176470588235,
            0.9019607843137255
        ];
        let x = a.solvec(&b).unwrap();
        println!("{:?}", x);
        assert!(false);
    }
}
