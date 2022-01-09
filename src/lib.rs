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
        let r = im.y_len();
        let c = im.x_len();
        let k = r * c;
        let dx = wx.iter_f().map(|v| -lamb * v).collect::<Vec<_>>();
        let dy = wy.iter_f().map(|v| -lamb * v).collect::<Vec<_>>();
        let dxa = wx.iter_2x().map(|v| -lamb * v).collect::<Vec<_>>();
        let dya = wy.iter_2y().map(|v| -lamb * v).collect::<Vec<_>>();
        let dxd1 = wx.iter_3x().map(|v| -lamb * v).collect::<Vec<_>>();
        let dyd1 = wy.iter_3y().map(|v| -lamb * v).collect::<Vec<_>>();
        let dxd2 = wx.iter_4x().map(|v| -lamb * v).collect::<Vec<_>>();
        let dyd2 = wy.iter_4y().map(|v| -lamb * v).collect::<Vec<_>>();
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

#[derive(Debug)]
pub struct IlluminationMap {
    values: Vec<Vec<f64>>,
}

impl IlluminationMap {
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
                let w = 1.0 / (self.values[y][x].abs() * other.values[y][x].abs()) + sharpness;
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
                b: bytes[i + 3 + 2],
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
