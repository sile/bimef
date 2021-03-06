use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(default_value = "examples/test.png")]
    image_path: PathBuf,
    #[structopt(default_value = "examples/enhanced.png")]
    output_path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    let file = std::fs::File::open(&opt.image_path)?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info()?;
    assert_eq!(reader.info().bit_depth, png::BitDepth::Eight);

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    if reader.info().color_type == png::ColorType::Rgb {
    } else if reader.info().color_type == png::ColorType::Rgba {
        let mut rgb_buf = Vec::new();
        for i in 0..(info.width * info.height) as usize {
            rgb_buf.push(buf[i * 4 + 0]);
            rgb_buf.push(buf[i * 4 + 1]);
            rgb_buf.push(buf[i * 4 + 2]);
        }
        buf = rgb_buf;
    } else {
        panic!("Unsupported {:?}", reader.info().color_type);
    }

    let image = bimef::Image::from_bytes(info.width, info.height, &buf);
    let bimef = bimef::Bimef::new();
    let image = image.to_rgb_f32();

    let start = std::time::Instant::now();
    let enhanced_image = bimef.enhance(image);
    eprintln!("Elapsed: {:?}", start.elapsed());

    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(std::fs::File::create(opt.output_path)?),
        enhanced_image.width as u32,
        enhanced_image.height as u32,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&enhanced_image.to_bytes())?;
    Ok(())
}
