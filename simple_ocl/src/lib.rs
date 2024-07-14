use std::{cell::RefCell, error::Error, fs, path::Path};

use ndarray::{Array, Dimension, Ix1, Ix2, Ix3};
use ocl::{builders::ProgramBuilder, core::DeviceInfo, Buffer, Device, OclPrm, Platform, ProQue};

pub use simple_ocl_proc::*;

/// Struct for keeping local buffer and corresponding remote buffer handles together
pub struct PairedBuffers<N: OclPrm, D> {
    /// local buffer to copy from/to
    pub host: Array<N, D>,
    /// handle to memory on OCL device (usually GPU?)
    pub device: Buffer<N>,
}

pub type PairedBuffers1<N> = PairedBuffers<N, Ix1>;
pub type PairedBuffers2<N> = PairedBuffers<N, Ix2>;
pub type PairedBuffers3<N> = PairedBuffers<N, Ix3>;

impl<N: OclPrm, D: Dimension> PairedBuffers<N, D> {
    pub fn create_from(host: Array<N, D>, que: &mut ProQue) -> Self {
        que.set_dims(host.len());
        let device = que.create_buffer::<N>().expect("buffer create error");
        Self { host, device }
    }

    pub fn to_device(&self) -> ocl::Result<()> {
        self.device.write(self.host.as_slice().unwrap()).enq()
    }

    pub fn from_device(&mut self) -> ocl::Result<()> {
        self.device.read(self.host.as_slice_mut().unwrap()).enq()
    }
}

thread_local! {
    static PLATFORM_DEVICE: RefCell<Option<(Platform, Device)>> = const { RefCell::new(None) };
}

pub fn set_ocl_device(platform_idx: usize, dev_idx: usize) {
    PLATFORM_DEVICE.with_borrow_mut(|rc| {
        let plats = Platform::list();
        if plats.is_empty() {
            println!("No OpenCL platforms found.");
            return;
        }
        if platform_idx >= plats.len() {
            println!(
                "Platform idx {platform_idx} is invalid on machine with {} drivers.",
                plats.len()
            );
            return;
        }
        let plat = plats[platform_idx];
        let plat_name = plat
            .name()
            .unwrap_or("<Could not get platform name>".into());
        println!("Setting platform idx {platform_idx} ({plat_name}).");

        if let Ok(devs) = Device::list_all(plat) {
            if dev_idx < devs.len() {
                let dev = devs[dev_idx];
                let dev_name = dev.name().unwrap_or("<Could not get device name>".into());
                *rc = Some((plat, dev));
                println!("Setting device idx {dev_idx} ({dev_name}) from {plat_name}.");
            } else {
                println!(
                    "Device idx {dev_idx} invalid for platform with {} devices.",
                    devs.len()
                );
            }
        } else {
            println!("Could not access devices for {plat_name}.",);
        }
    });
}

/// try to make OCL program from internal source string
pub fn try_prog_que_from_source(
    source: impl Into<String>,
    name: impl Into<String>,
    compiler_opts: Vec<String>,
) -> ocl::Result<ProQue> {
    let src = source.into();
    let prog_name = name.into();

    let mut prog_build = ProgramBuilder::new();
    prog_build.src(src);
    for copt in compiler_opts {
        prog_build.cmplr_opt(copt);
    }
    let copts = prog_build.get_compiler_options()?;

    let mut pqb = ProQue::builder();
    pqb.prog_bldr(prog_build);

    if let Some((plat, device)) = PLATFORM_DEVICE.with_borrow(|rc| *rc) {
        let dev_name = device
            .name()
            .unwrap_or("<Could not get device name>".into());
        println!("Using {dev_name} in {prog_name}");
        pqb.platform(plat);
        pqb.device(device);
    }

    let que = pqb.build()?;

    println!("{prog_name} compiled with options: {:?}\n", copts);

    Ok(que)
}

/// make OCL program from internal source string
pub fn prog_que_from_source(
    source: impl Into<String>,
    name: impl Into<String>,
    compiler_opts: Vec<String>,
) -> ProQue {
    try_prog_que_from_source(source, name, compiler_opts)
        .map_err(|e| {
            if let ocl::Error::OclCore(ocl::core::Error::ProgramBuild(pbe)) = e {
                eprintln!("{pbe}");
            }
        })
        .unwrap()
}

/// try to make OCL program loading source from path
pub fn try_prog_que_from_source_path(
    source_path: impl AsRef<Path>,
    compiler_opts: Vec<String>,
) -> Result<ProQue, Box<dyn Error>> {
    let src = fs::read_to_string(source_path.as_ref())?;
    Ok(try_prog_que_from_source(
        src,
        source_path.as_ref().to_str().unwrap(),
        compiler_opts,
    )?)
}

/// make OCL program loading source from path
pub fn prog_que_from_source_path(
    source_path: impl AsRef<Path>,
    compiler_opts: Vec<String>,
) -> ProQue {
    let src = fs::read_to_string(source_path.as_ref()).expect("could not load barycenter kernel");
    prog_que_from_source(src, source_path.as_ref().to_str().unwrap(), compiler_opts)
}

pub trait DeviceToFrom {
    fn send_pairedbuffs(&self) -> ocl::Result<()>;
    fn retrieve_pairedbuffs(&mut self) -> ocl::Result<()>;
}

pub fn print_ocl_short_info() {
    // TODO: Print selected DEVICE (thread_local)

    for (j, plat) in ocl::Platform::list().iter().enumerate() {
        let devs = ocl::Device::list_all(plat).unwrap();
        let spacing = "   ";

        println!("\nPlatform (driver) {j}: {}", plat.name().unwrap());

        for (i, dev) in devs.iter().enumerate() {
            println!("\n{spacing}Device {i}",);

            let spacing = spacing.repeat(2);
            match dev.name() {
                Ok(name) => println!("{spacing}Name: {name}"),
                Err(_) => println!("{spacing}Could not get device name."),
            }
            match dev.info(DeviceInfo::Extensions) {
                Ok(ext) => println!(
                    "{spacing}Supports doubles (f64): {}",
                    ext.to_string().contains("cl_khr_fp64")
                ),
                Err(_) => println!("{spacing}Could not get device extensions."),
            };
        }
    }
    println!();
}

pub fn selected_device_supports_doubles() -> Option<bool> {
    let (_, device) = PLATFORM_DEVICE.with_borrow(|rc| *rc)?;
    let ext = device
        .info(DeviceInfo::Extensions)
        .map_err(|_| println!("Could not get device extensions."))
        .ok()?;
    Some(ext.to_string().contains("cl_khr_fp64"))
}
