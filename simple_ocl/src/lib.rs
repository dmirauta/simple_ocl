use std::{error::Error, fs, path::Path};

use ndarray::{Array, Dimension, Ix1, Ix2, Ix3};
use ocl::{builders::ProgramBuilder, Buffer, OclPrm, ProQue};

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

/// try to make OCL program from internal source string
pub fn try_prog_que_from_source(
    source: impl Into<String>,
    name: impl Into<String>,
    compiler_opts: Vec<String>,
) -> ocl::Result<ProQue> {
    let src = source.into();
    let mut prog_build = ProgramBuilder::new();
    prog_build.src(src);
    for copt in compiler_opts {
        prog_build.cmplr_opt(copt);
    }
    let copts = prog_build.get_compiler_options()?;

    let que = ProQue::builder().prog_bldr(prog_build).build()?;

    println!("{} compiled with options: {:?}\n", name.into(), copts);

    Ok(que)
}

/// make OCL program from internal source string
pub fn prog_que_from_source(
    source: impl Into<String>,
    name: impl Into<String>,
    compiler_opts: Vec<String>,
) -> ProQue {
    try_prog_que_from_source(source, name, compiler_opts).unwrap()
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
