use ndarray::Array2;
use ocl::{Event, ProQue, Result};
use simple_ocl::{
    print_ocl_short_info, prog_que_from_source, DeviceToFrom, PairedBuffers, PairedBuffers2,
};

#[derive(DeviceToFrom)]
struct ExampleProg {
    que: ProQue,
    status: Event,
    #[dev_to_from(from = false)]
    a: PairedBuffers2<i32>,
    b: PairedBuffers2<f32>,
    #[dev_to_from(to = false)]
    c: PairedBuffers2<f64>,
}

static MY_CL_PROG: &str = "
__kernel void test_kernel(__global int    *a,
                          __global float  *b,
                          __global double *c)
{
    int k = get_global_id(0);
    c[k] = a[k] + b[k]; 
#ifdef CHANGE_B
    b[k] += 0.25;
#endif
}

// could have additional kernels...
";

impl ExampleProg {
    fn new(shape: (usize, usize)) -> Self {
        let mut que =
            prog_que_from_source(MY_CL_PROG, "my_cl_prog", vec!["-DCHANGE_B".to_string()]);

        let (_, width) = shape;
        let mut a_c = Array2::<i32>::zeros(shape);
        for ((i, j), v) in a_c.indexed_iter_mut() {
            *v = (i * width + j) as i32;
        }

        let mut b_c = Array2::<f32>::zeros(shape);
        b_c.fill(0.5);

        let c_c = Array2::<f64>::zeros(shape);

        let a = PairedBuffers::create_from(a_c, &mut que);
        let b = PairedBuffers::create_from(b_c, &mut que);
        let c = PairedBuffers::create_from(c_c, &mut que);

        let status = Event::empty();

        Self {
            que,
            status,
            a,
            b,
            c,
        }
    }

    fn run(&mut self) -> Result<()> {
        self.send_pairedbuffs()?;

        let kernel = self
            .que
            .kernel_builder("test_kernel")
            .arg(&self.a.device)
            .arg(&self.b.device)
            .arg(&self.c.device)
            .build()?;

        unsafe {
            kernel.cmd().enew(&mut self.status).enq()?;
        }

        // if this is to be blocking, then can just sync here
        // self.retrieve_pairedbuffs()?;

        Ok(())
    }
}

fn main() -> Result<()> {
    print_ocl_short_info();
    // set_ocl_device(1, 0);

    let mut example = ExampleProg::new((5, 5));
    example.run()?;

    //// Non-blocking related
    let mut i = 0;
    while !example.status.is_complete()? && i < 10000 {
        print!("Do some stuff in the meantime {i}\r");
        i += 1;
    }
    println!();

    example.retrieve_pairedbuffs()?;
    ////

    dbg!(&example.a.host);
    dbg!(&example.b.host);
    dbg!(&example.c.host);

    Ok(())
}
