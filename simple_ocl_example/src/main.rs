use ndarray::{Array1, Array2};
use ocl::{Event, ProQue, Result};
use simple_ocl::{
    print_ocl_short_info, prog_que_from_source, DeviceToFrom, PairedBuffers, PairedBuffers1,
    PairedBuffers2,
};

#[derive(DeviceToFrom)]
struct ExampleProg {
    que: ProQue,
    status: Event,
    shape: (usize, usize),
    #[dev_to_from(from = false)]
    a: PairedBuffers1<i32>,
    b: PairedBuffers1<f32>,
    #[dev_to_from(to = false)]
    c: PairedBuffers2<f64>,
}

static MY_CL_PROG: &str = "
__kernel void test_kernel(__global int    *a,
                          __global float  *b,
                          __global double *c,
                                   float   d)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width = get_global_size(1);
    int k = i*width + j;

    c[k] = a[i] + b[j]; 

#ifdef MUTATE_B
    if(i==0) {
        b[j] += d;
    } 
#endif
}

// could have additional kernels...
";

impl ExampleProg {
    fn new(shape: (usize, usize)) -> Self {
        let (height, width) = shape;

        let mut que =
            prog_que_from_source(MY_CL_PROG, "my_cl_prog", vec!["-DMUTATE_B".to_string()]);

        let mut a_c = Array1::<i32>::zeros(height);
        for (i, v) in a_c.indexed_iter_mut() {
            *v = i as i32;
        }

        let b_c = Array1::<f32>::range(0.0, (width as f32) * 0.25, 0.25);
        let c_c = Array2::<f64>::zeros(shape);

        let a = PairedBuffers::create_from(a_c, &mut que);
        let b = PairedBuffers::create_from(b_c, &mut que);
        let c = PairedBuffers::create_from(c_c, &mut que);

        let status = Event::empty();

        Self {
            que,
            status,
            shape,
            a,
            b,
            c,
        }
    }

    fn run(&mut self) -> Result<()> {
        self.send_pairedbuffs()?;

        self.que.set_dims(self.shape);
        let kernel = self
            .que
            .kernel_builder("test_kernel")
            .arg(&self.a.device)
            .arg(&self.b.device)
            .arg(&self.c.device)
            .arg(0.05f32)
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

    let mut example = ExampleProg::new((5, 6));
    example.run()?;

    //// Non-blocking related
    let mut i = 0;
    while !example.status.is_complete()? && i < 10000 {
        print!("Do some stuff in the meantime {i}\r");
        i += 1;
    }

    example.retrieve_pairedbuffs()?;
    ////

    println!("\n\nEnd results:");
    dbg!(&example.a.host);
    dbg!(&example.b.host);
    dbg!(&example.c.host);

    Ok(())
}
