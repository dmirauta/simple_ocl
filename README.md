# Simple OpenCL

Provides further wrapping around [ocl](https://docs.rs/ocl/latest/ocl/) to simplify program creation and buffer management.

Since there is often a (fixed) cpu counterpart to a gpu buffer handle, used to copy data from/to, `PairedBuffers` keeps track of which cpu buffer needs to be copied to which gpu buffer and vice versa.
This can be created from an [ndarray](https://docs.rs/ndarray/latest/ndarray/).

A struct with `PairedBuffers` type fields can `#[derive(DeviceToFrom)]`.
Fields can be marked with `#[dev_to_from(from = false)]` to only be sent to the device (kernel inputs), or `#[dev_to_from(to = false)]` to only be copied from the device (kernel outputs).
Buffers that are in/out need not be annotated as the default is both 'to' and 'from'.
A single call to either `self.send_pairedbuffs()` or `self.retrieve_pairedbuffs()` then synchronises host/device (usually cpu/gpu) data accordingly.
See [example](./simple_ocl_example/src/main.rs).
