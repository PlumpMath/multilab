#include "multilab/multilab_matlab.hpp"
#include "cl_wrapper/cl_wrapper.hpp"
#include "kernel.opencl.hpp"

namespace ml = multilab::matlab;

void mandelbrot(int nlhs, ml::untyped_array *lhs, 
    int nrhs, ml::untyped_array *rhs) {

  if(nrhs < 1)
    throw std::runtime_error("incorrect number of arguments");
  ml::typed_array<float> input = rhs[0];
  if(!input.is_complex())
    throw std::runtime_error("input must be complex");
  if(input.num_dims() != 1)
    throw std::runtime_error("input must be a vector");

  size_t num_input = input.dims()[0];
  ml::typed_array<float> output(1, &num_input, false);

  cl::platform platform = cl::platform::platforms()[0];
  cl::device device = platform.devices()[0];
  cl::context context(platform, 1, &device);

  cl::command_queue queue(context, device);

  cl::buffer input_real(context, CL_MEM_COPY_HOST_PTR, 
      sizeof(float)*num_input, input.real_ptr());
  cl::buffer input_imag(context, CL_MEM_COPY_HOST_PTR, 
      sizeof(float)*num_input, input.imag_ptr());
  cl::buffer out_buf(context, 0, sizeof(float)*num_input, NULL);

  cl::program program(context, kernel_source);
  program.build();

  cl::kernel kernel = program.get_kernel("mandelbrot");
  kernel.set_arg(0, input_real)
    .set_arg(1, input_imag)
    .set_arg(2, out_buf)
    .set_arg(3, num_input);

  queue.run_kernel(kernel, 1, &num_input, NULL).wait();
  queue.read_buffer(out_buf, 0, sizeof(float)*num_input, 
      output.real_ptr()).wait();

  if(nlhs != 1) 
    throw std::runtime_error("expected single output value");
  lhs[0] = output;
}

MEX_ENTRY_POINT(mandelbrot);

