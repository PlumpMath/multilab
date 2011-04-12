#include "multilab/multilab_matlab.hpp"
#include "cl_wrapper/cl_wrapper.hpp"
#include "kernel.opencl.hpp"

#include <algorithm>

namespace ml = multilab::matlab;

void mandelbrot(int nlhs, ml::untyped_array *lhs, 
    int nrhs, ml::untyped_array *rhs) {

  if(nrhs < 1)
    throw std::runtime_error("incorrect number of arguments");
  ml::typed_array<float> input = rhs[0];
  if(!input.is_complex())
    throw std::runtime_error("input must be complex");

  size_t w = input.dims()[0];
  size_t h = input.dims()[1];

  size_t num_input = std::max(w,h);
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

  cl_int param_num_input = num_input;
  cl_int num_iter = 200;
  cl::kernel kernel = program.get_kernel("mandelbrot");
  kernel.set_arg(0, input_real)
    .set_arg(1, input_imag)
    .set_arg(2, out_buf)
    .set_arg(3, param_num_input)
    .set_arg(4, num_iter);

  size_t block_size = 256;
  queue.run_kernel(kernel, 1, &num_input, &block_size).wait();
  queue.read_buffer(out_buf, 0, sizeof(float)*num_input, 
      output.real_ptr()).wait();

  if(nlhs != 1) 
    throw std::runtime_error("expected single output value");
  lhs[0] = output;
}

MEX_ENTRY_POINT(mandelbrot);

