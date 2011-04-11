__kernel void mandelbrot(
    __global float *input_real, 
    __global float *input_imag,
    __global float *output,
    int num_input,
    int max_iter) {
  float real = input_real[get_global_id(0)];
  float imag = input_imag[get_global_id(0)];
  float out = 0;
  for(int i=0; i<max_iter; ++i) {
    if(real*real + imag*imag > 4) {
      out = i * 1.f / max_iter;
      break;
    }
    float new_real = real*real - imag*imag;
    float new_imag = 2.f * real*imag;
    real = new_real;
    imag = new_imag;
  }
  output[get_global_id(0)] = out;
}

