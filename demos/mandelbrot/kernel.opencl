__kernel void mandelbrot(
    __global float *input_real, 
    __global float *input_imag,
    __global float *output,
    int num_input,
    int max_iter) {
  float oreal = input_real[get_global_id(0)];
  float oimag = input_imag[get_global_id(0)];
  float real = oreal;
  float imag = oimag;
  float out = 1.f;
  for(int i=0; i<max_iter; ++i) {
    if(real*real + imag*imag > 2) {
      out = i * 1.f / max_iter;
      break;
    }
    float new_real = real*real - imag*imag;
    float new_imag = 2.f * real*imag;
    real = new_real + oreal;
    imag = new_imag + oimag;
  }
  output[get_global_id(0)] = out;
}

