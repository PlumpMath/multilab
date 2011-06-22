#include <multilab/multilab_matlab.hpp>
#include <iostream>

namespace ml = multilab::matlab;

int main(int argc, char *argv[]) {
  ml::engine engine;
  engine.eval("z = eye(1024);");
  ml::typed_array<double, true> z(engine.get("z"));
  ml::typed_array<double, false> okay = z;

  // attempt to create and get a function handle
  // n.b. for whatever reason, the MATLAB gods frown on this
  // (read: this is why you should be trying to use python instead of
  // matlab)
  //engine.eval("f = @(x) x^2");
  //ml::untyped_array<false> f(engine.get("f"));
}

