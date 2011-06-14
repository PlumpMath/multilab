#include <multilab/multilab_matlab.hpp>
#include <iostream>

namespace ml = multilab::matlab;

int main(int argc, char *argv[]) {
  ml::engine engine;
  engine.eval("z = eye(1024);");
  ml::typed_array<double, true> z(engine.get("z"));
  ml::typed_array<double, false> okay = z;
}

