#include "multilab/multilab_mxarray.hpp"

namespace ml = multilab::matlab;

void new_array_test(int nargin, ml::Array<ml::Untyped> *argin,
    int nargout, ml::Array<ml::Untyped> *argout) {
  if(nargin != 2 || nargout != 2) {
    throw std::runtime_error("durr");
  }
  ml::Array<ml::Untyped> untyped = argin[0];
  ml::Array<double> typed = untyped;
  double val = typed.real(0);

  ml::Array<ml::Struct> command = argin[1];
  ml::Array<double> times = command.get_field("times");
  ml::Array<double> add = command.get_field("add");

  val = val*times.real(0) + add.real(0);

  Eigen::DynamicSparseMatrix<double> dd(5,5);
  for(int i=0; i<5; ++i) {
    dd.coeffRef(i,i) = val;
  }
  Eigen::SparseMatrix<double> d = dd;

  ml::Array<ml::Sparse> out = d;
  argout[0] = out;

  ml::Array<double> out2(1,1);
  out2.real(0) = 42;
  argout[1] = out2;
}

MEX_ENTRY_POINT2(new_array_test);

