#include "multilab_pyext.hpp"

using namespace boost::python;
namespace ml = ::multilab::matlab;
namespace numpy = boost::python::numeric;

namespace multilab {
namespace matlab {

//
// helpers
//
static int mx_class_to_numpy_type(mxArray *array) {
  mxClassID class_id = mxGetClassID(array);
  switch(class_id) {
    case mxDOUBLE_CLASS: return NPY_DOUBLE;
    case mxSINGLE_CLASS: return NPY_FLOAT;
    case mxINT8_CLASS: return NPY_BYTE;
    case mxUINT8_CLASS: return NPY_UBYTE;
    case mxINT16_CLASS: return NPY_SHORT;
    case mxUINT16_CLASS: return NPY_USHORT;
    case mxINT32_CLASS: return NPY_INT;
    case mxUINT32_CLASS: return NPY_UINT;
    case mxINT64_CLASS: return NPY_ULONGLONG;
    case mxUINT64_CLASS: return NPY_LONGLONG;
    default: return NPY_USERDEF;
  }
}

#if 0
static mxClassID numpy_type_to_mx_class(numpy::array array, 
    bool &is_complex) {
}
#endif

//
// python_untyped_array implementation
//

python_untyped_array::python_untyped_array() {
}

python_untyped_array::python_untyped_array(mxArray *a) {
  arr_ = boost::shared_ptr<ml::untyped_array<true> >(
      new ml::untyped_array<true>(a) );
}

python_untyped_array::~python_untyped_array() {
}

//
// python_engine implementation
//

python_engine::python_engine() 
    : engine() {
}

python_engine::python_engine(const std::string &cmd) 
    : engine(cmd) {
}

boost::shared_ptr<python_untyped_array> 
python_engine::get(const std::string &name) {
  boost::shared_ptr<python_untyped_array> to_return(new
      python_untyped_array(
        engine::get(name)
        ));
  return to_return;
}

void python_engine::put(const std::string &name, 
    boost::shared_ptr<python_untyped_array> array) {
}

void python_engine::eval(const std::string &str) {
  engine::eval(str);
}

}};

//
// define python binding
//
BOOST_PYTHON_MODULE(multilab_private) {
  // deep magic required for numpy
  import_array();
  numpy::array::set_module_and_type("numpy", "ndarray");

  // python_engine
  class_<ml::python_engine>("engine")
    .def(init<std::string>())
    .def("get", &ml::python_engine::get)
    .def("eval", &ml::python_engine::eval);
  class_<ml::python_untyped_array,
      boost::shared_ptr<ml::python_untyped_array> >
    ("untyped_array")
      .def(init<>());
}

// eof //

