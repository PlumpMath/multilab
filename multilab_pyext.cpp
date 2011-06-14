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
// python_engine implementation
//

python_engine::python_engine() 
    : engine() {
}

python_engine::python_engine(const std::string &cmd) 
    : engine(cmd) {
}

boost::python::object python_engine::get(const std::string &name) {
  ml::untyped_array<true> ml_obj(engine::get(name));

  // determine type
  int numpy_type = mx_class_to_numpy_type(ml_obj.get_ptr());
  if(numpy_type == NPY_USERDEF) {
    throw std::runtime_error(
        "currently only dense numeric matrices are supported");
  }

  return get_ndarray_(ml_obj);
}

void python_engine::put(const std::string &name, boost::python::object
    array) {
  numpy::array a = array;
  put_ndarray_(name, a);
}

void python_engine::eval(const std::string &str) {
  engine::eval(str);
}

object python_engine::get_ndarray_(ml::untyped_array<false> ml_obj) {
  // determine type
  int numpy_type = mx_class_to_numpy_type(ml_obj.get_ptr());
  if(numpy_type == NPY_USERDEF) {
    throw std::runtime_error(
        "currently only dense numeric matrices are supported");
  }

  // get dimensions
  std::vector<long> dims;
  dims.resize(ml_obj.num_dims());
  for(unsigned i=0; i<dims.size(); ++i) 
    dims[i] = ml_obj.dims()[i];

  // construct object
  object to_return;
  if(ml_obj.is_complex()) {
    // real data; rather simple implementation
    to_return = object(handle<>(PyArray_NewFromDescr(
            &PyArray_Type, 
            PyArray_DescrFromType( numpy_type ),
            dims.size(),
            &dims[0],
            NULL,
            mxGetData(ml_obj.get_ptr()),
            NPY_F_CONTIGUOUS | NPY_ENSURECOPY,
            NULL
            )));
  } else {
    // complex data: somewhat more complicated implementation
    // as numpy interleaves complex values whereas matlab stores
    // the two separately
    object real_data = object(handle<>(PyArray_NewFromDescr(
          &PyArray_Type, /* subtype */
          PyArray_DescrFromType( numpy_type ), /* descr */
          dims.size(), /* nd */
          &dims[0], /* dims */
          NULL, /* strides */
          mxGetData(ml_obj.get_ptr()), /* data */
          NPY_F_CONTIGUOUS | NPY_ENSURECOPY, /* flags */
          NULL  /* obj */
        ) ) );
    object imag_data = object(handle<>(PyArray_NewFromDescr(
          &PyArray_Type, /* subtype */
          PyArray_DescrFromType( numpy_type ), /* descr */
          dims.size(), /* nd */
          &dims[0], /* dims */
          NULL, /* strides */
          mxGetImagData(ml_obj.get_ptr()), /* data */
          NPY_F_CONTIGUOUS | NPY_ENSURECOPY, /* flags */
          NULL  /* obj */
        ) ) );
    // TODO: implement this more nicely
    // in lieu of doing the right thing and figuring out the
    // PyArray API, use Python to combine the real and 
    // imaginary parts
    object imaginary_unit =
        object(handle<>(PyComplex_FromDoubles(0,1)));
    to_return = real_data.attr("__add__")(
        imag_data.attr("__mul__")(imaginary_unit));
  }

  return to_return;
}

void python_engine::put_ndarray_(const std::string &name,
    boost::python::object array) {
}

}};

//
// define python binding
//
BOOST_PYTHON_MODULE(multilab) {
  // deep magic required for numpy
  import_array();
  numpy::array::set_module_and_type("numpy", "ndarray");

  // python_engine
  class_<ml::python_engine>("engine")
    .def(init<std::string>())
    .def("get", &ml::python_engine::get)
    .def("eval", &ml::python_engine::eval);
}

// eof //

