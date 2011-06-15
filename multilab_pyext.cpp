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

mxClassID python_untyped_array::get_type() const {
  if(arr_ == NULL) 
    throw std::runtime_error("NULL python_untyped_array");
  return arr_->get_type();
}

size_t python_untyped_array::num_dims() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return arr_->num_dims();
}

boost::python::tuple python_untyped_array::get_dims() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  const std::vector<size_t> &dims = arr_->get_dims();
  boost::python::list accum;
  for(unsigned i=0; i<dims.size(); ++i)
    accum.append(dims[i]);
  boost::python::tuple to_return(accum);
  return to_return;
}

bool python_untyped_array::is_complex() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return arr_->is_complex();
}

boost::python::object python_untyped_array::real_part() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return vec_to_ndarray_(mxGetData(arr_->get_ptr()));
}

boost::python::object python_untyped_array::imag_part() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return vec_to_ndarray_(mxGetImagData(arr_->get_ptr()));
}

boost::python::object python_untyped_array::vec_to_ndarray_(void *v) const {
  const std::vector<size_t> &dims = arr_->get_dims();
  std::vector<long> npy_dims(dims.size());
  for(unsigned i=0; i<dims.size(); ++i) npy_dims[i] = dims[i];

  int npy_type = mx_class_to_numpy_type(arr_->get_ptr());
  if(npy_type == NPY_USERDEF)
    throw std::runtime_error("attempted to get numerics from non-numeric "
        "array");

  object to_return;
  to_return = object(handle<>(PyArray_NewFromDescr(
          &PyArray_Type,
          PyArray_DescrFromType( npy_type ),
          dims.size(),
          &npy_dims[0],
          NULL,
          v,
          NPY_F_CONTIGUOUS | NPY_ENSURECOPY,
          NULL ) ) );

  return to_return;
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
      .def(init<>())
      .def("num_dims", &ml::python_untyped_array::num_dims)
      .def("get_dims", &ml::python_untyped_array::get_dims)
      .def("is_complex", &ml::python_untyped_array::is_complex)
      .def("real_part", &ml::python_untyped_array::real_part)
      .def("imag_part", &ml::python_untyped_array::imag_part);
}

// eof //

