#include "multilab_pyext.hpp"

#include <cstring>

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
    case mxLOGICAL_CLASS: return NPY_BOOL;
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

python_untyped_array::python_untyped_array(mxArray *a, bool managed) 
    : managed_(managed) {
  arr_ = boost::shared_ptr<ml::untyped_array<true> >(
      new ml::untyped_array<true>(a) );
}

python_untyped_array::~python_untyped_array() {
  if(!managed_) {
    // shocking hackery to outsmart shared_ptr.  i've secured a special
    // spot in hell for this one.
    char tmp[sizeof(boost::shared_ptr<ml::untyped_array<true> >)];
    boost::shared_ptr<ml::untyped_array<true> > *tmp2 =
      new(tmp) boost::shared_ptr<ml::untyped_array<true> >(arr_);
  }
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
  boost::python::object o = vec_to_ndarray_(mxGetData(arr_->get_ptr()));
  return o;
}

boost::python::object python_untyped_array::imag_part() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  boost::python::object o = vec_to_ndarray_(mxGetImagData(arr_->get_ptr()));
  return o;
}

boost::python::object python_untyped_array::as_string() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  ml::typed_array<char, false> c(*arr_);
  const std::string &s = c.to_string();
  boost::python::object o(s);
  return o;
}

boost::python::object python_untyped_array::as_logical() const {
  if(arr_ == NULL) 
    throw std::runtime_error("NULL python_untyped_array");
  ml::typed_array<bool, false> b(*arr_);
  return vec_to_ndarray_(b.logical_ptr());
}

int python_untyped_array::num_fields() const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return arr_->num_fields();
}

std::string python_untyped_array::field_name(int i) const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return arr_->field_name(i);
}

int python_untyped_array::field_number(const std::string &s) const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  return arr_->field_number(s);
}

python_untyped_array python_untyped_array::get_field(int i, int field)
    const {
  if(arr_ == NULL)
    throw std::runtime_error("NULL python_untyped_array");
  mxArray *m = arr_->get_field(i, field);
  python_untyped_array to_return(m, false);
  return to_return;
}

boost::python::object python_untyped_array::vec_to_ndarray_(void *v) const {
  const std::vector<size_t> &dims = arr_->get_dims();
  std::vector<long> npy_dims(dims.size());
  size_t num_elems = 1;
  for(unsigned i=0; i<dims.size(); ++i) {
    npy_dims[i] = dims[i];
    num_elems *= dims[i];
  }

  int npy_type = mx_class_to_numpy_type(arr_->get_ptr());
  if(npy_type == NPY_USERDEF)
    throw std::runtime_error("attempted to get numerics from non-numeric "
        "array");

  handle<> h(PyArray_NewFromDescr(
      &PyArray_Type,
      PyArray_DescrFromType( npy_type ),
      dims.size(),
      &npy_dims[0],
      NULL,
      NULL, /* well, i'll be damned.  i was using this parameter wrong. */
      NPY_F_CONTIGUOUS | NPY_ENSURECOPY,
      NULL ));
  object to_return(h);
  PyArrayObject *a = reinterpret_cast<PyArrayObject*>(to_return.ptr());
  // copy data into place
  memcpy(PyArray_DATA(a), v, 
      detail::matlab_class_size(mxGetClassID(arr_->get_ptr())) * num_elems);

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

python_untyped_array
python_engine::get(const std::string &name) {
  python_untyped_array to_return(engine::get(name));
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
  class_<ml::python_untyped_array>
    ("untyped_array")
      .def("get_type", &ml::python_untyped_array::get_type)
      .def("num_dims", &ml::python_untyped_array::num_dims)
      .def("get_dims", &ml::python_untyped_array::get_dims)

      .def("is_complex", &ml::python_untyped_array::is_complex)
      .def("real_part", &ml::python_untyped_array::real_part)
      .def("imag_part", &ml::python_untyped_array::imag_part)

      .def("as_string", &ml::python_untyped_array::as_string)

      .def("as_logical", &ml::python_untyped_array::as_logical)
      
      .def("num_fields", &ml::python_untyped_array::num_fields)
      .def("field_name", &ml::python_untyped_array::field_name)
      .def("field_number", &ml::python_untyped_array::field_number)
      .def("get_field", &ml::python_untyped_array::get_field);
  enum_<mxClassID>("mx_class_id")
      .value("unknown_class", mxUNKNOWN_CLASS)
      .value("cell_class", mxCELL_CLASS)
      .value("struct_class", mxSTRUCT_CLASS)
      .value("logical_class", mxLOGICAL_CLASS)
      .value("char_class", mxCHAR_CLASS)
      .value("void_class", mxVOID_CLASS)
      .value("double_class", mxDOUBLE_CLASS)
      .value("single_class", mxSINGLE_CLASS)
      .value("int8_class", mxINT8_CLASS)
      .value("uint8_class", mxUINT8_CLASS)
      .value("int16_class", mxINT16_CLASS)
      .value("uint16_class", mxUINT16_CLASS)
      .value("int32_class", mxINT32_CLASS)
      .value("uint32_class", mxUINT32_CLASS)
      .value("int64_class", mxINT64_CLASS)
      .value("uint64_class", mxUINT64_CLASS)
      .value("function_class", mxFUNCTION_CLASS);
}

// eof //

