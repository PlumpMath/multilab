#ifndef _MULTILAB_PYEXT_HPP_
#define _MULTILAB_PYEXT_HPP_

#include "multilab_matlab.hpp"
#include "multilab_numpy.hpp"

#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/shared_ptr.hpp>

#include <numpy/ndarrayobject.h>

#include <string>
#include <stdexcept>

namespace multilab {
namespace matlab {

namespace numpy = boost::python::numeric;
namespace ml = ::multilab::matlab;

class python_untyped_array {
public:
  python_untyped_array();
  python_untyped_array(mxArray *a);
  ~python_untyped_array();
  
  mxClassID get_type() const;
  size_t num_dims() const;
  boost::python::tuple get_dims() const;
  bool is_complex() const;

  boost::python::object real_part() const;
  boost::python::object imag_part() const;

private:
  boost::python::object vec_to_ndarray_(void *v) const;

  boost::shared_ptr<ml::untyped_array<true> > arr_;
};

class python_engine : public engine {
public:
  python_engine();
  python_engine(const std::string &cmd);

  boost::shared_ptr<python_untyped_array> get(const std::string &name);
  void put(const std::string &name,
      boost::shared_ptr<python_untyped_array> a);
  void eval(const std::string &str);
};

}}

#endif

