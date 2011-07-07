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
  python_untyped_array(mxArray *a, bool managed=true);
  ~python_untyped_array();
  
  mxClassID get_type() const;
  size_t num_dims() const;
  boost::python::tuple get_dims() const;
  bool is_complex() const;

  // sparse stuff
  bool is_sparse() const;
  boost::python::object row_coords() const;
  boost::python::object col_index_counts() const;

  // numerical stuff
  boost::python::object real_part() const;
  boost::python::object imag_part() const;

  // string stuff
  boost::python::object as_string() const;

  // logical stuff
  boost::python::object as_logical() const;

  // struct stuff
  int num_fields() const;
  std::string field_name(int i) const;
  int field_number(const std::string &s) const;
  python_untyped_array get_field(int i, int field) const;

private:
  boost::python::object vec_to_ndarray_(void *v) const;
  boost::python::object indices_to_ndarray_(void *v, long size) const;

  bool managed_;
  boost::shared_ptr<ml::untyped_array<true> > arr_;
};

class python_engine : public engine {
public:
  python_engine();
  python_engine(const std::string &cmd);

  python_untyped_array get(const std::string &name);
  void put(const std::string &name,
      boost::shared_ptr<python_untyped_array> a);
  void eval(const std::string &str);
};

}}

#endif

