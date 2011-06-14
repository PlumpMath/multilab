#ifndef _MULTILAB_PYEXT_HPP_
#define _MULTILAB_PYEXT_HPP_

#include "multilab_matlab.hpp"
#include "multilab_numpy.hpp"

#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>

#include <numpy/ndarrayobject.h>

#include <string>
#include <stdexcept>

namespace multilab {
namespace matlab {

namespace numpy = boost::python::numeric;
namespace ml = ::multilab::matlab;

class python_engine : public engine {
public:
  python_engine();
  python_engine(const std::string &cmd);

  boost::python::object get(const std::string &name);
  void put(const std::string &name, boost::python::object array);
  void eval(const std::string &str);

private:
  boost::python::object get_ndarray_(ml::untyped_array<false> arr);

  void put_ndarray_(const std::string &name, numpy::array array);
};

}}

#endif

