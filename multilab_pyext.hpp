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

private:
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

