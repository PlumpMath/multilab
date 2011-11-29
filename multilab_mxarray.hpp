#ifndef _MULTILAB_MATLAB2_HPP_
#define _MULTILAB_MATLAB2_HPP_

#include <complex>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <boost/noncopyable.hpp>

#include <alloca.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// MATLAB includes
// MATLAB's stupid, tell it we've got char16_t
// n.b., only seemed to cause issues w/ unsupported c++11 builds
// #define __STDC_UTF_16__

#include <engine.h>
#include <matrix.h>
#include <mex.h>

#ifndef MULTILAB_NO_SPARSE 
// include magic for sparse matrices
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Sparse>
#endif

namespace multilab {
namespace matlab {

struct Untyped { };
struct Sparse { };
struct Struct { };
struct Cell { };

// \brief contains nuts and bolts
// {{{ detail
namespace detail {
  template<mxClassID CLASS> struct matlab2cpp { };
  #define MATLAB2CPP(CLASSID, TYPE, NAME) \
  template<> struct matlab2cpp<CLASSID> { \
    typedef TYPE cpp_type; \
  };

  template<typename T> struct cpp2matlab { };
  template<typename T> struct cpp_type_name_fctor { };
  template<typename T> static const char* cpp_type_name() { 
    cpp_type_name_fctor<T> f;
    return f();
  }
  #define CPP2MATLAB(TYPE, CLASSID, NAME) \
  template<> struct cpp2matlab<TYPE> { \
    static const mxClassID matlab_class = CLASSID; \
  }; \
  template<> struct cpp_type_name_fctor<TYPE> { \
    const char* operator()() const { \
      return NAME; \
    } \
  };

  #define TIE_MATLAB_CPP(CLASSID, TYPE, NAME) \
  MATLAB2CPP(CLASSID, TYPE, NAME); \
  CPP2MATLAB(TYPE, CLASSID, NAME);
  // TODO: implement/figure out how to implement these:
  // TIE_MATLAB_CPP(mxUNKNOWN_CLASS, ?)
  // TIE_MATLAB_CPP(mxFUNCTION_CLASS, ?)
  TIE_MATLAB_CPP(mxSPARSE_CLASS, Sparse, "sparse");
  TIE_MATLAB_CPP(mxCELL_CLASS, Cell, "cell");
  TIE_MATLAB_CPP(mxSTRUCT_CLASS, Struct, "struct");
  TIE_MATLAB_CPP(mxLOGICAL_CLASS, bool, "bool");
  TIE_MATLAB_CPP(mxCHAR_CLASS, char, "char");
  TIE_MATLAB_CPP(mxDOUBLE_CLASS, double, "double");
  TIE_MATLAB_CPP(mxSINGLE_CLASS, float, "float");
  TIE_MATLAB_CPP(mxINT8_CLASS, int8_t, "int8");
  TIE_MATLAB_CPP(mxUINT8_CLASS, uint8_t, "uint8");
  TIE_MATLAB_CPP(mxINT16_CLASS, int16_t, "int16");
  TIE_MATLAB_CPP(mxUINT16_CLASS, uint16_t, "uint16");
  TIE_MATLAB_CPP(mxINT32_CLASS, int32_t, "int32");
  TIE_MATLAB_CPP(mxUINT32_CLASS, uint32_t, "uint32");
  TIE_MATLAB_CPP(mxINT64_CLASS, int64_t, "int64");
  TIE_MATLAB_CPP(mxUINT64_CLASS, uint64_t, "uint64");
  
  #undef TIE_MATLAB_CPP
  #undef CPP2MATLAB
  #undef MATLAB2CPP

  static const char* matlab_type_name(mxClassID id) {
    switch(id) {
      case mxCHAR_CLASS: return "char";
      case mxDOUBLE_CLASS: return "double";
      case mxSINGLE_CLASS: return "float";
      case mxINT8_CLASS: return "int8";
      case mxUINT8_CLASS: return "uint8";
      case mxINT16_CLASS: return "int16";
      case mxUINT16_CLASS: return "uint16";
      case mxINT32_CLASS: return "int32";
      case mxUINT32_CLASS: return "uint32";
      case mxINT64_CLASS: return "int64";
      case mxUINT64_CLASS: return "uint64";
      case mxLOGICAL_CLASS: return "bool";
      case mxCELL_CLASS: return "cell";
      case mxSTRUCT_CLASS: return "struct";
      case mxSPARSE_CLASS: return "sparse";
      default: throw std::runtime_error("unhandled MATLAB type");
    }
  }
} // }}}

// {{{ ArrayBase 
template<typename Derived>
class ArrayBase {
public:
  mxClassID get_type() const {
    return mxGetClassID(this->ptr());
  }
  template<typename T>
  bool is_of_type() const {
    return get_type() == detail::cpp2matlab<T>::matlab_class;
  }
  bool is_of_type(mxClassID c) const {
    return get_type() == c;
  }
  size_t num_dims() const {
    return mxGetNumberDimensions(ptr());
  }
  const mwSize* dims() const {
    return mxGetDimensions(ptr());
  }
  bool is_complex() const {
    return mxIsComplex(ptr());
  }
  bool is_sparse() const {
    return mxIsSparse(ptr());
  }
  mxArray* ptr() const {
    return ptr_;
  }

protected:
  ArrayBase(mxArray *ptr) : ptr_(ptr) { }

  void type_check() const {
    if(!is_of_type<typename Derived::value_t>()) {
      std::stringstream ss;
      ss << "expected type " << 
        detail::matlab_type_name(get_type()) <<
        " but got " <<
        detail::matlab_type_name(
            detail::cpp2matlab<typename
            Derived::value_t>::matlab_class);
      throw std::runtime_error(ss.str());
    }
  }

  mxArray *ptr_;
}; // }}}

template<typename T> class Array;

// {{{ dense numeric matrix 
template<typename Scalar>
class Array : public ArrayBase<Array<Scalar>> {
public:
  typedef Scalar value_t;

  Array()
      : ArrayBase<Array<Scalar>>(nullptr) { }
  Array(mxArray *ptr) 
      : ArrayBase<Array<Scalar>>(ptr) { 
    this->type_check();
  }
  template<typename O>
  Array(const Array<O> &o)
      : ArrayBase<Array<Scalar>>(o.ptr()) { 
    this->type_check();
  }
  Array(size_t rows, size_t cols, bool is_complex = false)
        : ArrayBase<Array<Scalar>>(mxCreateNumericMatrix(rows,
              cols, detail::cpp2matlab<Scalar>::matlab_class,
              is_complex ? mxCOMPLEX : mxREAL)) {
    if(!this->ptr()) {
      throw std::runtime_error("error allocating MATLAB array");
    }
  }
  Array(size_t ndim, size_t *dims, bool is_complex = false)
        : ArrayBase<Array<Scalar>>(nullptr) {
    mwSize ml_dims[ndim];
    std::copy(dims, dims+ndim, ml_dims);
    mxArray *t = mxCreateNumericArray(ndim, ml_dims,
        detail::cpp2matlab<Scalar>::matlab_class,
        is_complex ? mxCOMPLEX : mxREAL);
    if(!t) {
      throw std::runtime_error("error allocating MATLAB array");
    }
    this->ptr_ = t;
  }
  template<typename M>
  Array(const M &m)
        : ArrayBase<Array<Scalar>>(mxCreateNumericMatrix(
            m.rows(), m.cols(),
            detail::cpp2matlab<Scalar>::matlab_class,
            mxREAL)) {
    if(!this->ptr_) {
      throw std::runtime_error("error allocating MATLAB array");
    }
    std::copy(m.data(), m.data()+m.rows()*m.cols(), 
        real_ptr());
  }
  ~Array() { }

  Scalar* real_ptr() const {
    return reinterpret_cast<Scalar*>(
        mxGetData( this->ptr() ));
  }
  Scalar* imag_ptr() const {
    return reinterpret_cast<Scalar*>(
        mxGetImagData( this->ptr() ));
  }

  Scalar& real(int i) const {
    return real_ptr()[i];
  }
  Scalar& imag(int i) const {
    return imag_ptr()[i];
  }

  template<typename O>
  Array& operator=(const Array<O> &o) {
    this->ptr_ = o.ptr();
    this->type_check();
    return *this;
  }
};
// }}}

// {{{ untyped Array 
template<>
class Array<Untyped> : public ArrayBase<Array<Untyped>> {
public:
  typedef Untyped value_t;

  Array() 
      : ArrayBase<Array<Untyped>>(nullptr) { }
  Array(mxArray *ptr) 
      : ArrayBase<Array<Untyped>>(ptr) { }
  template<typename T>
  Array(const Array<T> &o) 
      : ArrayBase<Array<Untyped>>(o.ptr()) { }
  ~Array() { }

  template<typename T>
  Array<Untyped>& operator=(const ArrayBase<T> &t) {
    this->ptr_ = t.ptr();
    return *this;
  }

}; // }}}

// {{{ struct Array
template<>
class Array<Struct> : public ArrayBase<Array<Struct>> {
public:
  typedef Struct value_t;

  Array()
      : ArrayBase<Array<Struct>>(nullptr) { }
  Array(mxArray *ptr)
      : ArrayBase<Array<Struct>>(ptr) { 
    this->type_check();
  }
  template<typename T>
  Array(const Array<T> &o)
      : ArrayBase<Array<Struct>>(o.ptr()) {
    this->type_check();
  }
  ~Array() { }

  template<typename T>
  Array<Struct>& operator=(const ArrayBase<T> &t) {
    this->ptr_ = t.ptr();
    this->type_check();
    return *this;
  }

  int num_fields() const {
    return mxGetNumberOfFields(this->ptr_);
  }

  std::string field_name(int i) const {
    const char *c = mxGetFieldNameByNumber(this->ptr_, i);
    if(!c) {
      throw std::runtime_error("couldn't get field name");
    }
    return std::string(c);
  }

  int field_number(const std::string &name) const {
    int ret = mxGetFieldNumber(this->ptr_, name.c_str());
    if(ret == -1) {
      throw std::runtime_error("attempted to get nonexistant field");
    }
    return ret;
  }

  Array<Untyped> get_field(int field, int i=0) const {
    mxArray *m = mxGetFieldByNumber(this->ptr_, i, field);
    if(!m) {
      throw std::runtime_error("error getting field value");
    }
    return m;
  }

  Array<Untyped> get_field(const std::string &name, int i=0) {
    return get_field(field_number(name), i);
  }
<<<<<<< HEAD
}; // }}}

// {{{ char mxArray (string)
=======
};
// }}}

// {{{ char Array
template<>
class Array<char> : public ArrayBase<Array<char>> {
public:
  typedef char value_t;

  Array()
      : ArrayBase<Array<char>>(nullptr) { }
  Array(mxArray *ptr)
      : ArrayBase<Array<char>>(ptr) {
    this->type_check();
  }
  template<typename T>
  Array(const Array<T> &o)
      : ArrayBase<Array<char>>(o.ptr()) {
    this->type_check();
  }

  template<typename T>
  Array<char>& operator=(const ArrayBase<T> &t) {
    this->ptr_ = t.ptr();
    this->type_check();
    return *this;
  }

  std::string str() const {
    char *c = mxArrayToString(this->ptr_);
    if(!c) {
      throw std::runtime_error("error converting mxArray into string");
    }
    std::string to_return(c);
    mxFree(c);
    return to_return;
  }

  operator std::string() const {
    return str();
  }
  bool operator==(const std::string &s) {
    return str() == s;
  }
  bool operator==(const char *c) {
    return *this == std::string(c);
  }
}; // }}}

#ifndef MULTILAB_NO_SPARSE
// {{{ Sparse Array 
template<>
class Array<Sparse> : public ArrayBase<Array<Sparse>> {
public:
  typedef Sparse value_t;

  Array() 
      : ArrayBase<Array<Sparse>>(nullptr) { }
  Array(mxArray *ptr)
      : ArrayBase<Array<Sparse>>(ptr) {
    this->type_check();
  }
  template<typename T>
  Array(const Array<T> &o)
      : ArrayBase<Array<Sparse>>(o.ptr()) {
    this->type_check();
  }
  template<typename Scalar>
  Array(const Eigen::SparseMatrix<Scalar> &m) 
      : ArrayBase<Array<Sparse>>(nullptr) {
    *this = m;
  }
  // TODO complex sparse arrays
  ~Array() { }

  template<typename T>
  Array<Sparse>& operator=(const ArrayBase<T> &t) {
    this->ptr_ = t.ptr();
    this->type_check();
    return *this;
  }

  template<typename Scalar>
  Array<Sparse>& operator=(const Eigen::SparseMatrix<Scalar> &sp) {
    int r = sp.rows();
    int c = sp.cols();
    int nz = sp.nonZeros();

    mxArray *m = mxCreateSparse(r, c, nz, mxREAL);
    if(!m) {
      throw std::runtime_error("error creating sparse matlab matrix");
    }

    mwSize *ir, *jc;
    double *data;
    ir = mxGetIr(m);
    jc = mxGetJc(m);
    data = reinterpret_cast<double*>(mxGetData(m));
    int idatum = 0;

    for(int k=0; k<sp.cols(); ++k) {
      bool first = true;
      for(typename Eigen::SparseMatrix<Scalar>::InnerIterator it(sp, k);
          it; ++it) {
        if(first) {
          jc[k] = idatum;
          first = false;
        }
        ir[idatum] = it.row();
        data[idatum] = it.value();
        ++idatum;
      }
    }
    jc[sp.cols()] = nz;

    this->ptr_ = m;
    return *this;
  }
}; // }}}
#endif
};

}
}

#ifndef MEX_ENTRY_POINT2
#define MEX_ENTRY_POINT2(funct) \
extern "C" { \
  void mexFunction(int nlhs, mxArray *lhs[], int nrhs, \
      const mxArray *rhs[]) { \
    typedef multilab::matlab::Array< \
        multilab::matlab::Untyped> UntypedArray; \
    void *lhs_buf = alloca(nlhs * sizeof(mxArray*)); \
    void *rhs_buf = alloca(nrhs * sizeof(mxArray*)); \
    memcpy(lhs_buf, lhs, sizeof(mxArray*)*nlhs); \
    memcpy(rhs_buf, rhs, sizeof(mxArray*)*nrhs); \
    try { \
      funct (nrhs, (UntypedArray*)rhs_buf, \
          nlhs, (UntypedArray*)lhs_buf); \
      memcpy(lhs, lhs_buf, sizeof(nlhs) * sizeof(mxArray*)); \
    } catch(std::exception &e) { \
      mexPrintf("exception in MEX execution: %s\n", e.what()); \
      mexErrMsgTxt("fatal exception ocurred, aborting MEX execution"); \
    } \
  } \
}
#endif

#endif

