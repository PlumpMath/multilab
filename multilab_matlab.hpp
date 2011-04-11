#ifndef _MULTILAB_MATLAB_HPP_
#define _MULTILAB_MATLAB_HPP_

#include <matrix.h>
#include <mex.h>

#include <complex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <alloca.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace multilab {
namespace matlab {

/** \brief contains nuts and bolts; ignore */
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
  // TIE_MATLAB_CPP(mxCELL_CLASS, ?)
  // TIE_MATLAB_CPP(mxSTRUCT_CLASS, ?)
  // TIE_MATLAB_CPP(mxFUNCTION_CLASS, ?)
  // TIE_MATLAB_CPP(mxLOGICAL_CLASS, bool);
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
    }
  }
}

/** \brief untyped array, essentially a wrapper for a raw mxArray* */
template<int UNUSED=0>
class untyped_array_ {
public:
  /** \brief wrap a given mxArray */
  untyped_array_(mxArray *ptr) 
      : ptr_(ptr) {
  }
  ~untyped_array_() {
  }

  /** \brief MATLAB class checking */
  mxClassID get_type() const {
    return mxGetClassID(ptr_);
  }
  /** \brief MATLAB class checking */
  template<typename T> bool is_of_type() const {
    return get_type() == detail::cpp2matlab<T>::matlab_class;
  }
  /** \brief MATLAB class checking */
  bool is_of_type(mxClassID c) const {
    return get_type() == c;
  }

  /** \brief access the underlying mxArray */
  operator mxArray*() {
    return ptr_;
  }

  /** \brief access the underlying mxArray */
  mxArray* get_ptr() const {
    return ptr_;
  }

protected:
  mxArray *ptr_;
};
typedef untyped_array_<0> untyped_array;

/** \brief implements a typed mxArray wrapper.  this specialization is
 * for the numerical types.
 */
template<typename T>
class typed_array {
public:
  /** \brief wrap the given array */
  typed_array(mxArray *a) 
      : ptr_(a) {
    check_type();
  }
  /** \brief wrap the given array */
  typed_array(const untyped_array &a) 
      : ptr_(a.get_ptr()) {
    check_type();
  }
  /** \brief create a new array */
  typed_array(size_t rows, size_t cols, bool cpx) 
      : ptr_(mxCreateNumericMatrix(rows, cols, 
            detail::cpp2matlab<T>::matlab_class,
            cpx ? mxCOMPLEX : mxREAL)) {
    if(!ptr_) throw std::runtime_error("error creating MATLAB array");
  }
  /** \brief create a new array */
  typed_array(size_t ndim, size_t *mdims, bool cpx) 
      : ptr_(NULL) {
    mwSize matlab_dims[ndim];
    for(size_t i=0; i<ndim; ++i) matlab_dims[i] = mdims[i];
    ptr_ = mxCreateNumericArray(ndim, matlab_dims, 
        detail::cpp2matlab<T>::matlab_class,
        cpx ? mxCOMPLEX : mxREAL);
    if(!ptr_) throw std::runtime_error("error creating MATLAB array");
  }

  /** \brief wrap the given array */
  typed_array(const typed_array &r) : ptr_(mxDuplicateArray(r.ptr_)) { 
    check_type(); 
  }

  size_t num_dims() const {
    return mxGetNumberOfDimensions(ptr_);
  }
  const mwSize* dims() const {
    return mxGetDimensions(ptr_);
  }
  std::vector<size_t> get_dims() const {
    size_t nd = num_dims();
    std::vector<size_t> to_return(nd);
    mwSize *ml_dims = dims();
    for(size_t i=0; i<nd; ++i) to_return[i] = ml_dims[i];
    return to_return;
  }

  bool is_complex() const {
    return mxIsComplex(ptr_);
  }

  T* real_ptr() const {
    return reinterpret_cast<T*>(mxGetData(ptr_));
  }
  T* imag_ptr() const {
    return reinterpret_cast<T*>(mxGetImagData(ptr_));
  }

  typed_array& operator=(const typed_array &a) {
    if(ptr_ == a.ptr_) return *this;
    ptr_ = mxDuplicateArray(a.ptr_);
    return *this;
  }
  typed_array& operator=(mxArray *a) {
    ptr_ = a;
    return *this;
  }

  void check_type() {
    mxClassID class_id = mxGetClassID(ptr_);
    if(class_id != detail::cpp2matlab<T>::matlab_class) {
      std::stringstream ss;
      ss << "MATLAB array type mismatch: expected " 
        << detail::cpp_type_name<T>() << " got " 
        << detail::matlab_type_name(class_id);
      throw std::runtime_error(ss.str());
    }
  }

  operator mxArray*() {
    return ptr_;
  }
  operator untyped_array() {
    return untyped_array_<0>(ptr_);
  }
private:
  mxArray *ptr_;
};

template<>
class typed_array<char> {
public:
  typed_array(mxArray *a)
      : ptr_(a) {
    check_type();
  }
  typed_array(const untyped_array &a) 
      : ptr_(a.get_ptr()) {
    check_type();
  }
  typed_array(size_t ndim, size_t *mdims) 
      : ptr_(NULL) {
    mwSize matlab_dims[ndim];
    for(size_t i=0; i<ndim; ++i) matlab_dims[i] = mdims[i];
    ptr_ = mxCreateCharArray(ndim, matlab_dims);
    if(!ptr_) throw std::runtime_error("error creating MATLAB array");
  }
  ~typed_array() {
  }
  // copy ctors
  typed_array(const typed_array &r) : ptr_(mxDuplicateArray(r.ptr_)) { 
    check_type(); 
  }

  typed_array& operator=(const typed_array &a) {
    if(ptr_ == a.ptr_) return *this;
    ptr_ = mxDuplicateArray(a.ptr_);
    return *this;
  }
  typed_array& operator=(mxArray *a) {
    ptr_ = a;
    return *this;
  }

  void check_type() {
    mxClassID class_id = mxGetClassID(ptr_);
    if(class_id != detail::cpp2matlab<char>::matlab_class) {
      std::stringstream ss;
      ss << "MATLAB array type mismatch: expected " 
        << detail::cpp_type_name<char>() << " got " 
        << detail::matlab_type_name(class_id);
      throw std::runtime_error(ss.str());
    }
  }

  size_t num_dims() const {
    return mxGetNumberOfDimensions(ptr_);
  }
  const mwSize* dims() const {
    return mxGetDimensions(ptr_);
  }
  std::vector<size_t> get_dims() const {
    size_t nd = num_dims();
    std::vector<size_t> to_return(nd);
    const mwSize *ml_dims = dims();
    for(size_t i=0; i<nd; ++i) to_return[i] = ml_dims[i];
    return to_return;
  }

  std::string to_string() {
    char *c = mxArrayToString(ptr_);
    if(!c) throw std::runtime_error("error getting MATLAB string");
    std::string to_return(c);
    mxFree(c);
    return to_return;
  }

  operator mxArray*() {
    return ptr_;
  }
  operator untyped_array() {
    return untyped_array_<0>(ptr_);
  }

private:
  mxArray *ptr_;
};

}}

#ifndef MEX_ENTRY_POINT
#define MEX_ENTRY_POINT(funct) \
extern "C" { \
  void mexFunction(int nlhs, mxArray *lhs[], int nrhs, \
      const mxArray *rhs[]) { \
    void *lhs_buf = alloca(nlhs * sizeof(mxArray*)); \
    void *rhs_buf = alloca(nrhs * sizeof(mxArray*)); \
    memcpy(lhs_buf, lhs, sizeof(mxArray*)*nlhs); \
    memcpy(rhs_buf, rhs, sizeof(mxArray*)*nrhs); \
    try { \
      funct (nlhs, (multilab::matlab::untyped_array*)lhs_buf, \
          nrhs, (multilab::matlab::untyped_array*)rhs_buf); \
      memcpy(lhs, lhs_buf, sizeof(nlhs * sizeof(mxArray*))); \
    } catch(std::exception &e) { \
      mexPrintf("exception in MEX execution: %s\n", e.what()); \
      mexErrMsgTxt("fatal exception ocurred, aborting MEX execution"); \
    } \
  } \
}
#endif

#endif

