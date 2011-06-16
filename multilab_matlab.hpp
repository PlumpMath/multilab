#ifndef _MULTILAB_MATLAB_HPP_
#define _MULTILAB_MATLAB_HPP_

// MATLAB includes
#include <engine.h>
#include <matrix.h>
#include <mex.h>

#include <complex>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <boost/noncopyable.hpp>

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
      default: throw std::runtime_error("unhandled MATLAB type");
    }
  }

  static size_t matlab_class_size(mxClassID id) {
    switch(id) {
      case mxCHAR_CLASS: return
                         sizeof(matlab2cpp<mxCHAR_CLASS>::cpp_type);
      case mxDOUBLE_CLASS: return
                         sizeof(matlab2cpp<mxDOUBLE_CLASS>::cpp_type);
      case mxSINGLE_CLASS: return
                         sizeof(matlab2cpp<mxSINGLE_CLASS>::cpp_type);
      case mxINT8_CLASS: return
                         sizeof(matlab2cpp<mxINT8_CLASS>::cpp_type);
      case mxUINT8_CLASS: return
                         sizeof(matlab2cpp<mxUINT8_CLASS>::cpp_type);
      case mxINT16_CLASS: return
                         sizeof(matlab2cpp<mxINT16_CLASS>::cpp_type);
      case mxUINT16_CLASS: return
                         sizeof(matlab2cpp<mxUINT16_CLASS>::cpp_type);
      case mxINT32_CLASS: return 
                         sizeof(matlab2cpp<mxINT32_CLASS>::cpp_type);
      case mxUINT32_CLASS: return
                         sizeof(matlab2cpp<mxUINT32_CLASS>::cpp_type);
      case mxINT64_CLASS: return
                         sizeof(matlab2cpp<mxINT64_CLASS>::cpp_type);
      case mxUINT64_CLASS: return
                         sizeof(matlab2cpp<mxUINT64_CLASS>::cpp_type);
      default: throw std::runtime_error("unhandled MATLAB type");
    }
  }
}

/** \brief untyped array, essentially a wrapper for a raw mxArray* */
template<bool SCOPED=false>
class untyped_array { 
  private:
    untyped_array() { }
    template<bool B> untyped_array(const untyped_array<B> &c) { }
    template<bool B> void operator=(const untyped_array<B> &c) { }
};

// implementation of unscoped (not destroyed) untyped_array
template<>
class untyped_array<false> {
public:
  /** \brief wrap a given mxArray */
  untyped_array(mxArray *ptr) 
      : ptr_(ptr) {
  }
  // construction from any untyped_array is ok
  template<bool B>
  untyped_array(const untyped_array<B> &a) 
      : ptr_(a.get_ptr()) { }
  ~untyped_array() {
    // do not delete
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

  bool is_complex() const {
    return mxIsComplex(ptr_);
  }

  // struct stuff
  int num_fields() const {
    return mxGetNumberOfFields(ptr_);
  }
  std::string field_name(int i) const {
    const char *c = mxGetFieldNameByNumber(ptr_, i);
    if(c == NULL)
      throw std::runtime_error("couldn't get field name");
    return std::string(c);
  }
  int field_number(const std::string &s) const {
    int ret = mxGetFieldNumber(ptr_, s.c_str());
    if(ret == -1) 
      throw std::runtime_error("nonexistant field");
    return ret;
  }
  mxArray* get_field(int i, int field) const {
    mxArray *m = mxGetFieldByNumber(ptr_, i, field);
    if(m == NULL)
      throw std::runtime_error("error getting field value");
    return m;
  }

  // assignment from any untyped_array is okay.
  template<bool B>
  untyped_array<false>& operator=(const untyped_array<B> &a) {
    ptr_ = a.ptr_;
    return *this;
  }

  /** \brief access the underlying mxArray */
  mxArray* get_ptr() const {
    return ptr_;
  }
protected:
  mxArray *ptr_;
};

// implementation of scoped untyped_array
template<>
class untyped_array<true> {
public:
  /** \brief wrap a given mxArray */
  untyped_array(mxArray *ptr) 
      : ptr_(ptr) {
  }
  ~untyped_array() {
    if(ptr_) {
      mxDestroyArray(ptr_);
    }
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

  bool is_complex() const {
    return mxIsComplex(ptr_);
  }

  // struct stuff
  int num_fields() const {
    return mxGetNumberOfFields(ptr_);
  }
  std::string field_name(int i) const {
    const char *c = mxGetFieldNameByNumber(ptr_, i);
    if(c == NULL)
      throw std::runtime_error("couldn't get field name");
    return std::string(c);
  }
  int field_number(std::string s) const {
    int ret = mxGetFieldNumber(ptr_, s.c_str());
    if(ret == -1) 
      throw std::runtime_error("nonexistant field");
    return ret;
  }
  mxArray* get_field(int i, int field) const {
    mxArray *m = mxGetFieldByNumber(ptr_, i, field);
    if(m == NULL)
      throw std::runtime_error("error getting field value");
    return m;
  }

  untyped_array<true>& operator=(mxArray *a) {
    if(a == ptr_) return *this;
    if(ptr_) mxDestroyArray(ptr_);
    ptr_ = a;
    return *this;
  }

  /** \brief access the underlying mxArray */
  mxArray* get_ptr() const {
    return ptr_;
  }

protected:
  mxArray *ptr_;

private:
  // construction/assignment from any untyped_array is forbidden
  untyped_array(const untyped_array<false> &b) { }
  untyped_array(const untyped_array<true> &b) { }
  void operator=(const untyped_array<false> &b) { }
  void operator=(const untyped_array<true> &b) { }
};

/** \brief implements a typed mxArray wrapper.  this specialization is
 * for the numerical types.
 */
template<typename T, bool SCOPED=false>
class typed_array : public untyped_array<SCOPED> {
public:
  /** \brief wrap the given array */
  typed_array(mxArray *a) 
      : untyped_array<SCOPED>(a) {
    check_type();
  }
  /** \brief wrap the given array */
  template<bool B>
  typed_array(const untyped_array<B> &a) 
      : untyped_array<SCOPED>(a) {
    check_type();
  }
  /** \brief create a new array */
  typed_array(size_t rows, size_t cols, bool cpx) 
      : untyped_array<SCOPED>(mxCreateNumericMatrix(rows, cols, 
            detail::cpp2matlab<T>::matlab_class,
            cpx ? mxCOMPLEX : mxREAL)) {
    if(!untyped_array<SCOPED>::ptr_) 
      throw std::runtime_error("error creating MATLAB array");
#ifndef NDEBUG
    if(!SCOPED) {
      std::cerr 
        << "WARNING: unscoped typed_array used to hold new array; "
        << "memory leaks are likely.\n";
    }
#endif
  }
  /** \brief create a new array */
  typed_array(size_t ndim, size_t *mdims, bool cpx) 
      : untyped_array<SCOPED>(NULL) {
    mwSize matlab_dims[ndim];
    for(size_t i=0; i<ndim; ++i) matlab_dims[i] = mdims[i];
    untyped_array<SCOPED>::ptr_ = mxCreateNumericArray(ndim, matlab_dims, 
        detail::cpp2matlab<T>::matlab_class,
        cpx ? mxCOMPLEX : mxREAL);
    if(!untyped_array<SCOPED>::ptr_) 
      throw std::runtime_error("error creating MATLAB array");
#ifndef NDEBUG
    if(!SCOPED) {
      std::cerr
        << "WARNING: unscoped typed_array used to hold new array; "
        << "memory leaks are likely.\n";
    }
#endif
  }

  /** \brief wrap the given array */
  template<bool B>
  typed_array(const typed_array<T, B> &r) 
      : untyped_array<SCOPED>(r) { 
    check_type(); 
  }

  T* real_ptr() const {
    return reinterpret_cast<T*>(mxGetData(untyped_array<SCOPED>::ptr_));
  }
  T* imag_ptr() const {
    return reinterpret_cast<T*>(mxGetImagData(untyped_array<SCOPED>::ptr_));
  }

  T& real(int i) const {
    return real_ptr()[i];
  }
  T& imag(int i) const {
    return imag_ptr()[i];
  }

  mxArray* duplicate() const {
    return mxDuplicateArray(untyped_array<SCOPED>::ptr_);
  }

  void check_type() {
    mxClassID class_id = mxGetClassID(untyped_array<SCOPED>::ptr_);
    if(class_id != detail::cpp2matlab<T>::matlab_class) {
      std::stringstream ss;
      ss << "MATLAB array type mismatch: expected " 
        << detail::cpp_type_name<T>() << " got " 
        << detail::matlab_type_name(class_id);
      throw std::runtime_error(ss.str());
    }
  }

  typed_array& operator=(mxArray *m) {
    return untyped_array<SCOPED>::operator=(m);
  }
};

template<bool SCOPED>
class typed_array<char, SCOPED> : public untyped_array<SCOPED> {
public:
  typed_array(mxArray *a)
      : untyped_array<SCOPED>(a) {
    check_type();
  }
  template<bool B>
  typed_array(const untyped_array<B> &a) 
      : untyped_array<SCOPED>(a) {
    check_type();
  }
  typed_array(size_t ndim, size_t *mdims) 
      : untyped_array<SCOPED>(NULL) {
    mwSize matlab_dims[ndim];
    for(size_t i=0; i<ndim; ++i) matlab_dims[i] = mdims[i];
    untyped_array<SCOPED>::ptr_ = mxCreateCharArray(ndim, matlab_dims);
    if(!untyped_array<SCOPED>::ptr_) 
      throw std::runtime_error("error creating MATLAB array");
#ifndef NDEBUG
    if(!SCOPED) {
      std::cerr 
        << "WARNING: unscoped typed_array used to hold new array; "
        << "memory leaks are likely.\n";
    }
#endif
  }
  // copy ctors
  template<bool B>
  typed_array(const typed_array<char, B> &r) 
      : untyped_array<SCOPED>(r) { 
    check_type(); 
  }

  void check_type() {
    mxClassID class_id = mxGetClassID(untyped_array<SCOPED>::ptr_);
    if(class_id != detail::cpp2matlab<char>::matlab_class) {
      std::stringstream ss;
      ss << "MATLAB array type mismatch: expected " 
        << detail::cpp_type_name<char>() << " got " 
        << detail::matlab_type_name(class_id);
      throw std::runtime_error(ss.str());
    }
  }

  std::string to_string() {
    char *c = mxArrayToString(untyped_array<SCOPED>::ptr_);
    if(!c) throw std::runtime_error("error getting MATLAB string");
    std::string to_return(c);
    mxFree(c);
    return to_return;
  }
};

/** \brief instance of a MATLAB engine for embedding in python (or c++) */
template<int UNUSED=0>
class engine_ {
public:
  engine_()
      : eng_(engOpen("\0")) { 
    if(eng_ == NULL) 
      throw std::runtime_error("error starting MATLAB engine");
  }
  engine_(const std::string &command)
      : eng_(engOpen(command.c_str())) {
    if(eng_ == NULL)
      throw std::runtime_error("error starting MATLAB engine");
  }
  ~engine_() {
    engClose(eng_);
  }

  mxArray* get(const std::string &name) {
    mxArray *tmp = engGetVariable(eng_, name.c_str());
    if(tmp == NULL)
      throw std::runtime_error("error getting variable from MATLAB");
    return tmp;
  }
  template<bool B>
  void put(const std::string &name, untyped_array<B> &a) {
    if(engPutVariable(eng_, name.c_str(), a.get_ptr()) == 1)
      throw std::runtime_error("error putting variable into MATLAB");
  }
  void eval(const std::string &str) {
    if(engEvalString(eng_, str.c_str()) == 1) {
      throw std::runtime_error("error evaluating MATLAB command");
    }
  }

private:
  Engine *eng_;
};
typedef engine_<0> engine;

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

