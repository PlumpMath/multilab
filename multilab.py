import multilab_private
import numpy

class engine(object):
  def __init__(self, cmd = ""):
    self.engine_ = multilab_private.engine(cmd)
    self.convert_mapping_ = { \
        multilab_private.mx_class_id.cell_class: None,
        multilab_private.mx_class_id.char_class: \
            self.__get_string_vec,
        multilab_private.mx_class_id.double_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.function_class: None,
        multilab_private.mx_class_id.int8_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.int16_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.int32_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.int64_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.logical_class: \
            self.__get_logical_vec,
        multilab_private.mx_class_id.single_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.struct_class: \
            self.__get_struct,
        multilab_private.mx_class_id.uint8_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.uint16_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.uint32_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.uint64_class: \
            self.__get_numerical_vec,
        multilab_private.mx_class_id.void_class: None
        }

  def eval(self, cmd):
    self.engine_.eval(cmd)

  def get(self, varname):
    wrapper = self.engine_.get(varname)
    return self.pythonize_wrapper_(wrapper)

  def set_dense_real(self, varname, val):
    matlab_ordered = numpy.array(val, dtype=val.dtype, order='F')
    self.engine_.set_dense_vector_real(varname, matlab_ordered.shape,\
        matlab_ordered)

  def pythonize_wrapper_(self, wrapper):
    handler = self.convert_mapping_[wrapper.get_type()]
    if handler is not None:
      return handler(wrapper)
    else:
      raise TypeError("cannot convert from MATLAB type ",\
          wrapper.get_type())

  def __unwrap_coords(self, index, size):
    coord_list = []
    modulus = 1
    for i in range(len(size)):
      modulus = modulus * size[i]
      coord_list.append( index % modulus )
    return tuple(coord_list)

  def __get_struct(self, wrapper):
    dims = wrapper.get_dims()
    size = reduce(lambda x,y: x*y, dims)
    num_fields = wrapper.num_fields()
    if size > 1:
      # maybe there's a better way to do this, but this is fast and easy
      to_ret = {}
      for idx in range(size):
        stct = {}
        for fi in range(num_fields):
          field_wrapper = wrapper.get_field(idx, fi)
          stct[wrapper.field_name(fi)] = \
              self.pythonize_wrapper_(field_wrapper)
        idx_coords = self.__unwrap_coords(idx, dims)
        to_ret[idx_coords] = stct
      return to_ret
    else:
      # return singleton array
      to_ret = {}
      for fi in range(num_fields):
        field_wrapper = wrapper.get_field(0, fi)
        to_ret[wrapper.field_name(fi)] = \
            self.pythonize_wrapper_(field_wrapper)
      return to_ret

  def __get_numerical_vec(self, wrapper):
    vec = None
    if wrapper.is_complex():
      vec = wrapper.real_part() + (wrapper.imag_part() * 1j)
    else:
      vec = wrapper.real_part()
    if wrapper.is_sparse():
      return self.__sparsify(wrapper, vec.T.flatten())
    else:
      return vec

  def __sparsify(self, wrapper, vec):
    from scipy.sparse import csc_matrix
    row_ind = wrapper.row_coords()
    col_ptr = wrapper.col_index_counts()
    return csc_matrix( (vec, row_ind, col_ptr), wrapper.get_dims() )

  def __get_string_vec(self, wrapper):
    return wrapper.as_string()

  def __get_logical_vec(self, wrapper):
    return wrapper.as_logical()

