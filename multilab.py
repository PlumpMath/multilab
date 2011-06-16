import multilab_private
import numpy

class engine(object):
  def __init__(self, cmd = ""):
    self.engine_ = multilab_private.engine(cmd)
    self.convert_mapping_ = { \
        multilab_private.mx_class_id.cell_class: None,
        multilab_private.mx_class_id.char_class: None,
        multilab_private.mx_class_id.double_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.function_class: None,
        multilab_private.mx_class_id.int8_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.int16_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.int32_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.int64_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.logical_class: None,
        multilab_private.mx_class_id.single_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.struct_class: \
            self.get_struct_,
        multilab_private.mx_class_id.uint8_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.uint16_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.uint32_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.uint64_class: \
            self.get_numerical_vec_,
        multilab_private.mx_class_id.void_class: None
        }

  def eval(self, cmd):
    self.engine_.eval(cmd)

  def get(self, varname):
    wrapper = self.engine_.get(varname)
    handler = self.convert_mapping_[wrapper.get_type()]
    if handler is not None:
      return handler(wrapper)
    else:
      raise TypeError("cannot convert from MATLAB type ",\
          wrapper.get_type())

  def get_struct_(self, wrapper):
    pass

  def get_numerical_vec_(self, wrapper):
    if wrapper.is_complex():
      return wrapper.real_part() + (wrapper.imag_part() * 1j)
    else:
      return wrapper.real_part()

