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
    return self.pythonize_wrapper_(wrapper)

  def pythonize_wrapper_(self, wrapper):
    handler = self.convert_mapping_[wrapper.get_type()]
    if handler is not None:
      return handler(wrapper)
    else:
      raise TypeError("cannot convert from MATLAB type ",\
          wrapper.get_type())

  def get_struct_(self, wrapper):
    size = reduce(lambda x,y: x*y, wrapper.get_dims())
    if size > 1:
      raise TypeError("sorry, no support for struct arrays now")
    to_ret = {}
    num_fields = wrapper.num_fields()
    for fi in range(num_fields):
      field_wrapper = wrapper.get_field(0, fi)
      to_ret[wrapper.field_name(fi)] = \
          self.pythonize_wrapper_(field_wrapper)
    return to_ret

  def get_numerical_vec_(self, wrapper):
    if wrapper.is_complex():
      return wrapper.real_part() + (wrapper.imag_part() * 1j)
    else:
      return wrapper.real_part()

