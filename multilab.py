import multilab_private
import numpy

class matlab_handle(object):
  def __init__(self, engine_obj, obj_name):
    self.engine_ = engine_obj
    self.name_ = obj_name

class engine(object):
  def __init__(self, cmd = ""):
    self.engine_ = multilab_private.engine(cmd)
    self.convert_mapping_ = { \
        multilab_private.mx_class_id.cell_class: None,
        multilab_private.mx_class_id.char_class: \
            self.get_string_vec_,
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
        multilab_private.mx_class_id.logical_class: \
            self.get_logical_vec_,
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
    self.handles_ = {}

  def read_workspace(self):
    self.engine_.eval("multilab_workspace = whos")
    ml_workspace_objs = self.get("multilab_workspace")
    self.handles_.clear()

  def make_handle(self, obj_name):
    if obj_name in self.handles_.keys():
      return self.handles_.keys[obj_name]
    new_handle = matlab_handle(self, obj_name)
    self.handles_[obj_name] = new_handle
    return new_handle

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

  def unwrap_coords_(self, index, size):
    coord_list = []
    modulus = 1
    for i in range(len(size)):
      modulus = modulus * size[i]
      coord_list.append( index % modulus )
    return tuple(coord_list)

  def get_struct_(self, wrapper):
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
        idx_coords = self.unwrap_coords_(idx, dims)
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

  def get_numerical_vec_(self, wrapper):
    if wrapper.is_complex():
      return wrapper.real_part() + (wrapper.imag_part() * 1j)
    else:
      return wrapper.real_part()

  def get_string_vec_(self, wrapper):
    return wrapper.as_string()

  def get_logical_vec_(self, wrapper):
    return wrapper.as_logical()

