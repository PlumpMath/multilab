import multilab_private
import numpy

class engine(object):
  def __init__(self, cmd = ""):
    self.engine_ = multilab_private.engine(cmd)

  def eval(self, cmd):
    self.engine_.eval(cmd)

  def get(self, varname):
    wrapper = self.engine_.get(varname)
    if wrapper.is_complex():
      return wrapper.real_part() + wrapper.imag_part()*1j

