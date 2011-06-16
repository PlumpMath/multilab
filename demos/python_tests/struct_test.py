#!/usr/bin/env /bin/bash

import sys
sys.path.append("../../")
import multilab

e = multilab.engine()
e.eval("x = {}")
e.eval("x.a = 5")
e.eval("x.b = 42")
e.eval("x.c = eye(7)")

x = e.get("x")

