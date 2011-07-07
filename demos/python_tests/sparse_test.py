#!/usr/bin/env /bin/bash
import sys
sys.path.append("../../")
import multilab

e = multilab.engine()
e.eval("x = sparse(magic(5))")
x = e.get("x")

