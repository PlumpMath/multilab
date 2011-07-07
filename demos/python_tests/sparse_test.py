#!/usr/bin/env /bin/bash
import sys
sys.path.append("../../")
import multilab

e = multilab.engine()
e.eval("x = (magic(5))")
x = e.get("x")
print "dense:"
print x

e.eval("x = sparse(x)")
x = e.get("x").todense()
print "sparse:"
print x

