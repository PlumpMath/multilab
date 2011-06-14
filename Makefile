# Makefile for multilab
# 
# you may not need to build anything in order to use multilab:
# 1. if all you want is the matlab api wrapper, just include
# 	multilab_matlab.hpp in your code.  no building or linking is
# 	necessary beyond the matlab sources.  optionally install to a
# 	systemwide directory.
# 2. if you want to use the matlab engine module for python, set the
# 	variables BOOSTPREFIX, BOOSTLIB, MATLABDIR, MATLABARCH, PYTHONDIR,
# 	PYTHONLIB and PYSHARED for your machine and build.
ifndef PREFIX
PREFIX=/usr
endif

ifndef BOOSTPREFIX
BOOSTPREFIX=/usr/include
endif

ifndef BOOSTLIB
BOOSTLIB=-lboost_python-py27
endif

ifndef MATLABDIR
MATLABDIR=/opt/matlab2008b
endif

ifndef MATLABARCH
MATLABARCH=glnxa64
endif

ifndef PYTHONDIR
PYTHONDIR=/usr/include/python2.7
endif

ifndef PYTHONLIB
PYTHONLIB=-lpython2.7
endif

ifndef PYSHARED
PYSHARED=/usr/share/pyshared
endif

install_headers:
	rm -rf ${PREFIX}/include/multilab
	mkdir ${PREFIX}/include/multilab
	cp *.hpp ${PREFIX}/include/multilab

CXX=g++
CXXFLAGS=-g3 -Wall -Wextra -I${MATLABDIR}/extern/include \
				 -I${PYSHARED} \
				 -I${PYTHONDIR} \
				 -L${MATLABDIR}/bin/${MATLABARCH} \
				 -Wl,-rpath,${MATLABDIR}/bin/${MATLABARCH} \
				 -fPIC -shared \
				 ${BOOSTLIB} ${PYTHONLIB} \
				 -lmex -leng

EXT_OBJS=multilab_pyext.o
EXT_OUT=multilab.so

python: ${EXT_OUT}

${EXT_OUT}: ${EXT_OBJS}
	${CXX} ${CXXFLAGS} -o $@ $^

