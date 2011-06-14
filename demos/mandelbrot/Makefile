CLROOT=/usr/include/nvidia-current
MULTILABROOT=../../../
CLWRAPPERROOT=../../../
MATLAB=/opt/matlab2008b

MEX=${MATLAB}/bin/mex

CXX=g++
CXXFLAGS=-g3 -Wall -Wextra -I${CLROOT} -I${MULTILABROOT} \
				 -I${MATLAB}/extern/include -fPIC

OBJS= kernel.opencl.o mandelbrot.o
OUT=mandelbrot

all: ${OUT}

kernel.opencl.cpp: kernel.opencl
	${CLWRAPPERROOT}/cl_wrapper/clc/clc $^

${OUT}: ${OBJS}
	${MEX} -cxx -o ${OUT} ${OBJS} -lOpenCL

clean:
	${RM} kernel.opencl.[hc]pp ${OBJS} ${OUT}

