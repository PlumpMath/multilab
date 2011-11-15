MULTILABROOT=../../../
MATLAB=/opt/matlab2008b

MEX=${MATLAB}/bin/mex
MEXEXT=${MATLAB}/bin/mexext

CXX=g++
CXXFLAGS=-g3 -Wall -Wextra -I${MULTILABROOT} \
				 -I${MATLAB}/extern/include -fPIC \
				 -std=c++0x

OBJS= new_array.o
OUT=new_array.$(shell ${MEXEXT})

all: ${OUT}

${OUT}: ${OBJS}
	${MEX} -cxx -o ${OUT} ${OBJS}

clean:
	${RM} ${OBJS} ${OUT}

