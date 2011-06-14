ifndef MATLABDIR
MATLABDIR=/opt/matlab2008b
endif

ifndef MULTILABROOT
MULTILABROOT=../../../
endif

ifndef MATLABARCH
MATLABARCH=glnxa64
endif

CXX=g++
CXXFLAGS=-g3 -Wall -Wextra -I${MATLABDIR}/extern/include \
				 -I${MULTILABROOT} \
				 -L${MATLABDIR}/bin/${MATLABARCH} \
				 -Wl,-rpath,${MATLABDIR}/bin/${MATLABARCH}

OUT=matlab_embed

${OUT}: embed.o
	${CXX} ${CXXFLAGS} -o ${OUT} $^ -lmex -leng

clean:
	${RM} embed.o ${OUT}

