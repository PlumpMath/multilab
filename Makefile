PREFIX=/usr

install:
	rm -rf ${PREFIX}/include/multilab
	mkdir ${PREFIX}/include/multilab
	cp *.hpp ${PREFIX}/include/multilab
