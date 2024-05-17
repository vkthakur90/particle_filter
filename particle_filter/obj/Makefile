all:
	g++ --std=c++23 -O3 -fopenmp -march=native -fpic -shared ../src/filter_d.cpp -o ../bin/filter_d.dll -I../include 
	g++ --std=c++23 -O3 -fopenmp -march=native -fpic -shared ../src/filter_f.cpp -o ../bin/filter_f.dll -I../include
	gcc -fopenmp  ../src/main.c -o ../bin/pf -ldl -I../include  

clean:
	rm -f *.o 
	rm -f ../bin/*
