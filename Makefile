SOURCES = 2020CS10354.cpp 
FLAGS = -fopenmp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
test : 2020CS10354.cpp
	g++ -I /usr/include/mkl $(SOURCES) -o ./yourcode.out $(FLAGS)

clean:
	rm -rf *o ./yourcode.out
