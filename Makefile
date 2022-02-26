SOURCES = 2020CS10318_2020CS10354.cpp openblasmat.h
FLAGS = -fopenmp -lmkl_intel_lp64 -lmkl_core -pthread -lmkl_gnu_thread -lpthread -lm -ldl -lopenblas
test : 2020CS10318_2020CS10354.cpp
	g++ -I /usr/include/mkl -I /usr/include/openblas $(SOURCES) -o ./yourcode.out $(FLAGS)

clean:
	rm -rf *o ./yourcode.out
