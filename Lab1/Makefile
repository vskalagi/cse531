CXXFLAGS = -O2 -std=c++11 -fopenmp

all: solver_seq solver_omp

solver_seq: solver_seq.cc main.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

solver_omp: solver_omp.cc main.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

run: solver_omp
	./solver_omp dataset_easy.txt dataset_hard.txt dataset_benchmark.txt