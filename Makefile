# the compiler: gcc for C program, g++ for C++ program
CXX = g++
GIT_VERSION := "$(shell git describe --abbrev=0 --tags)"
# compiler flags:
#  -g      adds debugging information to the executable file
#  -Wall   turns on most, but not all, compiler warnings
#  -Wextra enables some extra warning flags that are not enabled by -Wall
CXX_FLAGS = --std=c++17 -g -Wall -Wextra -O3 -DVERSION=\"$(GIT_VERSION)\"
OPENMP = -fopenmp
#CUDA
CUDA_PATH ?= /usr/local/cuda
CUDA_CXXFLAGS =
CUDA_CXXFLAGS += -std=c++17
CUDA_CXXFLAGS += -m64
CUDA_CXXFLAGS += -O3 -arch=sm_86
CUDA_CXXFLAGS += -Xcompiler -fopenmp
CUDA_LDFLAGS = -lcublas -lcusparse -lcudart -lcusparseLt
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
# All output target executables
TARGETS = sudoku_main
# All object files
OBJECTS = *.o *.out
DEPENDENCIES = \
  ./src/SudokuBoard.cpp \
  ./src/SudokuBoardDeque.cpp \
  ./src/SudokuTest.cpp \
  ./src/SudokuSolver.cpp \
  ./src/SudokuSolver_SequentialBacktracking.cpp \
  ./src/SudokuSolver_SequentialBruteForce.cpp \
  ./src/SudokuSolver_ParallelBruteForce.cpp \
  ./src/SudokuSolver_SequentialForwardChecking.cpp \
  ./src/SudokuSolver_CudaBruteForce.cpp \
  ./src/SudokuSolver_NestedBruteForce.cpp \
  ./src/SudokuSolver_NestedBacktracking.cpp

CUDA_DEPENDENCIES = \
  ./cuda_src/SudokuSolver_CudaBruteForce_kernel.cu

all: $(TARGETS)

sudoku_main: sudoku_main.cpp $(CUDA_DEPENDENCIES) $(DEPENDENCIES)
		$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_LDFLAGS) -I ./inc -I ./cuda_inc -o $@ $^

clean:
		rm -f $(TARGETS) $(OBJECTS) solution.txt