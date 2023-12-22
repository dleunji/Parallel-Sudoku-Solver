#include <stdio.h>
#include "SudokuSolver_CudaBruteForce_kernel.hpp"

__global__
void backtrack_kernel()
{
  printf("threadIdx.x: %d\n", threadIdx.x);
}

void call_solve(int numberOfBoards, int N)
{
  dim3 gridDim(numberOfBoards);
  dim3 blockDim(N * N);

  backtrack_kernel<<<gridDim, blockDim, (N * N + N + 4) * sizeof(int)>>>();
}