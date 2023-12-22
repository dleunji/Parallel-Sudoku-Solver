#include "SudokuSolver_CudaBruteForce_kernel.cuh"

__global__
void backtrack_kernel(int n, int N, int *boards, int *found)
{
  printf("threadIdx.x: %d\n", threadIdx.x);
}

void call_backtrack(int *boards, int numberOfBoards, int n, int N, int &found)
{

  int *d_boards, *d_found;
  cudaMalloc((void **) d_boards, N * N * numberOfBoards * sizeof(int));
  cudaMalloc((void **) d_found, sizeof(int));

  cudaMemcpy(d_boards, boards, N * N * numberOfBoards * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_found, -1, sizeof(int));

  dim3 gridDim(numberOfBoards);
  dim3 blockDim(N, N);

  printf("call_backtrack! numberOfBoards: %d N: %d\n", numberOfBoards, N);
  backtrack_kernel<<<gridDim, blockDim, (N * N + N + 4) * sizeof(int)>>>(n, N, d_boards, d_found);
  cudaDeviceSynchronize();

  cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
  

}