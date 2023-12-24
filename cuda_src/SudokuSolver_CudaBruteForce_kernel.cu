#include "SudokuSolver_CudaBruteForce_kernel.cuh"

#define ROW(p, N) ((p) / N)
#define COL(p, N) ((p) % N)
#define BOX(p, n) ((p) / (n * n * n) * n + ((p) % (n * n)) / n)

__global__ void backtrack_kernel(int n, int N, int *boards, int *found)
{
  // use shared memory
  extern __shared__ int shared[];
  int board_num = boards[blockIdx.x * N * N + (threadIdx.x * N) + threadIdx.y];
  int *failed = shared + 4;
  int *stack = shared + N + 4;
  int top = 0;
  // put all locations of empty tiles in stack
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    int i;
    shared[0] = -1;
    for (i = 0; i < N * N; ++i)
    {
      if (boards[blockIdx.x * N * N + i] == 0)
        stack[top++] = i;
    }
    stack[top] = -1;
    top = 0;
  }
  int box_now = threadIdx.x / n * n + threadIdx.y / n;
  __syncthreads();

  int last_op = 0; // 0 - push stack, 1 - pop stack
  while (*found == -1)
  {
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      int stack_num = stack[top];
      shared[0] = stack_num % (N * N); // communicate nowp
      if (last_op == 0)
      {
        // first check if the board is filled
        if (stack_num == -1)
        {
          // answer found
          atomicCAS(found, -1, blockIdx.x);
          if (*found == blockIdx.x && threadIdx.x == 0 && threadIdx.y == 0)
          {
            for (int i = 0; i < N; i++)
            {
              for (int j = 0; j < N; j++)
              {
                printf("%d ", boards[blockIdx.x * N * N + i * N + j]);
              }
              printf("\n");
            }
          }
          shared[0] = -1;
        }
        // else initialize the number to try
        shared[1] = 1;
      }
      else
      {
        shared[1] = stack_num / (N * N) + 1;
      }
    }
    if (threadIdx.y == 0)
      failed[threadIdx.x] = 0;
    __syncthreads();

    // find next valid number
    int nowp = shared[0], i = shared[1];
    if (nowp == -1)
      break;
    int num_to_try = i;
    if (ROW(nowp, N) == threadIdx.x || COL(nowp, N) == threadIdx.y || BOX(nowp, n) == box_now)
    {
      for (; i <= N; ++i)
        if (i == board_num)
          failed[i - 1] = 1;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      for (i = num_to_try; i <= N && failed[i - 1] != 0; ++i)
        ;
      shared[2] = nowp; // record in shared memory for communication
      if (i <= N)
      {
        // push stack
        stack[top++] = i * N * N + nowp;
        shared[3] = i;
        last_op = 0;
      }
      else
      {
        // pop stack
        if (top == 0)
          shared[2] = -1;
        stack[top--] = nowp;
        shared[3] = 0;
        last_op = 1;
      }
    }
    __syncthreads();
    if (shared[2] == -1)
      break;
    // update local register of last step
    if (threadIdx.x * N + threadIdx.y == shared[2])
      board_num = shared[3];
  }

  if (*found == blockIdx.x)
    boards[blockIdx.x * N * N + (threadIdx.x * N) + threadIdx.y] = board_num;
}

bool call_backtrack(int *boards, int numberOfBoards, int n, int N)
{
  int found;
  int *d_boards, *d_found;
  checkCudaErrors(cudaMalloc((void **)&d_boards, N * N * numberOfBoards * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_found, sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_boards, boards, N * N * numberOfBoards * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_found, -1, sizeof(int)));

  dim3 gridDim(numberOfBoards);
  dim3 blockDim(N, N);

  backtrack_kernel<<<gridDim, blockDim, (N * N + N + 4) * sizeof(int)>>>(n, N, d_boards, d_found);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
  if (found >= 0)
  {
    checkCudaErrors(cudaMemcpy(boards, d_boards + found * N * N, N * N * sizeof(int), cudaMemcpyDeviceToHost));
    found = 1;
  }
  checkCudaErrors(cudaFree(d_found));
  checkCudaErrors(cudaFree(d_boards));
  return found == 1 ? true : false;
}