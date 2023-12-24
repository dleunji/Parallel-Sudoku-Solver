#include "SudokuSolver_CudaBruteForce_kernel.cuh"

#define ROW(p, N) ((p) / N)
#define COL(p, N) ((p) % N)
#define BOX(p, n) ((p) / (n * n * n) * n + ((p) % (n * n)) / n)

__global__ void backtrack_kernel(int n, int N, int *boards, int *found)
{
  // use shared memory
  extern __shared__ int shared[];
  int tid = blockIdx.x * N * N + (threadIdx.x * N) + threadIdx.y;
  // board_num = chosen number of the current position
  int board_num = boards[tid];
  // size = N, keep track of numbers that have been tried and failed for a particular position
  int *failed = shared + 4;
  // size = N * N
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
  if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
  {
    printf("top: %d\n", top);
  }
  while (*found == -1)
  {
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      // position -> shared[0], number -> shared[1]
      int stack_num = stack[top];
      if (blockIdx.x == 0)
      {
        printf("stack num : %d\n", stack_num);
      }
      shared[0] = stack_num % (N * N); // communicate nowp
      if (last_op == 0)
      {
        // first check if the board is filled (empty queue)
        if (stack_num == -1)
        {
          // answer found
          atomicCAS(found, -1, blockIdx.x);
          shared[0] = -1;
        }
        // else initialize the number to try (new position)
        shared[1] = 1;
      }
      else
      {
        // the number is saved in zero-base, but used in one-base
        shared[1] = stack_num / (N * N) + 1;
      }
    }

    // initialize failed array whose size of N
    if (threadIdx.y == 0)
      failed[threadIdx.x] = 0;
    __syncthreads();

    // find next valid number
    // nowp = the position to check in this board(block)
    int nowp = shared[0], i = shared[1];
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    {
      printf("CURR: i = %d nowp = %d\n", i, nowp);
    }
    if (nowp == -1)
      break;

    // temp for saving the `i`
    int num_to_try = i;
    // each thread checks whether the current position (`nowp`) is in it respective row, column, or box.
    // In case of the threads in the row, col, box including the `nowp`
    if (ROW(nowp, N) == threadIdx.x || COL(nowp, N) == threadIdx.y || BOX(nowp, n) == box_now)
    {
      for (; i <= N; ++i)
        if (i == board_num)
          failed[i - 1] = 1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    {
      printf("compare with board_num: %d\n", board_num);
      for (int i = 0; i < N; i++)
      {
        printf("%d ", failed[i]);
      }
      printf("\n");
    }
    // As the result, `i`  = has been tried and failed at this position
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
      // i have not been tried
      for (i = num_to_try; i <= N && failed[i - 1] != 0; ++i)
        ;
      shared[2] = nowp; // record in shared memory for communication
      if (i <= N)
      {
        // push stack
        // push `i`(number) and `nowp`(position) together
        stack[top++] = i * N * N + nowp;
        shared[3] = i;
        last_op = 0;
        if (blockIdx.x == 0)
        {
          printf("PUSH: i = %d nowp = %d\n", i, nowp);
        }
      }
      else
      {
        // pop stack
        if (top == 0)
          // the stack becomes empty!
          shared[2] = -1;
        stack[top--] = nowp;
        shared[3] = 0;
        last_op = 1;
        if (blockIdx.x == 0)
        {
          printf("PUSH: %d\n", nowp);
        }
      }
    }
    __syncthreads();
    if (shared[2] == -1)
      break;
    // update local register of last step
    // shared[3] matches with the position nowp = shared[2]
    if (threadIdx.x * N + threadIdx.y == shared[2])
      // shared[3] = the value of latest operation.
      board_num = shared[3];
  }

  // after cooperation of threads in the block, the final chose board_num is mapped to position
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