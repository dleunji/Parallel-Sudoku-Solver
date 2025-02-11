#ifndef SUDOKUSOLVER_CUDABRUTEFORCE_HPP
#define SUDOKUSOLVER_CUDABRUTEFORCE_HPP

#include "SudokuBoard.hpp"
#include "SudokuSolver.hpp"
#include "SudokuBoardDeque.hpp"
#include "SudokuSolver_CudaBruteForce_kernel.cuh"

class SudokuSolver_CudaBruteForce : public SudokuSolver
{
private:
    SudokuBoardDeque _board_deque;
    int *_boards;

public:
    SudokuSolver_CudaBruteForce(SudokuBoard &board, bool print_message = true);

    // Divides one Sudoku problem into several simpler sub-problems and push them to the end of board deque
    void bootstrap_openmp();
    void bootstrap();
    void bootstrap(SudokuBoardDeque &boardDeque, int indexOfRows);

    // Solves the given Sudoku board using parallel brute force algorithm
    virtual void solve() override
    {
        /* Choose one of the following kernels to execute */
        solve_kernel_1();
        // solve_kernel_2();
        // solve_bruteforce_par(_board, 0, 0);
    }

    void solve_kernel_1();
};

#endif // SUDOKUSOLVER_CUDABRUTEFORCE_HPP