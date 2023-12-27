#ifndef SudokuSolver_NestedBruteForce_HPP
#define SudokuSolver_NestedBruteForce_HPP

#include "SudokuBoard.hpp"
#include "SudokuSolver.hpp"

class SudokuSolver_NestedBruteForce : public SudokuSolver
{
public:
    SudokuSolver_NestedBruteForce(SudokuBoard &board, bool print_message = true);

    // Solves the given Sudoku board using sequential brute force algorithm
    virtual void solve() override { solve_kernel(0, 0); }
    void solve_kernel(int row, int col);
};

#endif // SudokuSolver_NestedBruteForce_HPP