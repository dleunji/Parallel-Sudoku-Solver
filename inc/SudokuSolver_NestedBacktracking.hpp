#ifndef SudokuSolver_NestedBacktracking_HPP
#define SudokuSolver_NestedBacktracking_HPP

#include "SudokuBoard.hpp"
#include "SudokuSolver.hpp"

class SudokuSolver_NestedBacktracking : public SudokuSolver
{
public:
    SudokuSolver_NestedBacktracking(SudokuBoard &board, bool print_message = true);

    // Solves the given Sudoku board using sequential backtracking algorithm
    virtual void solve() { solve_kernel(); }
    bool solve_kernel();
};

#endif // SudokuSolver_NestedBacktracking_HPP