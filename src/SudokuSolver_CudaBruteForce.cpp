#include "SudokuSolver_CudaBruteForce.hpp"
#include "SudokuSolver_SequentialBruteForce.hpp"
#include "termcolor.hpp"
#include <iostream>
#include <vector>
#include <omp.h>

SudokuSolver_CudaBruteForce::SudokuSolver_CudaBruteForce(SudokuBoard &board, bool print_message /*=true*/)
    : SudokuSolver(board)
{
    _mode = MODES::CUDA_BRUTEFORCE;
    if (print_message)
    {
        std::cout << "\n"
                  << "Cuda Sudoku solver using brute force algorithm starts, please wait..."
                  << "\n";
    }
}

void SudokuSolver_CudaBruteForce::bootstrap()
{
    // if no start boards in the board deque, then return
    if (_board_deque.size() == 0)
    {
        return;
    }

    SudokuBoard board = _board_deque.front();

    if (checkIfAllFilled(board))
    {
        return;
    }

    Position empty_cell_pos = find_empty(board);

    int row = empty_cell_pos.first;
    int col = empty_cell_pos.second;

    // fill in all possible numbers to the empty cell and then
    // add the corresponding possible board of solution to the end of board deque
    for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
    {
        if (isValid(board, num, empty_cell_pos))
        {
            board.set_board_data(row, col, num);
            _board_deque.push_back(board);
        }
    }

    _board_deque.pop_front();
}

void SudokuSolver_CudaBruteForce::bootstrap_openmp()
{
    // printf("thread num: %d\n", omp_get_thread_num());
    // if no start boards in the board deque, then return
    if (_board_deque.size() == 0)
    {
        return;
    }

    SudokuBoard board;
    bool valid = false;
#pragma omp critical
    {
        if (_board_deque.size() > 0)
        {
            valid = true;
            board = _board_deque.front();
            _board_deque.pop_front();
        }
    }

    if (!valid)
        return;

    // printf("thread num: %d, valid?\n", omp_get_thread_num());
    if (checkIfAllFilled(board))
    {
        return;
    }
    // printf("thread num: %d, valid!\n", omp_get_thread_num());

    Position empty_cell_pos = find_empty(board);

    int row = empty_cell_pos.first;
    int col = empty_cell_pos.second;

    // fill in all possible numbers to the empty cell and then
    // add the corresponding possible board of solution to the end of board deque
    for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
    {
        if (isValid(board, num, empty_cell_pos))
        {
            board.set_board_data(row, col, num);
#pragma omp critical
            {
                _board_deque.push_back(board);
            }
        }
    }
}

void SudokuSolver_CudaBruteForce::bootstrap(SudokuBoardDeque &boardDeque, int indexOfRows)
{
    // if no start boards in the board deque, then return
    if (boardDeque.size() == 0)
    {
        return;
    }

    while (!checkIfRowFilled(boardDeque.front(), indexOfRows))
    {
        SudokuBoard board = boardDeque.front();

        int empty_cell_col_index = find_empty_from_row(board, indexOfRows);

        // fill in all possible numbers to the empty cell and then
        // add the corresponding possible board of solution to the end of board deque
        for (int num = board.get_min_value(); num <= board.get_max_value(); ++num)
        {
            Position empty_cell_pos = std::make_pair(indexOfRows, empty_cell_col_index);

            if (isValid(board, num, empty_cell_pos))
            {
                board.set_board_data(indexOfRows, empty_cell_col_index, num);
                boardDeque.push_back(board);
            }
        }

        boardDeque.pop_front();
    }
}

void SudokuSolver_CudaBruteForce::solve_kernel_1()
{
    // push the board onto the board deque as the first element
    _board_deque.push_back(_board);

    // ensure some level of bootstrapping
    int N = _board.get_board_size();
    int num_bootstraps = get_num_threads() * 10;
    // #pragma omp parallel for schedule(static) default(none) shared(num_bootstraps)
    for (int i = 0; i < num_bootstraps; ++i)
    {
        bootstrap();
    }

    int numberOfBoards = _board_deque.size();

    _boards = (int *)malloc(numberOfBoards * N * N * sizeof(int));
    // For debugging
    // std::cout << "Number of Suodku boards on the board deque: " << numberOfBoards << "\n";
    // for (int i = 0; i < numberOfBoards; ++i)
    // {
    //     std::cout << "BOARD-" << i << "\n";
    //     print_board(_board_deque[i]);
    //     std::cout << "*****"
    //               << "\n";
    // }

#pragma omp parallel for schedule(static) default(none) shared(numberOfBoards, _boards, N)
    for (int i = 0; i < numberOfBoards; ++i)
    {
        int tid = omp_get_thread_num();
        std::vector<std::vector<int>> board_data = _board_deque[i].get_board_data();
        for (int j = 0; j < N; j++)
        {
            std::vector<int> row_vals = board_data[j];
            std::copy(row_vals.begin(), row_vals.end(), _boards + (i * N * N + j * N));
        }
    }

    _solved = call_backtrack(_boards, numberOfBoards, _board.get_box_size(), _board.get_board_size());
    if (_solved)
    {
        int *solved_sudoku = _boards;
        _solution = SudokuBoard(_board.get_box_size(), _board.get_board_size(), solved_sudoku);
        for (int r = 0; r < N; r++)
        {
            for (int c = 0; c < N; c++)
            {
                _solution.set_board_data(r, c, solved_sudoku[r * N + c]);
            }
        }
    }
    free(_boards);
}
