#include "utility.hpp"
#include <chrono>
#include <cstring>

// TODO: class Sudoku
// TODO: file reading
// TODO: add recursive depth
// TODO: solve 16 * 16

#define PRINT_TIME 1


bool solved;
int answer[N][N] = {0};


void solveSudoku_backtracking(int board[N][N])
{
	if (solved) return;

    if (checkIfAllFilled(board))   // base case
    {
        solved = true;
		std::memcpy(answer, board, SIZEOF_SUDOKU);
		return;
    }
    else
    {
        std::pair<int, int> empty_cell = find_empty(board);

        for (int num = 1; num <= N; num++)
        {
			int row = empty_cell.first;
			int col = empty_cell.second;

            if (isValid(board, num, empty_cell))
            {
                board[row][col] = num;
                solveSudoku_backtracking(board);
            }

			board[row][col] = 0;   // backtrack to the most recently filled cell
        }

        // None of the values solved the Sudoku
		solved = false;
        return;
    }
}


int main(void)
{
    // 0 means empty cells
    int board[N][N] = { { 3, 0, 6, 5, 0, 8, 4, 0, 0 },
                        { 5, 2, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 8, 7, 0, 0, 0, 0, 3, 1 },
                        { 0, 0, 3, 0, 1, 0, 0, 8, 0 },
                        { 9, 0, 0, 8, 6, 3, 0, 0, 5 },
                        { 0, 5, 0, 0, 9, 0, 6, 0, 0 },
                        { 1, 3, 0, 0, 0, 0, 2, 5, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 7, 4 },
                        { 0, 0, 5, 2, 0, 6, 3, 0, 0 } };
    // int board[N][N] =
    // { {0,1,2,0,0,4,0,0,0,0,5,0,0,0,0,0},
    //   {0,0,0,0,0,2,0,0,0,0,0,0,0,14,0,0},
    //   {0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0},
    //   {11,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0},
    //   {0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //   {0,0,0,16,0,0,0,0,0,0,2,0,0,0,0,0},
    //   {0,0,0,0,0,0,0,0,11,0,0,0,0,0,0,0},
    //   {0,0,14,0,0,0,0,0,0,4,0,0,0,0,0,0},
    //   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //   {0,0,0,0,0,1,16,0,0,0,0,0,0,0,0,0},
    //   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //   {0,0,0,0,0,0,0,0,0,0,14,0,0,13,0,0},
    //   {0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //   {0,0,11,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //   {16,0,0,0,0,0,11,0,0,3,0,0,0,0,0,0} };
 
    print_board(board);

	std::cout << "Sudoku solver starts, please wait..." << std::endl;

#if PRINT_TIME
    std::chrono::high_resolution_clock::time_point start, stop;
    start = std::chrono::high_resolution_clock::now();
#endif

	solveSudoku_backtracking(board);

#if PRINT_TIME
	int time_in_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << std::dec << "Operations executed in " << (double)time_in_microseconds / 1000000 << " seconds" << std::endl;
#endif

    print_board(board);

    // if (solveSudoku_backtracking(board)) {
    //     std::cout << "--------------------" << std::endl;
    //     print_board(board);
    // } else {
    //     std::cout << "No solution exists." << std::endl;
    // }

	std::cout << "Press any key to exit...";
    std::getchar();
    return 0;
}