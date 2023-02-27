#include <cstring>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <iostream>
using namespace std;
constexpr uint32_t kAll = 0x1ff;
/*
extern "C" size_t Solver(const char *input, char *solution) {
  memcpy(solution, input, 81);
  return 0;
}*/
static void PrintSudoku(const char *board) {
	cout<<"pp"<<endl;
    for (size_t row = 0; row < 9; ++row) {
      for (size_t col = 0; col < 9; ++col) {
        //stream << board[row * 9 + col];
	cout << board[row * 9 + col];
      }
      cout << std::endl;
    }
  }

struct SolverImpl {
  std::array<uint32_t, 9> rows{}, cols{}, boxes{};
  std::vector<uint32_t> cells_todo;
  size_t num_todo = 0, num_solutions = 0;
  char final_solution [81]{0};

  void Solve(size_t todo_index, char *solution, std::array<uint32_t, 9> &r, std::array<uint32_t, 9> &c, std::array<uint32_t, 9> &b) {
    char s[81]{0};
    //cout << "printinggg"<<endl;
    memcpy(s, solution, 81);
    std::array<uint32_t, 9> nrows = r;
    //for(int f=0;f<9;f++){
    //	    cout<<nrows[f]<<","<<r[f]<<","<<rows[f]<<endl;
    //}
    std::array<uint32_t, 9> ncols = c;
    std::array<uint32_t, 9> nboxes = b;
    uint32_t row_col_box = cells_todo[todo_index];
    auto row = row_col_box & kAll;
    auto col = (row_col_box >> 9) & kAll;
    auto box = (row_col_box >> 18) & kAll;
    auto candidates = nrows[row] & ncols[col] & nboxes[box];
    //cout<<row<<endl;
    while (candidates) {
      uint32_t candidate = candidates & -candidates;
      nrows[row] ^= candidate;
      ncols[col] ^= candidate;
      nboxes[box] ^= candidate;
      s[row * 9 + col] = (char)('0' + __builtin_ffs(candidate));
      if (todo_index < num_todo) {
        #pragma omp task shared(num_solutions)
        {
            Solve(todo_index + 1, s,nrows,ncols,nboxes);
        }
      } else {
	if(num_solutions==0){
        ++num_solutions;
        memcpy(final_solution,s, 81);
	//PrintSudoku(final_solution);
	}
	//std::cout << "Welcome to GFG"<<endl;
      }
      if (num_solutions > 0) {
        return;
      }
      nrows[row] ^= candidate;
      ncols[col] ^= candidate;
      nboxes[box] ^= candidate;
      candidates = candidates & (candidates - 1);
    }
  }

  bool Initialize(const char *input, char *solution) {
    rows.fill(kAll);
    cols.fill(kAll);
    boxes.fill(kAll);
    cells_todo.clear();
    num_solutions = 0;
    memcpy(solution, input, 81);

    for (int row = 0; row < 9; ++row) {
      for (int col = 0; col < 9; ++col) {
        int box = int(row / 3) * 3 + int(col / 3);
        if (input[row * 9 + col] == '.') {
          cells_todo.emplace_back(row | (col << 9) | (box << 18));
        } else {
          uint32_t value = 1u << (uint32_t)(input[row * 9 + col] - '1');
          if (rows[row] & value && cols[col] & value && boxes[box] & value) {
            rows[row] ^= value;
            cols[col] ^= value;
            boxes[box] ^= value;
          } else {
            return false;
          }
        }
      }
    }
    num_todo = cells_todo.size() - 1;
    return true;
  }
};

extern "C" size_t Solver(const char *input, char *solution) {
  static SolverImpl solver;
  if (solver.Initialize(input, solution)) {

	  #pragma omp parallel
    {
        #pragma omp single
        {
		#pragma omp taskgroup
		{
            solver.Solve(0, solution,solver.rows,solver.cols,solver.boxes);
		}
        }
    }
    //memcpy(solution, solver.final_solution, 81);
    //PrintSudoku(solver.final_solution, std::cerr);
    solver.Solve(0, solution,solver.rows,solver.cols,solver.boxes);
    memcpy(solution, solver.final_solution, 81);
    //PrintSudoku(solution);
    return solver.num_solutions;
  }
  return 0;
}
