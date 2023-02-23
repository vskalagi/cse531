#include <cstring>
#include <omp.h>

extern "C" size_t Solver(const char *input, char *solution) {
  memcpy(solution, input, 81);
  return 0;
}
