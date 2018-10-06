/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include "mic.h"

#define BUFSIZE 1024

static int _argc;
static const char **_argv;

/* Starter code function, don't touch */
const char *get_option_string(const char *option_name,
			      const char *default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return _argv[i + 1];
  return default_value;
}

/* Starter code function, do not touch */
int get_option_int(const char *option_name, int default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return atoi(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
float get_option_float(const char *option_name, float default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return (float)atof(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
static void show_help(const char *program_path)
{
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-p <SA_prob>\n");
    printf("\t-i <SA_iters>\n");
}


int main(int argc, const char *argv[])
{
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  auto init_start = Clock::now();
  double init_time = 0;
 
  _argc = argc - 1;
  _argv = argv + 1;

  /* You'll want to use these parameters in your algorithm */
  const char *input_filename = get_option_string("-f", NULL);
  int num_of_threads = get_option_int("-n", 1);
  double SA_prob = get_option_float("-p", 0.1f);
  int SA_iters = get_option_int("-i", 5);

  int error = 0;

  if (input_filename == NULL) {
    printf("Error: You need to specify -f.\n");
    error = 1;
  }

  if (error) {
    show_help(argv[0]);
    return 1;
  }
  
  printf("Number of threads: %d\n", num_of_threads);
  printf("Probability parameter for simulated annealing: %lf.\n", SA_prob);
  printf("Number of simulated anneling iterations: %d\n", SA_iters);
  printf("Input file: %s\n", input_filename);

  FILE *input = fopen(input_filename, "r");

  if (!input) {
    printf("Unable to open file: %s.\n", input_filename);
    return -1;
  }
 
  int dim_x, dim_y;
  int delta;
  int num_of_wires;

  fscanf(input, "%d %d\n", &dim_x, &dim_y);
  fscanf(input, "%d\n", &delta);
  assert(delta >= 0 && delta%2 == 0);
  fscanf(input, "%d\n", &num_of_wires);

  wire_t *wires = (wire_t *)calloc(num_of_wires, sizeof(wire_t));
  (void)wires;
 
  int x1, y1, x2, y2;
  int index = 0;
  while (fscanf(input, "%d %d %d %d\n", &x1, &y1, &x2, &y2) != EOF) {
    /* PARSE THE INPUT FILE HERE.
     * Define wire_t in wireroute.h and store 
     * x1, x2, y1, and y2 into the wires array allocated above
     * based on your wire_t definition. */
    index++;
  }

  if (index != num_of_wires) {
    printf("Error: wire count mismatch");
    return -1;
  }

  cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
  (void)costs;
  /* INITIALIZE YOUR COST MATRIX HERE */

  
  /* Initialize additional data structures needed in the algorithm 
   * here if you feel it's needed. */

  error = 0;

  init_time += duration_cast<dsec>(Clock::now() - init_start).count();
  printf("Initialization Time: %lf.\n", init_time);

  auto compute_start = Clock::now();
  double compute_time = 0;
#ifdef RUN_MIC /* Use RUN_MIC to distinguish between the target of compilation */

  /* This pragma means we want the code in the following block be executed in 
   * Xeon Phi.
   */
#pragma offload target(mic) \
  inout(wires: length(num_of_wires) INOUT)    \
  inout(costs: length(dim_x*dim_y) INOUT)
#endif
  {
    /* Implement the wire routing algorithm here
     * Feel free to structure the algorithm into different functions
     * Don't use global variables.
     * Use OpenMP to parallelize the algorithm. 
     * You should really implement as much of this (if not all of it) in
     * helper functions. */
  }

  compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
  printf("Computation Time: %lf.\n", compute_time);

  /* OUTPUT YOUR RESULTS TO FILES HERE 
   * When you're ready to output your data to files, uncommment this chunk of
   * code and fill in the specified blanks indicated by comments. More about
   * this in the README. */
  char input_filename_cpy[BUFSIZE];
  strcpy(input_filename_cpy, input_filename);
  char *filename = basename(input_filename_cpy);
  char output_filename[BUFSIZE];


  sprintf(output_filename, "costs_%s_%d.txt", filename, num_of_threads);
  FILE *output_costs_file = fopen(output_filename, "w");
  if (!output_costs_file) {
    printf("Error: couldn't output costs file");
    return -1;
  }

  fprintf(output_costs_file, "%d %d\n", dim_x, dim_y);
  
  // WRITE COSTS TO FILE HERE 

  fclose(output_costs_file);


  sprintf(output_filename, "output_%s_%d.txt", filename, num_of_threads);
  FILE *output_routes_file = fopen(output_filename, "w");
  if (!output_routes_file) {
    printf("Error: couldn't output routes file");
    return -1;
  }

  fprintf(output_routes_file, "%d %d\n", dim_x, dim_y);
  fprintf(output_routes_file, "%d\n", num_of_wires);
  
  // WRITE WIRES TO FILE HERE

  fclose(output_routes_file);

  return 0;
}
