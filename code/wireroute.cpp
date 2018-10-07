/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include <cmath>
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

cost_t matrix_cost(cost_t *matrix) {
    cost_t aggregate_cost = 0;
    for (unsigned int i = 0; i < sizeof(matrix) / sizeof(matrix[0]); i++) {
            if (matrix[i] > 1)
                aggregate_cost += matrix[i];
    }
    return aggregate_cost;
}

// find_mind_path_cost: takes in (wire.x1, wire.y1) to (wire.x2, wire.y2) and adds the 
// wire route to matrix
void find_min_path(int length, int dim_x, int dim_y, wire_t wire, cost_t *matrix) { 
    int x1, x2, y1, y2;
    if (wire.x1 <= wire.x2) {
        x1 = wire.x1;
        y1 = wire.y1;
        x2 = wire.x2;
        y2 = wire.y2;
    } else {
        x1 = wire.x2;
        y1 = wire.y2;
        x2 = wire.x1;
        y2 = wire.y1;
    }
    // check if wire has been placed on grid
    cost_t curr_cost = 0;
    int offset = dim_x * std::min(y1, y2);
    // int index = dim_x * y1 + x1;
    if (wire.cost == -1) { // if wire.cost == -1 => wire has not been placed on grid
        for (int i = x1; i <= x2; i++) {
            matrix[offset + i] += 1;
        }
        for (int i = std::max(y1, y2); i >= std::min(y1, y2); i--) {
            if (y1 <= y2) {
                matrix[i * dim_x + x2] += 1;
                curr_cost += matrix[i * dim_x + x2];
            } else {
                matrix[i * dim_x + x1] += 1;
                curr_cost += matrix[i * dim_x + x1];
            }
        }
        wire.cost = curr_cost;
    }
    // MAIN IDEA: horizontal keeps track of the path cost of a wire route that has a 
    // bend at row x
    // vertical keeps track of the path cost of a wire route that has a bend at column y
    int *horizontal = (cost_t *)calloc(dim_y, sizeof(cost_t));
    int *vertical = (cost_t *)calloc(dim_x, sizeof(cost_t));
    // checking horizonal first
    for (int i = 0; i < dim_y; i++) {
        cost_t row_cost = 0;
        for (int j = x1; j <= x2; j++) {
            row_cost += matrix[i*dim_x + j];
        horizontal[i] = row_cost;
    }
    for (int i = 0; i< dim_y; i++) {
        if (i < std::min(y1, y2)) {
            for (int j = i+1; j <= std::max(y1, y2); j++) {
                horizontal[i] += (j <= y1)* matrix[dim_x * j + x1];
                horizontal[i] += (j <= y2)* matrix[dim_x * j + x2];
            }// check for double counting
        } else if (i >= std::min(y1, y2) && i <= std::max(y1, y2)) {
            for (int j = std::min(y1, y2); j < i; j++) {
                horizontal[i] += (y1 < y2) * matrix[dim_x * j + x1];
                horizontal[i] += (y2 < y1) * matrix[dim_x * j + x2];
            }
            for (int j = i+1; j <= std::max(y1, y2); j++) {
                horizontal[i] += (y1 > y2) * matrix[dim_x * j + x1];
                horizontal[i] += (y1 < y2) * matrix[dim_x * j + x2];
            }
        } else if (i > std::max(y1, y2)) {
            for (int j = std::min(y1, y2); j < i; j++) {
                horizontal[i] += (j >= y1)* matrix[dim_x * j + x1];
                horizontal[i] += (j >= y2)* matrix[dim_x * j + x2];
            }
        }
    }
    // checking vertical
    for (int i = 0; i < dim_x; i++) {
        cost_t row_cost = 0;
        for (int j = y1; j <= y2; j++) {
            row_cost += matrix[j*dim_x + i];
        vertical[i] = row_cost;
    }
    for (int i = 0; i< dim_x; i++) {
        if (i < std::min(x1, x2)) {
            for (int j = i+1; j <= std::max(x1, x2); j++) {
                horizontal[i] += (j <= x1)* matrix[dim_x * y1 + j];
                horizontal[i] += (j <= x2)* matrix[dim_x * y2 + j];
            }// check for double counting
        } else if (i >= std::min(x1, x2) && i <= std::max(x1, x2)) {
            for (int j = std::min(x1, x2); j < i; j++) {
                horizontal[i] += (y1 < y2) * matrix[dim_x * y1 + j];
                horizontal[i] += (y2 < y1) * matrix[dim_x * y2 + j];
            }
            for (int j = i+1; j <= std::max(x1, x2); j++) {
                horizontal[i] += (y1 > y2) * matrix[dim_x * y1 + j];
                horizontal[i] += (y1 < y2) * matrix[dim_x * y2 + j];
            }
        } else if (i > std::max(x1, x2)) {
            for (int j = std::min(y1, y2); j < i; j++) {
                horizontal[i] += (j >= x1)* matrix[dim_x * y1 + j];
                horizontal[i] += (j >= x2)* matrix[dim_x * y2 + j];
            }
        }
    // finding min
    }

    curr_cost = 0; 
    for (int x = x2; x >= x1; x--) {
        curr_cost += matrix[y2 * dim_x + x];
        horizontal[x - x1] += curr_cost;
    }
    // VERTICAL
    // 
    if (wire.y1 <= wire.y2) {
        x1 = wire.x1;
        y1 = wire.y1;
        x2 = wire.x2;
        y2 = wire.y2;
    } else {
        x1 = wire.x2;
        y1 = wire.y2;
        x2 = wire.x1;
        y2 = wire.y1;
    }
    vertical[0] = matrix[dim_x * y1 + x1];
    for (int y = y1+ 1; y <= y2; y++) {
        vertical[y - y1] = vertical[y-y1-1] + matrix[dim_x * y + x1];
    }
    for (int y = y1; y <= y2; y++) {
        for (int x = std::min(x1, x2); x <= std::max(x1, x2); x++) {
            vertical[y - y1] += matrix[dim_x * y + x];
        }
    }
    curr_cost = 0; 
    for (int x = x2; x >= x1; x--) {
        curr_cost += matrix[y2 * dim_x + x];
        horizontal[x - x1] += curr_cost;
    }

    int min_val = horizontal[0];

    // keep track of bends
    for (int i = 0; i <= abs(x2 - x1); i++) {
        min_val = std::min(min_val, horizontal[i]);
    }
    for (int i = 0; i < abs(y2 - y1); i++) {
        min_val = std::min(min_val, vertical[i]);
    }
    wire.cost = min_val;
    wire.bend_x1 = 0;
    wire.bend_x2 = 
    free(horizontal);
    free(vertical);
}

cost_t *wire_routing(cost_t *matrix, wire_t *wires, int dim_x, int dim_y) {
    // cost_t min_cost_path;
    int delta = 0;
    for (unsigned int i = 0; i < sizeof(wires) / sizeof(wires[0]); i++) {
        int length = delta + abs(wires[i].x2 - wires[i].x1) + abs(wires[i].y2 - wires[i].y1);
        find_min_path(length, dim_x, dim_y, wires[i], matrix);
    }
    return NULL;
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
        wire_t wire;
        wire.x1 = x1;
        wire.x2 = x2;
        wire.y1 = y1;
        wire.y2 = y2;
        wire.bend_x1 = -1;
        wire.bend_y1 = -1;
        wire.bend_x2 = -1;
        wire.bend_y2 = -1;
        wire.cost = -1;
        wires[index] = wire;
        index++;
    }

    if (index != num_of_wires) {
        printf("Error: wire count mismatch");
        return -1;
    }

    cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    (void)costs;
    /* INITIALIZE YOUR COST MATRIX HERE */
	for (int i = 0; i < dim_y*dim_x; i++) {
			costs[i] = 0;
	} 
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
        printf("HELLO\n");
        costs = wire_routing(costs, wires, dim_x, dim_y); 
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
    fprintf(output_routes_file, "%d\n", delta);
    fprintf(output_routes_file, "%d\n", num_of_wires);
  
    // WRITE WIRES TO FILE HERE

    fclose(output_routes_file);

    return 0;
}
