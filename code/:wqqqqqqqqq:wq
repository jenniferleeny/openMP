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

void print(cost_t *array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)  
            printf("%d ", array[i*cols + j]);
        printf("\n");
    }
}

cost_t matrix_cost(cost_t *matrix, int num) {
    cost_t aggregate_cost = 0;
    for (int i = 0; i < num; i++) {
            if (matrix[i] > 1)
                aggregate_cost += matrix[i];
    }
    return aggregate_cost;
}

cost_t *change_wire_route_helper(cost_t* matrix, int x1, int y1, int x2, int y2,
        int dim_x, int dim_y, int increment) {
    if (y1 == y2) {
        for (int i = std::min(x1, x2); i < std::max(x1, x2); i++) {
            matrix[ y1 * dim_x + i] += increment;
        }
    } else {
        for (int i = std::min(y1, y2); i < std::max(y1, y2); i++) {
            matrix[i * dim_x + x1] += increment;
        }
    }
    return matrix;
}

cost_t *change_wire_route(cost_t *matrix, wire_t wire, int dim_x, int dim_y, 
        int increment) {
    // undo original wire routing
    // get rid of second line in route
    matrix = change_wire_route_helper(matrix, wire.x1, wire.y1, wire.bend_x1, 
            wire.bend_y1, dim_x, dim_y, increment);
    if (wire.bend_y2 >= 0 && wire.bend_x2 >= 0) {
        matrix = change_wire_route_helper(matrix, wire.bend_x1, wire.bend_y1,
                        wire.bend_x2, wire.bend_y2, dim_x, dim_y, increment);
        matrix = change_wire_route_helper(matrix, wire.bend_x2, wire.bend_y2,
                        wire.x2, wire.y2, dim_x, dim_y, increment);
    } else {
        matrix = change_wire_route_helper(matrix, wire.bend_x1, wire.bend_y1,
                        wire.x2, wire.y2, dim_x, dim_y, increment);
    }
    return matrix;
}

/*cost_t *add_wire_route(cost_t *matrix, wire_t wire, int bendx1, int bendy1, 
            int bendx2, int bendy2) {
    matrix = add_wire_route_helper(matrix, wire.x1, wire.y1, bendx1, bendy1);
    if (wire.bend_y2 >= 0 && wire.bend_x2 >= 0) {
        matrix = add_wire_route_helper(matrix, bendx1, bendy1, bendx2, bendy2);
        matrix = add_wire_route_helper(matrix, bendx1, bendy1, bendx2, bendy2);
    } else {
        matrix = add_wire_route_helper(matrix, bendx1, bendy1, wire.x2, wire.y2);
    }
    return matrix;
}*/

// find_mind_path_cost: takes in (wire.x1, wire.y1) to (wire.x2, wire.y2) and adds the 
// wire route to matrix
wire_t find_min_path(int length, int dim_x, int dim_y, wire_t wire, cost_t *matrix) { 
    print(matrix, dim_y, dim_x);
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
    cost_t curr_cost = 0;
    // if wire has been placed on grid or not
    if (wire.cost == -1) { // if wire.cost == -1 => wire has not been placed on grid
        for (int x = x1; x <= x2; x++) {
            curr_cost += 1 + matrix[std::min(y1, y2) * dim_x + x];
        }
        if (y1 <= y2) {
            for (int i = y1+1; i <= y2; i++) {
                curr_cost += 1 + matrix[i * dim_x + x2];
            }
        } else {
            for (int i = y2+1; i <= y1; i++) {
                curr_cost += 1 + matrix[i * dim_x + x1];
            } 
        }
        wire.cost = curr_cost;
        printf("wire cost: %d\n", wire.cost);
        wire.bend_x1 = x2;
        wire.bend_y1 = std::min(y1, y2);
    } else { // otws remove current route from matrix
        matrix = change_wire_route(matrix, wire, dim_x, dim_y, -1);
    }
    // MAIN IDEA: horizontal keeps track of the path cost of a wire route that has a 
    // bend at row x
    // vertical keeps track of the path cost of a wire route that has a bend at column y
    int *horizontal = (cost_t *)calloc(dim_y, sizeof(cost_t));
    int *vertical = (cost_t *)calloc(dim_x, sizeof(cost_t));
    // HORIZONTAL
    for (int i = 0; i < dim_y; i++) {
        cost_t row_cost = 0;
        for (int j = x1; j <= x2; j++) {
            row_cost += 1 + matrix[i*dim_x + j];
        }
        horizontal[i] = row_cost;
        printf("%d\n", row_cost);
    }
    for (int i = 0; i< dim_y; i++) {
        if (i < std::min(y1, y2)) {
            for (int j = i+1; j <= std::max(y1, y2); j++) {
                horizontal[i] += (j <= y1)* (1 + matrix[dim_x * j + x1]);
                horizontal[i] += (j <= y2)* (1 + matrix[dim_x * j + x2]);
            }// check for double counting
        } else if (i >= std::min(y1, y2) && i <= std::max(y1, y2)) {
            for (int j = std::min(y1, y2); j < i; j++) {
                horizontal[i] += (y1 < y2) * (1 + matrix[dim_x * j + x1]);
                horizontal[i] += (y2 < y1) * (1 + matrix[dim_x * j + x2]);
            }
            for (int j = i+1; j <= std::max(y1, y2); j++) {
                horizontal[i] += (y1 > y2) * (1 + matrix[dim_x * j + x1]);
                horizontal[i] += (y1 < y2) * (1 + matrix[dim_x * j + x2]);
            }
        } else if (i > std::max(y1, y2)) {
            for (int j = std::min(y1, y2); j < i; j++) {
                horizontal[i] += (j >= y1)* (1 + matrix[dim_x * j + x1]);
                horizontal[i] += (j >= y2)* (1 + matrix[dim_x * j + x2]);
            }
        }
    }
    print(horizontal, 1, dim_y);
    // checking vertical
    for (int i = 0; i < dim_x; i++) {
        cost_t row_cost = 0;
        for (int j = y1; j <= y2; j++) {
            row_cost += matrix[j*dim_x + i];
        }
        vertical[i] = row_cost;
    }
    for (int i = 0; i < dim_x; i++) {
        if (i < std::min(x1, x2)) {
            for (int j = i+1; j <= std::max(x1, x2); j++) {
                horizontal[i] += (j <= x1)* (1 + matrix[dim_x * y1 + j]);
                horizontal[i] += (j <= x2)* (1 + matrix[dim_x * y2 + j]);
            }// check for double counting
        } else if (i >= std::min(x1, x2) && i <= std::max(x1, x2)) {
            for (int j = std::min(x1, x2); j < i; j++) {
                horizontal[i] += (y1 < y2) * (1 + matrix[dim_x * y1 + j]);
                horizontal[i] += (y2 < y1) * (1 + matrix[dim_x * y2 + j]);
            }
            for (int j = i+1; j <= std::max(x1, x2); j++) {
                horizontal[i] += (y1 > y2) * (1 + matrix[dim_x * y1 + j]);
                horizontal[i] += (y1 < y2) * (1 + matrix[dim_x * y2 + j]);
            }
        } else if (i > std::max(x1, x2)) {
            for (int j = std::min(y1, y2); j < i; j++) {
                horizontal[i] += (j >= x1)* (1 + matrix[dim_x * y1 + j]);
                horizontal[i] += (j >= x2)* (1 + matrix[dim_x * y2 + j]);
            }
        }
    }
    // print(vertical, 1, dim_x);
    int min_val = wire.cost;
    int new_bendx1 = -1;
    int new_bendy1 = -1;
    int new_bendx2 = -1;
    int new_bendy2 = -1;
    // keep track of bends
    for (int i = 0; i <= dim_y; i++) {
        if (min_val > horizontal[i]) {
            min_val = horizontal[i];
            new_bendx1 = wire.x1;
            new_bendy1 = i;
            if (i != wire.y1 && i != wire.y2) {
                new_bendx2 = wire.x2;
                new_bendy2 = i;
            }
        }   
    }
    for (int i = 0; i < dim_x; i++) {
        if (min_val > vertical[i]) {
            min_val = vertical[i];
            new_bendx1 = i;
            new_bendy1 = wire.y1;
            if (i != wire.x1 && i != wire.x2) {
                new_bendx2 = i;
                new_bendy2 = wire.y2;
            }
        }
    }
    if ( wire.cost > min_val) {
        wire.bend_x1 = new_bendx1;
        wire.bend_y1 = new_bendy1;
        wire.bend_x2 = new_bendx2;
        wire.bend_y2 = new_bendy2;
        wire.cost = min_val;
        matrix = change_wire_route(matrix, wire, dim_x, dim_y, 1);
    }
    free(horizontal);
    free(vertical);
    return wire;
}

cost_t *wire_routing(cost_t *matrix, wire_t *wires, int dim_x, int dim_y, int num_wires) {
    // cost_t min_cost_path;
    int delta = 0;
    for (int i = 0; i < num_wires; i++) {
        int length = delta + abs(wires[i].x2 - wires[i].x1) + abs(wires[i].y2 - wires[i].y1);
        wires[i] = find_min_path(length, dim_x, dim_y, wires[i], matrix);
        printf("cost wire %d: %d\n", i, wires[i].cost);
    }
    return matrix;
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
        for (int i = 0; i < SA_iters; i++) {
            costs = wire_routing(costs, wires, dim_x, dim_y, num_of_wires); 
        }
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
    for (int i = 0; i < dim_y; i++) {
        for (int j = 0; j < dim_x; j++) {
            fprintf(output_costs_file, "%d ", (int)costs[i*dim_x + j]);
        }
        fprintf(output_costs_file, "\n");
    }
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

    for (int i = 0; i < num_of_wires; i++) {
        wire_t wire = wires[i];
        fprintf(output_routes_file, "%d %d %d %d %d %d %d %d\n", wire.x1, wire.y1,
             wire.bend_x1,wire.bend_y1, wire.bend_x2, wire.bend_y2, wire.x2, wire.y2);
    }  
    // WRITE WIRES TO FILE HERE

    fclose(output_routes_file);

    return 0;
}
