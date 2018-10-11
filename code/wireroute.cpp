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
    printf("\n");
}

cost_t matrix_max_overlap(cost_t *matrix, int num) {
    cost_t max_cost = 0;
    for (int i = 0; i < num; i++) {
        max_cost = std::max(max_cost, matrix[i]);
    }
    return max_cost;
}

cost_t matrix_agg_overlap(cost_t *matrix, int num) {
    cost_t max_cost = 0;
    for (int i = 0; i < num; i++) {
        if (matrix[i] > 1)
            max_cost += matrix[i];
    }
    return max_cost;
}


cost_t find_max_overlap_h(cost_t *matrix, int i, int x1, int y1, int x2, int y2, 
                    int dim_x, int dim_y) {
    cost_t max_cost = 0;
    for (int j = x1; j <= x2; j++) {
        max_cost = std::max(max_cost, 1 + matrix[i*dim_x + j]);
    }
    if (i < std::min(y1, y2)) {
        for (int j = i+1; j <= std::max(y1, y2); j++) {
            max_cost = std::max(max_cost, (j <= y1) * (1+matrix[dim_x*j+x1]));
            max_cost = std::max(max_cost, (j <= y2) * (1+matrix[dim_x*j+x2]));
        }// check for double counting
    } else if (i >= std::min(y1, y2) && i <= std::max(y1, y2)) {
        for (int j = std::min(y1, y2); j < i; j++) {
            max_cost = std::max(max_cost, (y1 <= y2) * (1+matrix[dim_x*j+x1]));
            max_cost = std::max(max_cost, (y2 < y1) * (1+matrix[dim_x*j+x2]));
        }
        for (int j = i+1; j <= std::max(y1, y2); j++) {
            max_cost = std::max(max_cost, (y1 >= y2) * (1+matrix[dim_x*j+x1]));
            max_cost = std::max(max_cost, (y1 < y2) * (1+matrix[dim_x*j+x2]));
        }
    } else if (i > std::max(y1, y2)) {
        for (int j = std::min(y1, y2); j < i; j++) {
            max_cost = std::max(max_cost, (j >= y1)* (1 + matrix[dim_x * j + x1]));
            max_cost = std::max(max_cost, (j >= y2)* (1 + matrix[dim_x * j + x2]));
        }
    }
    return max_cost;
}

cost_t find_max_overlap_v(cost_t *matrix, int i, int x1, int y1, int x2, int y2,
                        int dim_x, int dim_y) {// i indicates ith vertical segment
    cost_t max_cost = 0;
    for (int j = std::min(y1, y2); j <= std::max(y1, y2); j++) {
        max_cost = std::max(max_cost, 1 + matrix[j*dim_x + i]);
    }
        if (i < x1) {
        for (int j = i+1; j <= x2; j++) {
            max_cost = std::max(max_cost, (j <= x1)* (1 + matrix[dim_x * y1 + j]));
            max_cost = std::max(max_cost, (j <= x2)* (1 + matrix[dim_x * y2 + j]));
        }// check for double counting
    } else if (i >= x1 && i <= x2) {
        for (int j = x1; j < i; j++) {
            max_cost = std::max(max_cost, (y1 <= y2) * (1 + matrix[dim_x * y1 + j]));
            max_cost = std::max(max_cost, (y2 < y1) * (1 + matrix[dim_x * y2 + j]));
        }
        for (int j = i+1; j <= x2; j++) {
            max_cost = std::max(max_cost, (y1 >= y2) * (1 + matrix[dim_x * y1 + j]));
            max_cost = std::max(max_cost, (y1 < y2) * (1 + matrix[dim_x * y2 + j]));
        }
    } else if (i > x2) {
        for (int j = x1; j < i; j++) {
            max_cost = std::max(max_cost, (j >= x1)* (1 + matrix[dim_x * y1 + j]));
            max_cost = std::max(max_cost, (j >= x2)* (1 + matrix[dim_x * y2 + j]));
        }
    }
    return max_cost;
}


cost_t find_wire_cost_helper(cost_t *matrix, int x1, int y1, int x2, int y2,
       int dim_x, int dim_y) {
    cost_t cost = 0;
    if (y1 == y2) {
        for (int i = std::min(x1, x2); i < std::max(x1, x2); i++) {
            cost += matrix[y1 * dim_x + i];
        }
    } else {
        for (int i = std::min(y1, y2); i < std::max(y1, y2); i++) {
            cost += matrix[i * dim_x + x1];
        }
    }
    return cost;
} 

cost_t find_wire_cost(cost_t *matrix, wire_t wire, int dim_x, int dim_y) {
    cost_t cost = find_wire_cost_helper(matrix, wire.x1, wire.y1, 
                        wire.bend_x1, wire.bend_y1, dim_x, dim_y);
    if (wire.bend_y2 >= 0 && wire.bend_x2 >= 0) {
        cost += find_wire_cost_helper(matrix, wire.bend_x1, wire.bend_y1,
                        wire.bend_x2, wire.bend_y2, dim_x, dim_y);
        cost += find_wire_cost_helper(matrix, wire.bend_x2, wire.bend_y2,
                        wire.x2, wire.y2, dim_x, dim_y);
    } else {
        cost += find_wire_cost_helper(matrix, wire.bend_x1, wire.bend_y1,
                        wire.x2, wire.y2, dim_x, dim_y);
    }
    cost += matrix[wire.y2 * dim_x + wire.x2]; 
    return cost;
}

void change_wire_route_helper(cost_t* matrix, int x1, int y1, int x2, int y2,
        int dim_x, int dim_y, int increment) {
    // printf("change wire route\n ");
    // printf("%d %d %d %d\n", x1, y1, x2, y2);
    if (x1 == x2 && y1 == y2)
        return; 
    if (y1 == y2) {
        if (x1 <= x2) {
            for (int i = x1; i < x2; i++) {
                matrix[ y1 * dim_x + i] += increment;
            }
        } else {
            // printf("KMS %d %d\n", x2, x1);
            for (int i = x1; i > x2; i-=1) {
                // printf("wtf %d\n", i);
                matrix[y1 * dim_x + i] += increment;
            }
        }
    } else {
        if (y1 <= y2) {
            for (int i = y1; i < y2; i++) {
                matrix[i * dim_x + x1] += increment;
            }
        } else {
            // printf("%d %d %d\n", __LINE__, y2, y1);
            for (int i = y1; i > y2; i--) {
                // printf("%d\n", __LINE__);
                matrix[i * dim_x + x1] += increment;
            }
            // printf("%d\n", __LINE__);

        }
    }
}

void change_wire_route(cost_t *matrix, wire_t wire, int dim_x, int dim_y, 
        int increment) {
    // printf("%d %d %d %d %d %d %d %d\n", wire.x1, wire.y1, wire.bend_x1,
     //       wire.bend_y1, wire.bend_x2, wire.bend_y2, wire.x2, wire.y2);
    if (wire.bend_x1 == -1 && wire.bend_y1 == -1) {
        change_wire_route_helper(matrix, wire.x1, wire.y1, wire.x2, 
            wire.y2, dim_x, dim_y, increment);
        matrix[wire.y2 * dim_x + wire.x2] += increment; 
        return;
    }
    change_wire_route_helper(matrix, wire.x1, wire.y1, wire.bend_x1, wire.bend_y1,
                                dim_x, dim_y, increment);
    if (wire.bend_y2 >= 0 && wire.bend_x2 >= 0) {
        change_wire_route_helper(matrix, wire.bend_x1, wire.bend_y1,
                        wire.bend_x2, wire.bend_y2, dim_x, dim_y, increment);
        change_wire_route_helper(matrix, wire.bend_x2, wire.bend_y2,
                        wire.x2, wire.y2, dim_x, dim_y, increment);
    } else {
        change_wire_route_helper(matrix, wire.bend_x1, wire.bend_y1,
                        wire.x2, wire.y2, dim_x, dim_y, increment);
    }
    matrix[wire.y2 * dim_x + wire.x2] += increment; 
}

void create_vert_horiz(int row, cost_t *matrix, wire_t wire, cost_t *horizontal,
                        cost_t *vertical, int dim_x, int dim_y, int delta) {
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
    int x_max = std::min(dim_x-1, (int)(delta/2) + x2);
    
    int x_min = std::max(0, x1 - (int)(delta/2));
    int y_max = std::min(dim_y-1, (int)(delta/2) + std::max(y1, y2));
    int y_min = std::max(0, std::min(y1, y2) - (int)(delta/2));
    if (wire.y1 == wire.y2) {
        x_min = x1;
        x_max = x2;
    }   
    if (wire.x1 == wire.x2) {
        y_min = y1;
        y_max = y2;
    }
    int vert_lock, horiz_lock;
   // printf("%d %d %d %d\n", x_min, x_max, y_min, y_max);
    int i; 
    for (i = x_min; i <= x_max; i++) {
        // printf("%d %d\n", __LINE__, i);
        if (row >= y_min && row <= y_max ) {
            horizontal[row] += (1 <= matrix[dim_x * row + i]);
        } if (row >= std::min(wire.y1, wire.y2)  && row <= std::max(wire.y1, wire.y2)) {
            vertical[i] += (1 <= matrix[dim_x * row + i]);
        }
    } 
} 

/*cost_t populate_horizontal(cost_t *matrix, int i, int x1, int y1, int x2, int y2,
                        int dim_x, int dim_y) {// i indicates ith horizontal segment
    cost_t row_cost = 0;
    for (int j = x1; j <= x2; j++) {
        row_cost += (matrix[i*dim_x + j] >= 1);
    }
    if (i < std::min(y1, y2)) {
        for (int j = i+1; j <= std::max(y1, y2); j++) {
            row_cost += (j <= y1) * (1 <= matrix[dim_x * j + x1]);
            row_cost += (j <= y2) * (1 <= matrix[dim_x * j + x2]);
        }// check for double counting
    } else if (i >= std::min(y1, y2) && i <= std::max(y1, y2)) {
        for (int j = std::min(y1, y2); j < i; j++) {
            row_cost += (y1 <= y2) * (1 <= matrix[dim_x * j + x1]);
            row_cost += (y2 < y1) * (1 <= matrix[dim_x * j + x2]);
        }
        for (int j = i+1; j <= std::max(y1, y2); j++) {
            row_cost += (y1 >= y2) * (1 <= matrix[dim_x * j + x1]);
            row_cost += (y1 < y2) * (1 <= matrix[dim_x * j + x2]);
        }
    } else if (i > std::max(y1, y2)) {
        for (int j = std::min(y1, y2); j < i; j++) {
            row_cost += (j >= y1)* (1 <= matrix[dim_x * j + x1]);
            row_cost += (j >= y2)* (1 <= matrix[dim_x * j + x2]);
        }
    }
    return row_cost;
}

cost_t populate_vertical(cost_t *matrix, int i, int x1, int y1, int x2, int y2,
                        int dim_x, int dim_y) {// i indicates ith vertical segment
    cost_t col_cost = 0;
    for (int j = std::min(y1, y2); j <= std::max(y1, y2); j++) {
        col_cost += (1 <= matrix[j*dim_x + i]);
    }
    if (i < x1) {
        for (int j = i+1; j <= x2; j++) {
            col_cost += (j <= x1)* (1 <= matrix[dim_x * y1 + j]);
            col_cost  += (j <= x2)* (1 <= matrix[dim_x * y2 + j]);
        }// check for double counting
    } else if (i >= x1 && i <= x2) {
        for (int j = x1; j < i; j++) {
            col_cost += (y1 <= y2) * (1 <= matrix[dim_x * y1 + j]);
            col_cost += (y2 < y1) * (1 <= matrix[dim_x * y2 + j]);
        }
        for (int j = i+1; j <= x2; j++) {
            col_cost += (y1 >= y2) * (1 <= matrix[dim_x * y1 + j]);
            col_cost += (y1 < y2) * (1 <= matrix[dim_x * y2 + j]);
        }
    } else if (i > x2) {
        for (int j = x1; j < i; j++) {
            col_cost += (j >= x1)* (1 <= matrix[dim_x * y1 + j]);
            col_cost += (j >= x2)* (1 <= matrix[dim_x * y2 + j]);
        }
    }
    return col_cost;
}*/

void anneal(wire_t &wire, cost_t *matrix, int dim_x, int dim_y) {
    // anneal
    if (wire.cost != -1)
        change_wire_route(matrix, wire, dim_x, dim_y, -1);
    int dx = abs(wire.x2 - wire.x1) + 1;
    int dy = abs(wire.y2 - wire.y1) + 1;
    int random_index = rand() % (dx + dy);
    if (random_index <= dx) {
        // this will be a 'vertical' path
        if (random_index == 0) {
            wire.bend_x1 = wire.x1;
            wire.bend_y1 = wire.y2;
            wire.bend_x2 = -1;
            wire.bend_y2 = -1;
        }
        else {
            int dir = (wire.x2 - wire.x1) / dx;
            wire.bend_x1 = wire.x1 + dir * (random_index);
            wire.bend_y1 = wire.y1;
            wire.bend_x2 = wire.bend_x1 == wire.x2 ? -1 : wire.bend_x1;
            wire.bend_y2 = wire.bend_x1 == wire.x2 ? -1 : wire.y2;
        }
    }
    else {
        // this will be a 'horizontal' path
        random_index -= dx;
        if (random_index == 0) {
            wire.bend_x1 = wire.x2;
            wire.bend_y1 = wire.y1;
            wire.bend_x2 = -1;
            wire.bend_y2 = -1;
        }
        else {
            int dir = (wire.y2 - wire.y1) / dy;
            wire.bend_x1 = wire.x1;
            wire.bend_y1 = wire.y1 + dir * (random_index + 1);
            wire.bend_x2 = wire.bend_y1 == wire.y2 ? -1 : wire.x2;
            wire.bend_y2 = wire.bend_y1 == wire.y2 ? -1 : wire.bend_y1;
        }
    }
    wire.cost = 2;
    change_wire_route(matrix, wire, dim_x, dim_y, 1);
}



// find_mind_path_cost: takes in (wire.x1, wire.y1) to (wire.x2, wire.y2) and adds the 
// wire route to matrix
void find_min_path(int delta, int dim_x, int dim_y, wire_t &wire, 
                   cost_t *matrix, double anneal_prob,
                   int *horizontal, int *vertical, int *max_overlap_horiz, int *max_overlap_vert) {
    
    double prob_sample = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    // printf("x1 y1: %d %d, x2 y2: %d %d\n", wire.x1, wire.y1, wire.x2, wire.y2);
    if (prob_sample < anneal_prob) {
        anneal(wire, matrix, dim_x, dim_y);
        return;
    }
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
    int x_max = std::min(dim_x-1, (int)(delta/2) + x2);
    int x_min = std::max(0, x1 - (int)(delta/2));
    int y_max = std::min(dim_y-1, (int)(delta/2) + std::max(y1, y2));
    int y_min = std::max(0, std::min(y1, y2) - (int)(delta/2));
    if (wire.y1 == wire.y2) {
        x_min = x1;
        x_max = x2;
    }   
    if (wire.x1 == wire.x2) {
        y_min = y1;
        y_max = y2;
    }
    if (wire.cost != -1) { 
        wire.cost = find_wire_cost(matrix, wire, dim_x, dim_y);
        change_wire_route(matrix, wire, dim_x, dim_y, -1);
    }
    // printf("removed wire\n");
    // print(matrix, dim_y, dim_x);
    /* int *horizontal = (cost_t *)calloc(y_max-y_min+1, sizeof(cost_t));
    int *vertical = (cost_t *)calloc(x_max-x_min+1, sizeof(cost_t));
    int *max_overlap_horiz = (cost_t *)calloc(y_max-y_min+1, sizeof(cost_t));
    int *max_overlap_vert = (cost_t *)calloc(x_max-x_min+1, sizeof(cost_t)); */

    memset(horizontal, 0, sizeof(cost_t) * dim_y);
    memset(vertical, 0, sizeof(cost_t) * dim_x);
    memset(max_overlap_horiz, 0, sizeof(cost_t) * dim_y);
    memset(max_overlap_vert, 0, sizeof(cost_t) * dim_x);
    int i;

    // HORIZONTAL
#pragma omp parallel for default(shared) private(i) schedule(dynamic)
    for (i = y_min; i <= y_max; i++) {
        //horizontal[i] = populate_horizontal(matrix, i, x1, y1, 
        //                                            x2, y2, dim_x, dim_y);
        create_vert_horiz(i, matrix, wire, horizontal, vertical, dim_x, dim_y, delta);
        max_overlap_horiz[i] = find_max_overlap_h(matrix, i, x1, y1, 
                                                        x2, y2, dim_x, dim_y);
    }
    //printf("OG horizontal\n");
    //print(horizontal, 1, dim_y);
    //VERTICAL

#pragma omp parallel for default(shared) private(i) schedule(dynamic)
   for (i = x_min; i <= x_max; i++) {
        //vertical[i] = populate_vertical(matrix, i, x1, y1,
          //                                      x2, y2, dim_x, dim_y);
        max_overlap_vert[i] = find_max_overlap_v(matrix, i, x1, y1, 
                                                        x2, y2, dim_x, dim_y);
    }
    /*printf("OG vertical\n");
    print(vertical, 1, dim_x); 
    printf("max overlap horiz\n");
    print(max_overlap_horiz, 1, dim_y);
    printf("max overlap vert\n");
    print(max_overlap_vert, 1, dim_x);*/
   
    int max_overlap = matrix_max_overlap(matrix, dim_x * dim_y);
    int new_bendx1 = -1;
    int new_bendy1 = -1;
    int new_bendx2 = -1;
    int new_bendy2 = -1;
    // recalculate max overlap to be max of path max and overall max
    for (int i = y_min; i <= y_max; i++) {
        cost_t temp =  max_overlap_horiz[i]; 
        max_overlap_horiz[i] = std::max(temp, max_overlap);
    }
    for (int i = x_min; i <= x_max; i++) {
        cost_t temp = max_overlap_vert[i];
        max_overlap_vert[i] = std::max(temp, max_overlap);
    }
    /*printf("max overlap horiz\n");
    print(max_overlap_horiz, 1, y_max - y_min+1);
    printf("max overlap vert\n");
    print(max_overlap_vert, 1, x_max - x_min+1);
    */
    // find optimal wire route
    int min_max = max_overlap_horiz[y_min] + 1;
    int min_agg = horizontal[y_min] + 1;
    int best = y_min;

    for (int i = y_min; i <= y_max; i++) {
        if (max_overlap_horiz[i] < min_max || 
                (max_overlap_horiz[i] == min_max &&
                 horizontal[i] < min_agg)) {
            best = i;
            min_max = max_overlap_horiz[best];
            min_agg = horizontal[best];
            new_bendy1 = i;
            new_bendx1 = wire.x1;
            // case wire is line segment
            new_bendx1 = -1;
            new_bendy1 = -1;
            new_bendx2 = -1;
            new_bendy2 = -1;
            if ( !(wire.y1 == wire.y2 && i == wire.y1) ) {
                new_bendy1 = i;
                if (i == wire.y1) {
                    new_bendx1 = wire.x2;
                } else {
                    // if horizontal segment aligns w/ y2 or != y1
                    new_bendx1 = wire.x1;
                } if (i != wire.y1 && i != wire.y2) {
                    new_bendx2 = wire.x2;
                    new_bendy2 = i;
                }
            }
        }
    }   
    for (int i = x_min; i < x_max; i++) {
        if (max_overlap_vert[i] < min_max ||
                (max_overlap_vert[i] == min_max && 
                 vertical[i] < min_agg)) {
            best = i;
            min_max = max_overlap_vert[best];
            min_agg = vertical[best];
            new_bendx1 = -1;
            new_bendy1 = -1;
            new_bendx2 = -1;
            new_bendy2 = -1;
            if ( !(wire.x1 == wire.x2 && i == wire.x1) ) {
                // if horizontal segment is align y1
                new_bendx1 = i;
                if (i == wire.x1) {
                    new_bendy1 = wire.y2;
                } else {
                    // if horizontal segment aligns w/ y2 or != y1
                    new_bendy1 = wire.y1;
                } if (i != wire.x1 && i != wire.x2) {
                    new_bendx2 = i;
                    new_bendy2 = wire.y2;
                }
            }
        }
    }
    wire.bend_x1 = new_bendx1;
    wire.bend_y1 = new_bendy1;
    wire.bend_x2 = new_bendx2;
    wire.bend_y2 = new_bendy2;
    wire.cost = min_agg;
    // printf("bend: %d \t%d %d \t%d %d \t%d %d\n", wire.cost, wire.x1, wire.y1,
    //        wire.bend_x1, wire.bend_y1, wire.bend_x2, wire.bend_y2);
    // printf("min_max: %d, best: %d\n", min_max, best);
    change_wire_route(matrix, wire, dim_x, dim_y, 1);
    // printf("%d %d %d %d %d\n", __LINE__, wire.x1, wire.y1, wire.x2, wire.y2);
    // print(matrix, dim_y, dim_x);
}

cost_t *wire_routing(cost_t *matrix, wire_t *wires, int dim_x, int dim_y, 
                    int num_wires, int delta, double anneal_prob,
                    int *horizontal, int *vertical, int *max_overlap_horiz, int *max_overlap_vert) {
    int i;
// #pragma omp parallel for default(shared) private(i) schedule(dynamic)
    for (i = 0; i < num_wires; i++) {
        find_min_path(delta, dim_x, dim_y, wires[i], matrix, anneal_prob, 
                      horizontal, vertical, max_overlap_horiz, max_overlap_vert);
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
    int *horizontal = (cost_t *)calloc(dim_y, sizeof(cost_t));
    int *vertical = (cost_t *)calloc(dim_x, sizeof(cost_t));
    int *max_overlap_horiz = (cost_t *)calloc(dim_y, sizeof(cost_t));
    int *max_overlap_vert = (cost_t *)calloc(dim_x, sizeof(cost_t));

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
    inout(costs: length(dim_x*dim_y) INOUT)     \
    inout(horizontal: length(dim_y) INOUT)      \
    inout(vertical: length(dim_x) INOUT)        \
    inout(max_overlap_horiz: length(dim_y) INOUT) \
    inout(max_overlap_vert: length(dim_x) INOUT)
    #endif
    {
        for (int i = 0; i < SA_iters; i++) {
            costs = wire_routing(costs, wires, dim_x, dim_y, num_of_wires, delta, SA_prob,
                        horizontal, vertical, max_overlap_horiz, max_overlap_vert); 
        }
       
         /* Implement the wire routing algorithm here
        * Feel free to structure the algorithm into different functions
        * Don't use global variables.
        * Use OpenMP to parallelize the algorithm. 
         * You should really implement as much of this (if not all of it) in
        * helper functions. */
    }
    cost_t max_overlap = matrix_max_overlap(costs, dim_x * dim_y);
    printf("Max overlap: %d\n", max_overlap);
    

    cost_t agg_overlap = matrix_agg_overlap(costs, dim_x * dim_y);
    printf("Max overlap: %d\n", agg_overlap);
    
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
        fprintf(output_routes_file, "%d %d ", wire.x1, wire.y1);
        if (wire.bend_x1 >= 0 && wire.bend_y1 >= 0 &&
            !(wire.bend_x1 == wire.x1 && wire.bend_y1 == wire.y1))
            fprintf(output_routes_file, "%d %d ", wire.bend_x1, wire.bend_y1);
        if (wire.bend_x2 >= 0 && wire.bend_y2 >= 0)
            fprintf(output_routes_file, "%d %d ", wire.bend_x2, wire.bend_y2);
        fprintf(output_routes_file, "%d %d\n", wire.x2, wire.y2);
    }  
    // WRITE WIRES TO FILE HERE

    fclose(output_routes_file);

    free(horizontal);
    free(vertical);
    free(max_overlap_horiz);
    free(max_overlap_vert);

    return 0;
}
