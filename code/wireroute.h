/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#ifndef __WIREOPT_H__
#define __WIREOPT_H__

#include <omp.h>

typedef int cost_t;
typedef struct
{
    int x1;
    int x2;
    int y1;
    int y2;
    int bend_x1;
    int bend_x2;
    int bend_y1;
    int bend_y2;
    cost_t cost;
} wire_t;

const char *get_option_string(const char *option_name, const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

#endif
