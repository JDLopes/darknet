#ifndef TREE_H
#define TREE_H
#include "darknet.h"

int hierarchy_top_prediction(Unum4 *predictions, tree *hier, Unum4 thresh, int stride);
Unum4 get_hierarchy_probability(Unum4 *x, tree *hier, int c, int stride);

#endif
