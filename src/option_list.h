#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
Unum4 option_find_Unum4(list *l, char *key, Unum4 def);
Unum4 option_find_Unum4_quiet(list *l, char *key, Unum4 def);
void option_unused(list *l);

#endif
