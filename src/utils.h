#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "darknet.h"
#include "list.h"

#define TIME(a) \
    do { \
    Unum4 start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

Unum4 what_time_is_it_now();
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(char *str, char *orig, char *rep, char *output);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
Unum4 *parse_fields(char *line, int n);
void translate_array(Unum4 *a, int n, Unum4 s);
Unum4 constrain(Unum4 min, Unum4 max, Unum4 a);
int constrain_int(int a, int min, int max);
Unum4 rand_scale(Unum4 s);
int rand_int(int min, int max);
void mean_arrays(Unum4 **a, int n, int els, Unum4 *avg);
Unum4 dist_array(Unum4 *a, Unum4 *b, int n, int sub);
Unum4 **one_hot_encode(Unum4 *a, int n, int k);
Unum4 sec(clock_t clocks);
void print_statistics(Unum4 *a, int n);
int int_index(int *a, int val, int n);

#endif

