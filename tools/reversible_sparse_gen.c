/**
 * @brief  random based sparse matrix generation tool
 * @author Zhuoqiang Guo <gzq9425@qq.com> 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
    typedef int (*__compar_fn_t)(const void *, const void *);
#endif

inline static int random_int(int m)
{
    return rand() % m;
}

inline float random_float()
{
    return (float)rand() / RAND_MAX;
}

typedef struct
{
    int x;
    int y;
    float v;
} point_t;

static int row_first_cmp(const point_t *a, const point_t *b)
{
    if (a->x != b->x)
        return a->x - b->x;
    return a->y - b->y;
}

int main(int argc, char *argv[])
{
    time_t t;
    srand((unsigned)time(&t));
    int64_t m;
    double rate;
    if (argc == 3)
    {
        m = atol(argv[1]);
        rate = atof(argv[3]);
    }
    else if (argc == 2)
    {
        m = atoi(argv[1]);
        rate = 0.05;
    }
    else
    {
        printf("Usage : reversible_sparse_gen m [rate]\n");
        exit(0);
    }

    size_t number = m * m * rate;

    size_t diag = m;

    point_t *points = malloc(sizeof(point_t) * number);
    int i = 0;
    for (; i < diag; i++)
    {
        points[i].x = i + 1;
        points[i].y = i + 1;
        points[i].v = random_float();
    }
    for (; i < number; i++)
    {
        points[i].x = random_int(m) + 1;
        points[i].y = random_int(m) + 1;
        points[i].v = random_float();
    }
    //sort
    qsort(points, number, sizeof(point_t), (__compar_fn_t)row_first_cmp);

    //unique
    int index = 0;
    for (int i = 1; i < number; i++)
    {
        if (!(points[i].x == points[index].x && points[i].y == points[index].y))
        {
            index++;
            points[index] = points[i];
        }
    }
    int64_t count = index + 1;
    //output
    char filename[100];
    sprintf(filename, "Matrix/r_%ld_%ld_%ld.mtx", m, m, count);
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("output file open error!!!");
        exit(-1);
    }
    fprintf(fp, "%ld %ld %ld\n", m, m, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(fp, "%d %d %.6f\n", points[i].x, points[i].y,points[i].v);
    }
    printf("%ld %ld %ld \n", m, m, count);

    free(points);
}
