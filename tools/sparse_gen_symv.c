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
    int64_t m, n;
    double rate;
    m = 729000;
    n = 729000;
    int64_t nnz = 20000000;
    rate = nnz/m*n;
    //if (argc == 4)
    //{
    //    m = atol(argv[1]);
    //    n = atol(argv[2]);
    //    rate = atof(argv[3]);
    //}
    //else if (argc == 3)
    //{
    //    m = atoi(argv[1]);
    //    n = atoi(argv[2]);
    //    rate = 0.05;
    //}
    //else if (argc == 2)
    //{
    //    m = n = atoi(argv[1]);
    //    rate = 0.05;
    //}
    //else
    //{
    //    printf("Usage : sparse_gen m [n] [rate]\n");
    //    exit(0);
    //}
    size_t number = nnz;
    point_t *points = malloc(sizeof(point_t) * number);
    for (int i = 0; i < number; i++)
    {
        points[i].x = random_int(m) + 1;
        points[i].y = random_int(200) + 1;
        points[i].v = random_float();

        points[i].y = points[i].y - 100 + points[i].x - 1;          
        if(points[i].y <= 0){
            points[i].y += n;
        }

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
    sprintf(filename, "Matrix/symv_%ld_%ld_%ld.mtx", m, n, count);
    FILE *fp = fopen(filename, "w");
    if(fp == NULL){
        printf("output file open error!!!");
        exit(-1);
    }
    fprintf(fp, "%ld %ld %ld\n", m, n, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(fp, "%d %d %.6f\n", points[i].x, points[i].y,points[i].v);
    }
    printf("%ld %ld %ld \n", m, n, count);

    free(points);
}
