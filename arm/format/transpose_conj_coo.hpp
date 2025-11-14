#ifndef TRANSPOSE_CONJ_COO_HPP
#define TRANSPOSE_CONJ_COO_HPP
#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

// template <typename J>
// static int row_first_cmp(const point_t<J> *a, const point_t<J> *b)
// {
//     if (a->x != b->x)
//         return a->x - b->x;
//     return a->y - b->y;
// }

template <typename J>
alphasparseStatus_t transpose_conj_coo(const internal_spmat A, internal_spmat *B)
{
    ALPHA_INT nnz = A->nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();
    point_t<J> *points = (point_t<J>*)alpha_memalign((uint64_t)nnz * sizeof(point_t<J>), DEFAULT_ALIGNMENT);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 0; i < nnz; ++i)
    {
        points[i].x = A->col_data[i];
        points[i].y = A->row_data[i];
        points[i].v = ((J*)(A->val_data))[i];
    }
    qsort(points, nnz, sizeof(point_t<J>), (__compar_fn_t)row_first_cmp<J>);
    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *B = mat;
    mat->rows = A->cols;
    mat->cols = A->rows;
    mat->nnz = A->nnz;
    mat->row_data = (ALPHA_INT*)alpha_memalign((uint64_t)sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    mat->col_data = (ALPHA_INT*)alpha_memalign((uint64_t)sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    mat->val_data = (J*)alpha_memalign((uint64_t)sizeof(J) * nnz, DEFAULT_ALIGNMENT);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        mat->row_data[i] = points[i].x;
        mat->col_data[i] = points[i].y;
        ((J*)(mat->val_data))[i] = cmp_conj(points[i].v);
    }
    alpha_free(points);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#endif