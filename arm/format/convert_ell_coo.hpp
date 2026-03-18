#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdio.h>
#include "convert_csr_coo.hpp"

// static void print_coo_s(const spmat_coo_s_t *mat)
// {
//     printf("nnz:%d, cols:%d, rows:%d\n", mat->nnz, mat->cols, mat->rows);
//     for (ALPHA_INT i = 0; i < mat->nnz; i++)
//     {
//         printf("#%d, val:%f, row:%d, col:%d\n", i, mat->values[i], mat->row_indx[i], mat->col_indx[i]);
//     }
//     printf("=====================================\n\n");
// }

// static void print_ell_s(const spmat_ell_s_t *mat)
// {
//     printf("ld:%d, cols:%d, rows:%d\n", mat->ld, mat->cols, mat->rows);
//     for(ALPHA_INT i = 0; i < mat->ld; i++)
//     {
//         for(ALPHA_INT j = 0; j < mat->rows; j++)
//         {
//             printf("%f ", mat->values[i*mat->rows + j]);
//         }
//         printf("\n");
//     }
//     printf("=====================================\n\n");
// }

template <typename I, typename J, typename T>
alphasparseStatus_t convert_ell_coo(const T *source, T **dest)
{
    T *mat = (T*)alpha_malloc(sizeof(T));
    *dest = mat;

    T *csr;
    convert_csr_coo<I, J, T>(source, &csr);

    I m = csr->rows;
    I n = csr->cols;

    mat->rows = m;
    mat->cols = n;

    I ld = 0;
    for (I i = 0; i < m; i++)
    {
        I row_nnz = csr->row_data[i+1] - csr->row_data[i];
        ld = ld > row_nnz ? ld : row_nnz;
    }
    mat->ell_width = ld;
    double ell_padding_ratio = 1.0 * ld * m / source->nnz;
    printf("padding ratio is %lf\n",ell_padding_ratio);
    if((uint64_t )ld * m >= 1l<<31){
        fprintf(stderr,"nnz nums overflow!!!:%ld\n",(uint64_t )ld * m);
        exit(EXIT_FAILURE);
    }
    J *values = (J*)alpha_memalign((uint64_t)ld * m * sizeof(J), DEFAULT_ALIGNMENT);
    I *indices = (I*)alpha_memalign((uint64_t)ld * m * sizeof(I), DEFAULT_ALIGNMENT);
    memset(values,0,(uint64_t)ld * m * sizeof(J));
    memset(indices,0,(uint64_t)ld * m * sizeof(I));

    const ALPHA_INT thread_num = alpha_get_thread_num();
    // i列j行, 列优先
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(thread_num)
    #endif
    for (ALPHA_INT i = 0; i < ld; i++)
    {
        for (ALPHA_INT j = 0; j < m; j++)
        {
            ALPHA_INT csr_rs = csr->row_data[j];
            ALPHA_INT csr_re = csr->row_data[j+1];
            if (csr_rs + i < csr_re)
            {
                values[i * m + j] = ((J*)csr->val_data)[csr_rs + i];
                indices[i * m + j] = csr->col_data[csr_rs + i];
            }
        }
    }
    mat->val_data = values;
    mat->ind_data = indices;
// #ifndef COMPLEX
// #ifndef DOUBLE    
//     print_ell_s(mat);
// #endif
// #endif

    // mat->d_values  = NULL;
    // mat->d_indices = NULL;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
