#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include <memory.h>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename J>
alphasparseStatus_t spmm_csr(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    check_return(B->rows != A->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *matC = mat;
    mat->rows = A->rows;
    mat->cols = B->cols;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    // 计算所需空间
    ALPHA_INT64 flop[m];
    memset(flop,'\0',m*sizeof(ALPHA_INT64));
    ALPHA_INT *row_offset = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * (m + 1), DEFAULT_ALIGNMENT);
    mat->row_data = row_offset;
    ALPHA_INT * rows_end = row_offset + 1;
    memset(row_offset,'\0',sizeof(ALPHA_INT)*(m+1));

    ALPHA_INT num_thread = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        bool flag[n];
        memset(flag, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->row_data[ar]; ai < A->row_data[ar+1]; ai++)
        {
            ALPHA_INT br = A->col_data[ai];
            flop[ar] += B->row_data[br+1] - B->row_data[br];
            for (ALPHA_INT bi = B->row_data[br]; bi < B->row_data[br+1]; bi++)
            {
                if (!flag[B->col_data[bi]])
                {
                    rows_end[ar] += 1;
                    flag[B->col_data[bi]] = true;
                }
            }
        }
    }
    
    for(ALPHA_INT i = 1;i < m;++i)
    {
        flop[i] += flop[i - 1];
        rows_end[i] += rows_end[i-1];
    }
    ALPHA_INT nnz = mat->row_data[m];

    mat->col_data = (ALPHA_INT *)alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = alpha_memalign(nnz * sizeof(J), DEFAULT_ALIGNMENT);

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        J values[n];
        memset(values, '\0', sizeof(J) * n);
        bool write_back[n];
        memset(write_back, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->row_data[ar]; ai < A->row_data[ar+1]; ai++)
        {
            ALPHA_INT br = A->col_data[ai];
            J av = ((J*)A->val_data)[ai];
            ALPHA_INT bis = B->row_data[br];
            ALPHA_INT bie = B->row_data[br+1];
            ALPHA_INT bil = bie-bis;
            const ALPHA_INT* B_col = &B->col_data[bis];
            const J* B_val = &((J*)B->val_data)[bis];
            ALPHA_INT bi = 0;
            for (; bi < bil-3; bi+=4)
            {
                ALPHA_INT bc0 = B_col[bi];
                ALPHA_INT bc1 = B_col[bi+1];
                ALPHA_INT bc2 = B_col[bi+2];
                ALPHA_INT bc3 = B_col[bi+3];
                J bv0 = B_val[bi];
                J bv1 = B_val[bi+1];
                J bv2 = B_val[bi+2];
                J bv3 = B_val[bi+3];
                values[bc0] = alpha_madd(av, bv0, values[bc0]);
                values[bc1] = alpha_madd(av, bv1, values[bc1]);
                values[bc2] = alpha_madd(av, bv2, values[bc2]);
                values[bc3] = alpha_madd(av, bv3, values[bc3]);
                write_back[bc0] = true;
                write_back[bc1] = true;
                write_back[bc2] = true;
                write_back[bc3] = true;
            }
            for (; bi < bil; bi++)
            {
                ALPHA_INT bc = B_col[bi];
                J bv = B_val[bi];
                values[bc] = alpha_madd(av, bv, values[bc]);
                write_back[bc] = true;
            }
        }
        
        ALPHA_INT index = mat->row_data[ar];
        for (ALPHA_INT c = 0; c < n; c++)
        {
            if (write_back[c])
            {
                mat->col_data[index] = c;
                ((J*)mat->val_data)[index] = values[c];
                index += 1;
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
