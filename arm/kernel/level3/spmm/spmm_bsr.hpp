#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <stdbool.h>
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t spmm_bsr(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    // check_return(A->cols != B->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    // check_return(A->block_dim != B->block_dim, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    // check_return(A->block_layout != B->block_layout, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *matC = mat;
    mat->rows         = A->rows;
    mat->cols         = B->cols;
    mat->block_layout = A->block_layout;
    mat->block_dim   = A->block_dim;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    ALPHA_INT bs = A->block_dim;
    // ��������ռ�
    bool *flag = (bool *)alpha_memalign(sizeof(bool) * n, DEFAULT_ALIGNMENT);
    ALPHA_INT nnz = 0; //��¼matC�з���Ԫ�صĸ���
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        memset(flag, '\0', sizeof(bool) * n); //flagһ�α��mat��һ��
        for (ALPHA_INT ai = A->row_data[ar]; ai < A->row_data[ar+1]; ai++)
        {
            ALPHA_INT br = A->col_data[ai];
            for (ALPHA_INT bi = B->row_data[br]; bi < B->row_data[br+1]; bi++)
            {
                if (!flag[B->col_data[bi]])
                {
                    nnz += 1;
                    flag[B->col_data[bi]] = true;
                }
		        /*if(flag[B->col_data[bi]]==true)
                    continue;
                if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                {
                    // block row major
                    for(ALPHA_INT block_ar = 0; block_ar < A->block_dim; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < A->block_dim; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < B->block_dim; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                TYPE av = ((TYPE *)A->val_data)[bs*bs*ai + bs*block_ar + block_ac];
                                TYPE bv = ((TYPE *)B->val_data)[bs*bs*bi + bs*block_br + block_bc];
                                if(flag[B->col_data[bi]]==false && av*bv !=0.)
                                {
                                    nnz+=1;
                                    flag[B->col_data[bi]]=true;
                                }
                            }
                        }
                    }
                }
                else
                {
                    // block col major
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }*/
            }
        }
    }
    alpha_free(flag);
    //printf("nnz:%d\n", (int)nnz);

    ALPHA_INT *row_offset = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * (m + 1), DEFAULT_ALIGNMENT);
    mat->row_data = row_offset;
    ALPHA_INT * rows_end = row_offset + 1;
    mat->col_data = (ALPHA_INT *)alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = alpha_memalign(nnz * bs * bs * sizeof(TYPE), DEFAULT_ALIGNMENT);
    memset(mat->val_data, '\0', sizeof(TYPE) * nnz * bs * bs);

    TYPE *values = (TYPE *)alpha_memalign(sizeof(TYPE) * n * bs * bs, DEFAULT_ALIGNMENT);

    ALPHA_INT index = 0;
    mat->row_data[0] = 0;
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        bool *flaggg = (bool *)alpha_memalign(sizeof(bool) * n, DEFAULT_ALIGNMENT); //��mkl���
	    memset(flaggg, '\0', sizeof(bool) * n);
	    memset(values, '\0', sizeof(TYPE) * n * bs * bs);
        for (ALPHA_INT ai = A->row_data[ar]; ai < A->row_data[ar+1]; ai++) //��A�ĵ�ar��������˷�
        {
            ALPHA_INT br = A->col_data[ai];
            //TYPE av = ((TYPE *)A->val_data)[ai];
            for (ALPHA_INT bi = B->row_data[br]; bi < B->row_data[br+1]; bi++)
            {
		        ALPHA_INT bc = B->col_data[bi];
                //values[bc] += av * ((TYPE *)B->val_data)[bi];
                if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                {
                    // row major
                    for(ALPHA_INT block_ar = 0; block_ar < A->block_dim; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < A->block_dim; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < B->block_dim; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                TYPE av = ((TYPE *)A->val_data)[bs*bs*ai + bs*block_ar + block_ac];
                                TYPE bv = ((TYPE *)B->val_data)[bs*bs*bi + bs*block_br + block_bc];
                                //matC[index2(ar*bs+block_ar, bc*bs+block_bc, ldc)] += av*bv;
                                //values[bc*bs*bs + block_ar*bs + block_bc] += av * bv;
                                values[bc*bs*bs + block_ar*bs + block_bc] = alpha_madde(values[bc*bs*bs + block_ar*bs + block_bc], av, bv);
				                flaggg[B->col_data[bi]]=true;
                            }
                        }
                    }
                }
                else
                {
                    // col major
		            for(ALPHA_INT block_ar = 0; block_ar < A->block_dim; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < A->block_dim; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < B->block_dim; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                TYPE av = ((TYPE *)A->val_data)[bs*bs*ai + bs*block_ac + block_ar];
                                TYPE bv = ((TYPE *)B->val_data)[bs*bs*bi + bs*block_bc + block_br];
                                //matC[index2(ar*bs+block_ar, bc*bs+block_bc, ldc)] += av*bv;
                                //values[bc*bs*bs + block_bc*bs + block_ar] += av * bv;
                                values[bc*bs*bs + block_bc*bs + block_ar] = alpha_madde(values[bc*bs*bs + block_bc*bs + block_ar], av, bv);
                                flaggg[B->col_data[bi]]=true;
			                }
			            }
		            }
                }
            }
        }
        for (ALPHA_INT c = 0; c < n; c++)// ��matC�ĵ�ar�еķ���Ԫ�ر����csr
        {
            /*bool flag = false;
            // ��c��block�Ƿ�Ϊ�����
            for(ALPHA_INT i = 0; i < bs*bs; i++)
            {
                if(values[c*bs*bs+i] != 0.)
                {
                    flag = true;
                    continue;
                }
            }
            if (flag == true)*/
	        if(flaggg[c] == true)
            {
                mat->col_data[index] = c;
                for(int i=0; i < bs*bs; i++)
                {
                    ((TYPE *)mat->val_data)[index*bs*bs+i] = values[c*bs*bs+i];
                }
                index += 1;
            }
        }
        rows_end[ar] = index;
	    //printf("rows_end[ar:%d],index:%d\n", (int)ar, (int)index);
    }

    alpha_free(values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
