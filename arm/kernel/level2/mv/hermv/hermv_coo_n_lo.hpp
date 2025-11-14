#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"
#include <stdio.h>


template <typename TYPE>
alphasparseStatus_t
hermv_coo_n_lo(const TYPE alpha,
      const internal_spmat A,
      const TYPE *x,
      const TYPE beta,
      TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
	const ALPHA_INT nnz = A->nnz;
	

    for (ALPHA_INT i = 0; i < m; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}

    for(ALPHA_INT i = 0; i < nnz; ++i)
	{
		const ALPHA_INT r = A->row_data[i];
		const ALPHA_INT c = A->col_data[i];
        const TYPE origin_val = ((TYPE *)A->val_data)[i];
        const TYPE conj_val = cmp_conj(origin_val);
		if(r < c)
		{
			continue;
		}
		TYPE v,v_c;
		v = alpha_mul(origin_val, alpha);
		v_c = alpha_mul(conj_val, alpha);
		if(r == c)
		{
			y[r] = alpha_madde(y[r], v, x[c]);
		}
		else
		{
			y[r] = alpha_madde(y[r], v, x[c]);
			y[c] = alpha_madde(y[c], v_c, x[r]);
	 	}
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;

}