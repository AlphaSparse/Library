#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"
#include <stdio.h>

template <typename TYPE>
alphasparseStatus_t
hermv_coo_n_lo_trans(const TYPE alpha,
      const internal_spmat A,
      const TYPE *x,
      const TYPE beta,
      TYPE *y)
{
    //TODO 
    internal_spmat transposed_mat;
    transpose_coo<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = hermv_coo_n_hi(    alpha,
		                                                    transposed_mat,
                                                            x,
                                                            beta,
                                                            y);
    destroy_coo(transposed_mat);
    return status;

}