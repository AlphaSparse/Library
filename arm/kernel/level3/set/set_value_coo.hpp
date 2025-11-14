#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t set_value_coo(internal_spmat A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const TYPE value)
{
	bool find = false;
	for(ALPHA_INT ai = 0; ai < A->nnz; ++ai)
		if(A->row_data[ai] == row && A->col_data[ai] == col)
		{
			((TYPE *)A->val_data)[ai] = value;
			find = true;
			break;
		}
	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
