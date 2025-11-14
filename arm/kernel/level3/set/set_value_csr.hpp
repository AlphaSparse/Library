#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t set_value_csr(internal_spmat A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const TYPE value)
{
	bool find = false;
	for(ALPHA_INT ai = A->row_data[row]; ai < A->row_data[row+1]; ++ai)
	{
		const ALPHA_INT ac = A->col_data[ai];
		if(ac == col)
		{
			((TYPE *)A->val_data)[ai] = value;
			find = true;
			break;
		}
	}

	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
