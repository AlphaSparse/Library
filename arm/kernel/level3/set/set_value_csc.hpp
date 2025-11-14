#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t set_value_csc(internal_spmat A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const TYPE value)
{
	bool find = false;
	for(ALPHA_INT ai = A->col_data[col]; ai < A->col_data[col+1]; ++ai)
	{
		const ALPHA_INT ar = A->row_data[ai];
		if(ar == row)
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
