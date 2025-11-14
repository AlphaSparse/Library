#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t set_value_bsr(internal_spmat A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const TYPE value)
{
	bool find = false;
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT block_row = row / bs;
	ALPHA_INT block_col = col / bs;
	ALPHA_INT inblock_row = row % bs;
	ALPHA_INT inblock_col = col % bs;
	for(ALPHA_INT ai = A->row_data[block_row]; ai < A->row_data[block_row+1]; ++ai)
	{
		const ALPHA_INT ac = A->col_data[ai];
		if(ac == block_col)
		{
			ALPHA_INT base = ai * bs * bs;
			((TYPE *)A->val_data)[base + inblock_row * bs + inblock_col] = value;
			find = true;
			break;
		}
	}

	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
