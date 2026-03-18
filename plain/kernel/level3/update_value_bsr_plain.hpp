#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t update_values_bsr_plain(internal_spmat A, 
										const ALPHA_INT nvalues, 
										const ALPHA_INT *indx, 
										const ALPHA_INT *indy, 
										TYPE *values)
{
	bool find = false;
	if(indx != NULL && indy != NULL)
	{
		for(ALPHA_INT i = 0; i < nvalues; i++)
		{
			ALPHA_INT row = indx[i];
			ALPHA_INT col = indy[i];
			ALPHA_INT bs = A->block_dim;
			ALPHA_INT block_row = row / bs;
			ALPHA_INT block_col = col / bs;
			ALPHA_INT block_row_inside = row % bs;
			ALPHA_INT block_col_inside = col % bs;
			for(ALPHA_INT ai = A->row_data[block_row]; ai < A->row_data[block_row+1]; ai++)
			{
				const ALPHA_INT ac = A->col_data[ai];
				if(ac == block_col)
				{
					ALPHA_INT idx = 0;
					if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
						idx = ai * bs * bs + block_row_inside * bs + block_col_inside;
					else
						idx = ai * bs * bs + block_row_inside + block_col_inside * bs;
					
						((TYPE*)A->val_data)[idx] = values[i];
					find = true;
					break;
				}
			}
		}
	}
	else // updates all elements of the matrix A
	{
		for(ALPHA_INT i = 0; i < nvalues; i++)
			((TYPE*)A->val_data)[i] = values[i];
	}
	
	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
