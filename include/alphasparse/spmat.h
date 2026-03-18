#pragma once

/**
 *  @brief header for the internal sparse matrix definitions
 */
#include "spdef.h"
// #include "types.h"
#ifdef __CUDA__
#include "./spmat_def/coo.def.cuh"
#include "./spmat_def/csr.def.cuh"
#else
#include "./spmat_def/coo.def.h"
#include "./spmat_def/csr.def.h"
#endif

#include "./spmat_def/csc.def.h"
#include "./spmat_def/bsr.def.h"
#include "./spmat_def/dia.def.h"
#include "./spmat_def/sky.def.h"
#include "./spmat_def/ell.def.h"
#include "./spmat_def/gebsr.def.h"
#include "./spmat_def/hyb.def.h"
#include "./spmat_def/csr5.def.h"
#include "./spmat_def/cooaos.def.h"
#include "./spmat_def/sell_c_sigma.def.h"
#include "./spmat_def/ellr.def.h"