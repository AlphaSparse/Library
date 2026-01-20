#pragma once

/**
 * @brief header for optimizetion related parameter definitions
 */

#include "alphasparse/util/thread.h"

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
    typedef int (*__compar_fn_t)(const void *, const void *);
#endif
