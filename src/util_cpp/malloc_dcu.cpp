/**
 * @brief implement for malloc utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include "alphasparse/util/malloc.h"
#ifdef __cplusplus
}
#endif

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#ifdef __DCU__
#include <hip/hip_runtime_api.h>
#endif

void alpha_free_dcu(void *point)
{
    if (point) {
#ifdef __DCU__
        hipFree(point);
#else
        alpha_free(point);
#endif
    }
}
