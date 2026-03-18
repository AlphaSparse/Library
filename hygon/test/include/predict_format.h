#ifndef PREDICT_FORMAT_H
#define PREDICT_FORMAT_H

#include "alphasparse/opt.h"
#include "alphasparse/format.h"
#include "alphasparse/spdef.h"

alphasparseFormat_t predict_best_format_from_coo(const internal_spmat mat, int *best_C, int *best_sigma);

#endif // PREDICT_FORMAT_H
