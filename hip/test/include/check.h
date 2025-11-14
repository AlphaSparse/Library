#pragma once
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include "alphasparse/spapi.h"

int check(const int32_t *answer_data, size_t answer_size, const int32_t *result_data,
          size_t result_size);
int check(const int8_t *answer_data, size_t answer_size, const int8_t *result_data,
          size_t result_size);
int check(const float *answer_data, size_t answer_size, const float *result_data, size_t result_size);
int check(const double *answer_data, size_t answer_size, const double *result_data, size_t result_size);
int check(const hipFloatComplex *answer_data, size_t answer_size, const hipFloatComplex *result_data, size_t result_size);
int check(const hipDoubleComplex *answer_data, size_t answer_size, const hipDoubleComplex *result_data, size_t result_size);

int check_s_l2(const float *answer_data, size_t answer_size, const float *result_data, size_t result_size, const float *x, const float *y, const float alpha, const float beta, int argc, const char *argv[]);
int check_s_l3(const float *answer_data, const int ldans, size_t answer_size, const float *result_data, const int ldres, size_t result_size, const int *res_col_indx, const float *x, const int ldx, const float *y, const int ldy, const float alpha, const float beta, int argc, const char *argv[]);
int check_d_l2(const double *answer_data, size_t answer_size, const double *result_data, size_t result_size, const double *x, const double *y, const double alpha, const double beta, int argc, const char *argv[]);
int check_d_l3(const double *answer_data, const int ldans, size_t answer_size, const double *result_data, const int ldres, size_t result_size, const int *res_col_indx, const double *x, const int ldx, const double *y, const int ldy, const double alpha, const double beta, int argc, const char *argv[]);
int check_c_l2(const hipFloatComplex *answer_data, size_t answer_size, const hipFloatComplex *result_data, size_t result_size, const hipFloatComplex *x, const hipFloatComplex *y, const hipFloatComplex alpha, const hipFloatComplex beta, int argc, const char *argv[]);
int check_c_l3(const hipFloatComplex *answer_data, const int ldans, size_t answer_size, const hipFloatComplex *result_data, const int ldres, size_t result_size, const int *res_col_indx, const hipFloatComplex *x, const int ldx, const hipFloatComplex *y, const int ldy, const hipFloatComplex alpha, const hipFloatComplex beta, int argc, const char *argv[]);
int check_z_l2(const hipDoubleComplex *answer_data, size_t answer_size, const hipDoubleComplex *result_data, size_t result_size, const hipDoubleComplex *x, const hipDoubleComplex *y, const hipDoubleComplex alpha, const hipDoubleComplex beta, int argc, const char *argv[]);
int check_z_l3(const hipDoubleComplex *answer_data, const int ldans, size_t answer_size, const hipDoubleComplex *result_data, const int ldres, size_t result_size, const int *res_col_indx, const hipDoubleComplex *x, const int ldx, const hipDoubleComplex *y, const int ldy, const hipDoubleComplex alpha, const hipDoubleComplex beta, int argc, const char *argv[]);

void check_int_vec(int *answer_data, int size_ans, int *result_data, int size_res);