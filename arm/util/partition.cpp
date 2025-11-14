#include "alphasparse/util.h"
#include "alphasparse/types.h"
#include "alphasparse/util/thread.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/util/bisearch.h"
#include "alphasparse/util/partition.h"
#include <math.h>
#include <limits.h>
#include <stdint.h>

void find_divisors(ALPHA_INT n, ALPHA_INT *divisors, ALPHA_INT *num)
{
    ALPHA_INT count = 0;
    for (ALPHA_INT i = 1; i <= n; ++i)
    {
        if (n % i == 0)
        {
            divisors[count++] = i;
        }
    }
    *num = count;
}

void balanced_divisors2(ALPHA_INT m, ALPHA_INT n, ALPHA_INT num_threads, ALPHA_INT *divm_p, ALPHA_INT *divn_p)
{
    double target_size = sqrt((double)m * n / num_threads);
    ALPHA_INT divm = 1;
    ALPHA_INT divn = 1;
    double error = LONG_MAX;
    ALPHA_INT divisors[num_threads];
    ALPHA_INT div_num;
    find_divisors(num_threads, divisors, &div_num);
    for (ALPHA_INT i = 0; i < div_num; i++)
    {
        ALPHA_INT div1 = divisors[i];
        ALPHA_INT div2 = num_threads / div1;
        double current_error = fabs(m / div1 - target_size) +
                               fabs(n / div2 - target_size);
        if (current_error < error)
        {
            divm = div1;
            divn = div2;
            error = current_error;
        }
    }
    *divm_p = divm;
    *divn_p = divn;
}

int lower_bound_int(const ALPHA_INT *t, ALPHA_INT l, ALPHA_INT r, ALPHA_INT value)
{
    while (r > l)
    {
        ALPHA_INT m = (l + r) / 2;
        if (t[m] < value)
        {
            l = m + 1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}
int lower_bound_int64(const ALPHA_INT64 *t, ALPHA_INT64 l, ALPHA_INT64 r, ALPHA_INT64 value)
{
    while (r > l)
    {
        ALPHA_INT m = (l + r) / 2;
        if (t[m] < value)
        {
            l = m + 1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}

void balanced_partition_row_by_nnz(const ALPHA_INT *acc_sum_arr, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition)
{
    ALPHA_INT nnz = acc_sum_arr[rows - 1];
    ALPHA_INT ave = nnz / num_threads;
    partition[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 1; i < num_threads; i++)
    {
        partition[i] = lower_bound_int(acc_sum_arr, 0, rows - 1, (ave * i));
    }
    partition[num_threads] = rows;
}
ALPHA_INT partition_and_locate_uppernnz(const ALPHA_INT *acc_sum_arr, const ALPHA_INT *index, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition, ALPHA_INT *row_start)
{
    ALPHA_INT *nnz_nums = (ALPHA_INT *)alpha_malloc((uint64_t)sizeof(ALPHA_INT) * (rows));
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 0; i < rows; i++)
    {
        ALPHA_INT start = alpha_lower_bound(&index[acc_sum_arr[i - 1]], &index[acc_sum_arr[i]], i) - index;
        row_start[i] = start;
        nnz_nums[i] = acc_sum_arr[i] - start;
    }
    //inclusive prefix sum
    for (ALPHA_INT i = 1; i < rows; i++)
    {
        nnz_nums[i] += nnz_nums[i - 1];
    }

    ALPHA_INT nnz = nnz_nums[rows - 1];
    ALPHA_INT ave = nnz / num_threads;
    partition[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 1; i < num_threads; i++)
    {
        partition[i] = lower_bound_int(nnz_nums, 0, rows - 1, (ave * i));
    }
    partition[num_threads] = rows;
    alpha_free(nnz_nums);
    return nnz;
}
ALPHA_INT partition_and_locate_lowernnz(const ALPHA_INT *acc_sum_arr, const ALPHA_INT *index, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition, ALPHA_INT *row_end)
{

    ALPHA_INT *nnz_nums = (ALPHA_INT *)alpha_malloc((uint64_t)sizeof(ALPHA_INT) * (rows));
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 0; i < rows; i++)
    {
        ALPHA_INT end = alpha_upper_bound(&index[acc_sum_arr[i - 1]], &index[acc_sum_arr[i]], i) - index;
        row_end[i] = end;
        nnz_nums[i] = end - acc_sum_arr[i - 1];
    }
    //inclusive prefix sum
    for (ALPHA_INT i = 1; i < rows; i++)
    {
        nnz_nums[i] += nnz_nums[i - 1];
    }

    ALPHA_INT nnz = nnz_nums[rows - 1];
    ALPHA_INT ave = nnz / num_threads;
    partition[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 1; i < num_threads; i++)
    {
        partition[i] = lower_bound_int(nnz_nums, 0, rows - 1, (ave * i));
    }
    partition[num_threads] = rows;
    alpha_free(nnz_nums);
    return nnz;
}

void balanced_partition_row_by_flop(const ALPHA_INT64 *acc_sum_arr, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition)
{
    ALPHA_INT64 nnz = acc_sum_arr[rows - 1];
    ALPHA_INT64 ave = nnz / num_threads;
    partition[0] = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 1; i < num_threads; i++)
    {
        partition[i] = lower_bound_int64(acc_sum_arr, 0, rows - 1, (ave * i));
    }
    partition[num_threads] = rows;
}

void block_partition(ALPHA_INT *pointerB, ALPHA_INT *pointerE, ALPHA_INT *block_indx, ALPHA_INT block_dim_len, ALPHA_INT another_dim_len, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
{
    ALPHA_INT block_num = (block_dim_len + block_size - 1) / block_size;
    ALPHA_INT ldp = block_num + 1;
    ALPHA_INT *pos = (ALPHA_INT *)alpha_memalign((uint64_t)another_dim_len * ldp * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    for (ALPHA_INT64 r = 0; r < another_dim_len; ++r)
    {
        ALPHA_INT ai = pointerB[r];
        ALPHA_INT aie = pointerE[r];
        pos[index2(r, 0, ldp)] = ai;
        ALPHA_INT block_upper_bound = block_size;
        for (ALPHA_INT bi = 1; bi <= block_num; ++bi, block_upper_bound += block_size)
        {
            while (ai < aie && block_indx[ai] < block_upper_bound)
            {
                ai += 1;
            }
            pos[index2(r, bi, ldp)] = ai;
        }
    }
    *pos_p = pos;
    *block_num_p = block_num;
    *ldp_p = ldp;
}

ALPHA_INT alpha_range_search(ALPHA_INT a[], ALPHA_INT start, ALPHA_INT end, ALPHA_INT target)
{
    for (ALPHA_INT i = start; i < end; i++)
    {
        if (a[i] == target)
        {
            return (i);
        }
    }
    return -1;
}

void csr_col_partition(const internal_spmat mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
{
    block_partition(&mat->row_data[rs], &mat->row_data[rs+1], mat->col_data, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
}
// void csr_d_col_partition(const spmat_csr_d_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }
// void csr_c_col_partition(const spmat_csr_c_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }
// void csr_z_col_partition(const spmat_csr_z_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }

// void csc_s_row_partition(const spmat_csc_s_t *mat, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->cols_start[cs], &mat->cols_end[cs], mat->row_indx, mat->rows, ce - cs, block_size, pos_p, block_num_p, ldp_p);
// }
// void csc_d_row_partition(const spmat_csc_d_t *mat, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->cols_start[cs], &mat->cols_end[cs], mat->row_indx, mat->rows, ce - cs, block_size, pos_p, block_num_p, ldp_p);
// }
// void csc_c_row_partition(const spmat_csc_c_t *mat, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->cols_start[cs], &mat->cols_end[cs], mat->row_indx, mat->rows, ce - cs, block_size, pos_p, block_num_p, ldp_p);
// }
// void csc_z_row_partition(const spmat_csc_z_t *mat, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->cols_start[cs], &mat->cols_end[cs], mat->row_indx, mat->rows, ce - cs, block_size, pos_p, block_num_p, ldp_p);
// }

static inline void csr_uppercol_trunc(const ALPHA_INT rows, ALPHA_INT cs, ALPHA_INT ce, const ALPHA_INT *old_start, const ALPHA_INT *old_end, const ALPHA_INT *col_indx, ALPHA_INT *new_start, ALPHA_INT *new_end)
{
    if ((!new_start) || (!new_end))
        exit(-1);
    int num_thread = alpha_get_thread_num();
    const ALPHA_INT rs = 0;
    const ALPHA_INT re = ce;

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT r = rs; r < re; r++)
    {
        ALPHA_INT dr = r - rs;
        ALPHA_INT sr = old_start[r];
        ALPHA_INT er = old_end[r+1];
        ALPHA_INT target_cs = alpha_max(cs, r);
        new_start[dr] = alpha_lower_bound(&col_indx[sr], &col_indx[er], target_cs) - col_indx;
        new_end[dr] = alpha_upper_bound(&col_indx[sr], &col_indx[er], ce) - col_indx;
    }
}
void csr_uppercol_truncate(const internal_spmat A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
{
    csr_uppercol_trunc(A->rows, cs, ce, A->row_data, A->row_data, A->col_data, new_start, new_end);
}
// void csr_d_uppercol_truncate(const spmat_csr_d_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
// {
//     csr_uppercol_trunc(A->rows, cs, ce, A->rows_start, A->rows_end, A->col_indx, new_start, new_end);
// }
// void csr_c_uppercol_truncate(const spmat_csr_c_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
// {
//     csr_uppercol_trunc(A->rows, cs, ce, A->rows_start, A->rows_end, A->col_indx, new_start, new_end);
// }
// void csr_z_uppercol_truncate(const spmat_csr_z_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
// {
//     csr_uppercol_trunc(A->rows, cs, ce, A->rows_start, A->rows_end, A->col_indx, new_start, new_end);
// }

static inline void csr_lowercol_trunc(const ALPHA_INT rows, ALPHA_INT cs, ALPHA_INT ce, const ALPHA_INT *old_start, const ALPHA_INT *old_end, const ALPHA_INT *col_indx, ALPHA_INT *new_start, ALPHA_INT *new_end)
{
    if ((!new_start) || (!new_end))
        exit(-1);
    int num_thread = alpha_get_thread_num();
    const ALPHA_INT rs = cs;
    const ALPHA_INT re = rows;

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT r = rs; r < re; r++)
    {
        ALPHA_INT dr = r - rs;
        ALPHA_INT sr = old_start[r];
        ALPHA_INT er = old_end[r+1];
        ALPHA_INT target_ce = alpha_min(ce, r);
        new_start[dr] = alpha_lower_bound(&col_indx[sr], &col_indx[er], cs) - col_indx;
        new_end[dr] = alpha_upper_bound(&col_indx[sr], &col_indx[er], target_ce) - col_indx;
    }
}
void csr_lowercol_truncate(const internal_spmat A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
{
    csr_lowercol_trunc(A->rows, cs, ce, A->row_data, A->row_data, A->col_data, new_start, new_end);
}
// void csr_d_lowercol_truncate(const spmat_csr_d_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
// {
//     csr_lowercol_trunc(A->rows, cs, ce, A->rows_start, A->rows_end, A->col_indx, new_start, new_end);
// }
// void csr_c_lowercol_truncate(const spmat_csr_c_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
// {
//     csr_lowercol_trunc(A->rows, cs, ce, A->rows_start, A->rows_end, A->col_indx, new_start, new_end);
// }
// void csr_z_lowercol_truncate(const spmat_csr_z_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT *new_start, ALPHA_INT *new_end)
// {
//     csr_lowercol_trunc(A->rows, cs, ce, A->rows_start, A->rows_end, A->col_indx, new_start, new_end);
// }

// void bsr_s_col_partition(const spmat_bsr_s_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }
// void bsr_d_col_partition(const spmat_bsr_d_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }
// void bsr_c_col_partition(const spmat_bsr_c_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }
// void bsr_z_col_partition(const spmat_bsr_z_t *mat, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p)
// {
//     block_partition(&mat->rows_start[rs], &mat->rows_end[rs], mat->col_indx, mat->cols, re - rs, block_size, pos_p, block_num_p, ldp_p);
// }
