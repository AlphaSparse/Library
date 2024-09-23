#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_add_c_gebsr(const spmat_gebsr_c_t *A, const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *B, spmat_gebsr_c_t **C);
alphasparse_status_t dcu_add_c_gebsr_trans(const spmat_gebsr_c_t *A, const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *B, spmat_gebsr_c_t **C);
alphasparse_status_t dcu_add_c_gebsr_conj(const spmat_gebsr_c_t *A, const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *B, spmat_gebsr_c_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t dcu_gemv_c_gebsr(alphasparse_dcu_handle_t handle,
                                     alphasparse_layout_t dir,
                                     ALPHA_INT mb,
                                     ALPHA_INT nb,
                                     ALPHA_INT nnzb,
                                     const ALPHA_Complex8 alpha,
                                     const ALPHA_Complex8 *bsr_val,
                                     const ALPHA_INT *bsr_row_ptr,
                                     const ALPHA_INT *bsr_col_ind,
                                     ALPHA_INT row_block_dim,
                                     ALPHA_INT col_block_dim,
                                     const ALPHA_Complex8 *x,
                                     const ALPHA_Complex8 beta,
                                     ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t dcu_gemv_c_gebsr_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t dcu_gemv_c_gebsr_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_n_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_u_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_n_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_u_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D+L')*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t dcu_symv_c_gebsr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_n_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_u_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_n_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_u_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D+L')^T*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')^T*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)^T*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)^T*x + beta*y
alphasparse_status_t dcu_hermv_c_gebsr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_n_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_u_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_n_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_u_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_c_gebsr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*D*x + beta*y
alphasparse_status_t dcu_diagmv_c_gebsr_n(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*x + beta*y
alphasparse_status_t dcu_diagmv_c_gebsr_u(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t dcu_gemm_c_gebsr(alphasparse_dcu_handle_t handle,
                                     alphasparse_layout_t dir,
                                     ALPHA_INT mb,
                                     ALPHA_INT n,
                                     ALPHA_INT kb,
                                     ALPHA_INT nnzb,
                                     const ALPHA_Complex8 alpha,
                                     const ALPHA_Complex8 *bsr_val,
                                     const ALPHA_INT *bsr_row_ptr,
                                     const ALPHA_INT *bsr_col_ind,
                                     ALPHA_INT row_block_dim,
                                     ALPHA_INT col_block_dim,
                                     const ALPHA_Complex8 *B,
                                     ALPHA_INT ldb,
                                     const ALPHA_Complex8 beta,
                                     ALPHA_Complex8 *C,
                                     ALPHA_INT ldc);

alphasparse_status_t dcu_gemm_c_gebsr_transA(alphasparse_dcu_handle_t handle,
                                            alphasparse_layout_t dir,
                                            ALPHA_INT mb,
                                            ALPHA_INT n,
                                            ALPHA_INT kb,
                                            ALPHA_INT nnzb,
                                            const ALPHA_Complex8 alpha,
                                            const ALPHA_Complex8 *bsr_val,
                                            const ALPHA_INT *bsr_row_ptr,
                                            const ALPHA_INT *bsr_col_ind,
                                            ALPHA_INT row_block_dim,
                                            ALPHA_INT col_block_dim,
                                            const ALPHA_Complex8 *B,
                                            ALPHA_INT ldb,
                                            const ALPHA_Complex8 beta,
                                            ALPHA_Complex8 *C,
                                            ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemm_c_gebsr_transB(alphasparse_dcu_handle_t handle,
                                            alphasparse_layout_t dir,
                                            ALPHA_INT mb,
                                            ALPHA_INT n,
                                            ALPHA_INT kb,
                                            ALPHA_INT nnzb,
                                            const ALPHA_Complex8 alpha,
                                            const ALPHA_Complex8 *bsr_val,
                                            const ALPHA_INT *bsr_row_ptr,
                                            const ALPHA_INT *bsr_col_ind,
                                            ALPHA_INT row_block_dim,
                                            ALPHA_INT col_block_dim,
                                            const ALPHA_Complex8 *B,
                                            ALPHA_INT ldb,
                                            const ALPHA_Complex8 beta,
                                            ALPHA_Complex8 *C,
                                            ALPHA_INT ldc);

alphasparse_status_t dcu_gemm_c_gebsr_transAB(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             ALPHA_INT mb,
                                             ALPHA_INT n,
                                             ALPHA_INT kb,
                                             ALPHA_INT nnzb,
                                             const ALPHA_Complex8 alpha,
                                             const ALPHA_Complex8 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT row_block_dim,
                                             ALPHA_INT col_block_dim,
                                             const ALPHA_Complex8 *B,
                                             ALPHA_INT ldb,
                                             const ALPHA_Complex8 beta,
                                             ALPHA_Complex8 *C,
                                             ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemm_c_gebsr_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t dcu_gemm_c_gebsr_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_c_gebsr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparse_status_t dcu_hermm_c_gebsr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_c_gebsr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t dcu_diagmm_c_gebsr_n_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t dcu_diagmm_c_gebsr_u_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t dcu_diagmm_c_gebsr_n_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t dcu_diagmm_c_gebsr_u_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t dcu_spmmd_c_gebsr_row(const spmat_gebsr_c_t *matA, const spmat_gebsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t dcu_spmmd_c_gebsr_col(const spmat_gebsr_c_t *matA, const spmat_gebsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_c_gebsr_row_trans(const spmat_gebsr_c_t *matA, const spmat_gebsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_c_gebsr_col_trans(const spmat_gebsr_c_t *matA, const spmat_gebsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

// A^T*B
alphasparse_status_t dcu_spmmd_c_gebsr_row_conj(const spmat_gebsr_c_t *matA, const spmat_gebsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_c_gebsr_col_conj(const spmat_gebsr_c_t *matA, const spmat_gebsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

alphasparse_status_t dcu_spmm_c_gebsr(const spmat_gebsr_c_t *A, const spmat_gebsr_c_t *B, spmat_gebsr_c_t **C);
alphasparse_status_t dcu_spmm_c_gebsr_trans(const spmat_gebsr_c_t *A, const spmat_gebsr_c_t *B, spmat_gebsr_c_t **C);
alphasparse_status_t dcu_spmm_c_gebsr_conj(const spmat_gebsr_c_t *A, const spmat_gebsr_c_t *B, spmat_gebsr_c_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t dcu_trsv_c_gebsr_n_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L)*x
alphasparse_status_t dcu_trsv_c_gebsr_u_lo(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparse_status_t dcu_trsv_c_gebsr_n_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparse_status_t dcu_trsv_c_gebsr_u_hi(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_c_gebsr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(D)*x
alphasparse_status_t dcu_diagsv_c_gebsr_n(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*x
alphasparse_status_t dcu_diagsv_c_gebsr_u(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_c_gebsr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t dcu_diagsm_c_gebsr_n_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t dcu_diagsm_c_gebsr_u_row(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t dcu_diagsm_c_gebsr_n_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t dcu_diagsm_c_gebsr_u_col(const ALPHA_Complex8 alpha, const spmat_gebsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

alphasparse_status_t dcu_set_value_c_gebsr(spmat_gebsr_c_t *A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex8 value);