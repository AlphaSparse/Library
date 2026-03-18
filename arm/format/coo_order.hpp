
#include <stdio.h>
#include <stdlib.h>

#include "alphasparse/spmat.h"

template <typename TYPE>
struct Node {
  ALPHA_INT x;
  ALPHA_INT y;
  TYPE v;
};

template <typename TYPE>
static int cmp(const void *a, const void *b) {
  if (((struct Node<TYPE> *)a)->x ==
      ((struct Node<TYPE> *)b)->x)  //如果两个结构体的x相同，按它们的y值从小到大排列
    return ((struct Node<TYPE> *)a)->y > ((struct Node<TYPE> *)b)->y;
  else
    return ((struct Node<TYPE> *)a)->x > ((struct Node<TYPE> *)b)->x;  // 反之按x从小到大排列
}

template <typename TYPE>
alphasparseStatus_t coo_order(internal_spmat mat) {
  if (mat->ordered) {
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  ALPHA_INT nnz = mat->nnz;
  struct Node<TYPE> *nodes = (struct Node<TYPE> *)malloc(sizeof(struct Node<TYPE>) * nnz);
  for (ALPHA_INT i = 0; i < nnz; i++) {
    nodes[i].x = mat->row_data[i];
    nodes[i].y = mat->col_data[i];
    nodes[i].v = ((TYPE*)(mat->val_data))[i];
  }

  qsort(nodes, nnz, sizeof(nodes[0]), cmp<TYPE>);

  for (ALPHA_INT i = 0; i < nnz; i++) {
    mat->row_data[i] = nodes[i].x;
    mat->col_data[i] = nodes[i].y;
    ((TYPE*)(mat->val_data))[i] = nodes[i].v;
  }
  mat->ordered = true;
  free(nodes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

// #define C_IMPL(ONAME, TYPE)                         \
//   alphasparseStatus_t ONAME(internal_spmat mat) {   \
//     return coo_order_template(mat);                 \
//   }

// C_IMPL(alphasparse_s_create_coo, float);
// C_IMPL(alphasparse_d_create_coo, double);
// C_IMPL(alphasparse_c_create_coo, ALPHA_Complex8);
// C_IMPL(alphasparse_z_create_coo, ALPHA_Complex16);
// #undef C_IMPL