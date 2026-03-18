
./cuda/test/spsv_csr_r_f64_test_metrics --data-file=../matrix_test/mhd1280b.mtx --transA=N --fillA=L --diagA=N --iter=10 --warmup=2 --check --metrics --alg_num=8

# alg_num=1: capellini-spsv
# alg_num=2: capellini-spsv + row_map
# alg_num=3: rocsparse (1warp/1row + row_map)
# alg_num=4: smblk (1vector/1row + row_map + (blknum = n * #sm))
# alg_num=5: smblk_rearrange (1warp/1row + row_map + (blknum = n * #sm) + rearrange)
# alg_num=6: yuenyeung
# alg_num=7: nv2011

