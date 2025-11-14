mkdir -p log/spmm

#better submit separately by sbatch
#parameters: matb_layout, data_type, lib_name, alg_num, N
# python3 test_utils/spmm/test_perf.py row r_f32 hip 1 4
# python3 test_utils/spmm/test_perf.py row r_f32 hip 2 4
# python3 test_utils/spmm/test_perf.py row r_f32 hip 3 4
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 1 4
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 2 4
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 3 4
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 4 4
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 5 4

# python3 test_utils/spmm/test_perf.py row r_f32 hip 1 16
# python3 test_utils/spmm/test_perf.py row r_f32 hip 2 16
# python3 test_utils/spmm/test_perf.py row r_f32 hip 3 16
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 1 16
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 2 16
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 3 16
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 4 16
# python3 test_utils/spmm/test_perf.py row r_f32 alpha 5 16

python3 test_utils/spmm/test_perf.py row r_f64 hip 1 64
python3 test_utils/spmm/test_perf.py row r_f64 hip 2 64
python3 test_utils/spmm/test_perf.py row r_f64 hip 3 64
python3 test_utils/spmm/test_perf.py row r_f64 alpha 1 64
python3 test_utils/spmm/test_perf.py row r_f64 alpha 2 64
python3 test_utils/spmm/test_perf.py row r_f64 alpha 3 64
python3 test_utils/spmm/test_perf.py row r_f64 alpha 4 64
python3 test_utils/spmm/test_perf.py row r_f64 alpha 5 64

python3 test_utils/spmm/test_perf.py row r_f64 hip 1 128
python3 test_utils/spmm/test_perf.py row r_f64 hip 2 128
python3 test_utils/spmm/test_perf.py row r_f64 hip 3 128
python3 test_utils/spmm/test_perf.py row r_f64 alpha 1 128
python3 test_utils/spmm/test_perf.py row r_f64 alpha 2 128
python3 test_utils/spmm/test_perf.py row r_f64 alpha 3 128
python3 test_utils/spmm/test_perf.py row r_f64 alpha 4 128
python3 test_utils/spmm/test_perf.py row r_f64 alpha 5 128

#merge result
#need pandas lib in python
# python3 test_utils/spmm/collect ./log/spmm/ row r_f32 4
# python3 test_utils/spmm/collect ./log/spmm/ row r_f32 16
python3 test_utils/spmm/collect ./log/spmm/ row r_f64 64
python3 test_utils/spmm/collect ./log/spmm/ row r_f64 128