# 测试矩阵集
# 需要先在执行目录下创建两个名为metrics和results的文件夹
# 所选算法alg_num测试所得数据会以csv文件形式存储至./results/目录下
# 	每行数据为：[matrix_name, hip_time, alpha_time, speedup]

# alg_num=1: capellini-spsv
# alg_num=2: cublk
# alg_num=3: nnz-balance

# 设置alg_num=1时，已进行非转置&转置、单元对角线&非单元对角线、左下角&右上角、f64&f32的测试，运算结果正确
# 设置alg_num=2或3时，代码处于实验阶段，支持csr、非转置、非单元对角线、左下角、f64&f32的测试

if [ -f './metrics.txt' ]; then
    rm ./metrics.txt
fi

function get_files {
    local l_files=$(find /public/home/guochengxin_ict/zk/matrix/ -name "*.mtx")
    echo ${l_files[@]}
}

function get_iter_warmup {
    local ret=()
    local l_iter=1
    local l_warmup=0
    ret+=(${l_iter})
    ret+=(${l_warmup})
    echo "${ret[@]}"
}


res=($(get_files))
# 顺序文件路径序列
files=${res[@]}

# 倒序文件路径序列
# files=($(echo ${res[@]} | tac -s ' '))
# files=${files[@]}

res=($(get_iter_warmup))
iter=${res[0]}
warmup=${res[1]}

TIMEOUT=180

date_token=`date +"%m%d%H%M"`
new_filename_token='spsv_test.csv'                    
speedup_file='./results/'${date_token}'_results_'${new_filename_token}
metrics_file='./metrics/'${date_token}'_metrics_'${new_filename_token}    
echo ${speedup_file}
echo ${metrics_file}
# 添加表头
echo "mtx,hip,alpha,speedup," | tee -a ${speedup_file}
for file in ${files}; do
	cur_mat_name=$(basename ${file})
	echo -n ${cur_mat_name}"," | tee -a ${speedup_file}
	{
		timeout --preserve-status --signal=SIGTERM ${TIMEOUT} /public/home/guochengxin_ict/gcx/alphasparse_for_test/build/hip/test/spsv_csr_r_f64_test_metrics \
			--data-file=${file} \
			--diagA=N --fillA=L --transA=N --alg_num=2 \
			--iter=${iter} --warmup=${warmup} \
			--metrics # \
			# --check
			if [ $? -ne 0 ]; then
			echo -n "0,"
		fi
	} | tee -a ${speedup_file}
	echo "" | tee -a ${speedup_file}
done
if [ -f './metrics.txt' ]; then
	mv ./metrics.txt ${metrics_file}
fi

# 清理无效数据行
awk -F',' 'NF > 2 && $0 !~ /,0,/' ${speedup_file} > "${speedup_file}.tmp"
mv "${speedup_file}.tmp" "${speedup_file}"
