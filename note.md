# 测试环境
sugon-419

## 硬件
- CPU: Hygon C86 7185 32-core Processor

- 内存：DDR4 (8*16GB, 2666MHz, 8 channels)

- 加速器：Hygon DCU (64 CU, 16GB MHB2)

## 软件
- OS: Red Hat

- ROCm version: 5.4.3

- HIP version: 5.4.23453 

- packageDir: /public/software/compiler/dtk/dtk-23.10.1/llvm/bin

- cmake version 3.24.4

- gcc version: 7.3.1

# 编译项目
- NV GPU
```shell
mkdir build
cd build
cmake .. -DALPHA_BUILD_CUDA=ON
make
```

- sugon-419
```shell
mkdir build
cd build
cmake .. -DALPHA_BUILD_HIP=ON
make
```

# spsv测试执行
进入`build`目录

- NV GPU
```shell
sh ../cuda/test/test_spsv.sh
```

- sugon-419
```shell
sh ../hip/test/test_spsv.sh
```
# 矩阵集路径
`../matrix_test`
