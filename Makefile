# --- 用户配置区 ---
ALPHA_BUILD_CUDA ?= OFF
ALPHA_BUILD_HIP ?= ON
ALPHA_BUILD_HYGON ?= OFF
ALPHA_BUILD_ARM ?= OFF
ALPHA_BUILD_PLAIN ?= OFF

# --- 内部逻辑区 ---

BUILD_DIR := build

CMAKE_ARGS := \
	-DALPHA_BUILD_CUDA=${ALPHA_BUILD_CUDA} \
	-DALPHA_BUILD_HIP=${ALPHA_BUILD_HIP} \
	-DALPHA_BUILD_HYGON=${ALPHA_BUILD_HYGON} \
	-DALPHA_BUILD_ARM=${ALPHA_BUILD_ARM} \
	-DALPHA_BUILD_PLAIN=${ALPHA_BUILD_PLAIN}

ENV_CMD :=

.PHONY: all build configure clean

all: build

build: $(BUILD_DIR)/Makefile
	@echo "--- Executing Build with Direct Flag Injection ---"
	$(ENV_CMD) cmake --build $(BUILD_DIR) -j

$(BUILD_DIR)/Makefile: CMakeLists.txt
	@echo "--- Configuring with Direct Flag Injection ---"
	$(ENV_CMD) cmake -B $(BUILD_DIR) $(CMAKE_ARGS)

configure:
	@echo "--- Forcing Re-Configuration with Direct Flag Injection ---"
	$(ENV_CMD) cmake -B $(BUILD_DIR) $(CMAKE_ARGS)

clean:
	@echo "--- Cleaning Build Directory ---"
	rm -rf $(BUILD_DIR)