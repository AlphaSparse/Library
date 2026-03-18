.PHONY: configure build clean

build: configure
	cmake --build build -j

configure:
	cmake -B build  -DALPHA_BUILD_HYGON=1 -DALPHA_BUILD_MKL_ENABLE=1
clean:
	rm -rf build