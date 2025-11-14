#pragma once
#include <vector>
#include "stream.h"
#include "stdio.h"

namespace spECK {
    // get device attributes for best performance and creates cudaStreams
    struct spECKConfig {
        int sm;
        int maxStaticSharedMemoryPerBlock;
        int maxDynamicSharedMemoryPerBlock;
        std::vector<hipStream_t> streams;
        hipEvent_t completeStart = 0, completeEnd = 0, individualStart = 0, individualEnd = 0;

        static spECKConfig initialize(int cudaDeviceNumber) {
			spECKConfig config;
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, cudaDeviceNumber);
            config.sm = prop.multiProcessorCount;
            config.maxStaticSharedMemoryPerBlock = prop.sharedMemPerBlock;
            config.maxDynamicSharedMemoryPerBlock = prop.sharedMemPerBlock;

            for (int i = 0; i < 6; i++) {
                config.streams.push_back(0);
                hipStreamCreate(&config.streams[i]);
            }
            hipEventCreate(&config.completeStart);
            hipEventCreate(&config.completeEnd);
            hipEventCreate(&config.individualStart);
            hipEventCreate(&config.individualEnd);
            return config;
        }

        void cleanup() {
            for (auto s : streams) {
                hipStreamDestroy(s);
            }
            hipEventDestroy(completeStart);
            hipEventDestroy(completeEnd);
            hipEventDestroy(individualStart);
            hipEventDestroy(individualEnd);
            streams.clear();
        }

        ~spECKConfig() {
            // cleanup();
        }

    private:
		spECKConfig() {

        }
    };
}