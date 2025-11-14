


#ifndef INCLUDED_CUDA_STREAM
#define INCLUDED_CUDA_STREAM

#pragma once

#include <hip/hip_runtime.h>

#include "unique_handle.h"


namespace CU
{
	struct StreamDestroyDeleter
	{
		void operator ()(hipStream_t stream) const
		{
			hipStreamDestroy(stream);
		}
	};
	
	using unique_stream = unique_handle<hipStream_t, nullptr, StreamDestroyDeleter>;
	
	unique_stream createStream(unsigned int flags = hipStreamDefault);
}

#endif  // INCLUDED_CUDA_STREAM
