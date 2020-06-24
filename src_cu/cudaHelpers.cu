#include "cudaHelpers.cuh"

void cudaAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void cufftAssert(cufftResult_t code, const char *file, int line, bool abort)
{
	if (code != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFTassert: %s %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}



#ifdef _CUFFT_H_
// cuFFT API errors
const char *_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}


#endif
__device__ float d_Lerp(float a, float b, float x)
{
	return a + (b - a) * x;
}