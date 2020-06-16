#include "Warp_GPU.h"
#include <cassert>
void ResizeMapGPU(MultidimArray<float> &img, int3 newDim)
{
	float* dout = MallocDevice(newDim.x*newDim.y*newDim.z * sizeof(float));
	float * din = MallocDeviceFromHost(img.data, img.nzyxdim);

	Scale(din, dout, make_int3(img.xdim, img.ydim, img.zdim), newDim, 1, 0, 0, NULL, NULL);
	img.resizeNoCopy(newDim.z, newDim.y, newDim.x);
	CopyDeviceToHost(dout, img.data, img.nzyxdim);
	FreeDevice(din);
	FreeDevice(dout);
}


void ResizeMapGPU(MultidimArray<float> &img, int2 newDim)
{
	float* dout = MallocDevice(newDim.x*newDim.y*img.zdim * sizeof(float));
	float* din = MallocDeviceFromHost(img.data, img.nzyxdim);

	Scale(din, dout, make_int3(img.xdim, img.ydim, 1), gtom::toInt3(newDim), img.zdim, 0, 0, NULL, NULL);
	img.resizeNoCopy(img.zdim, newDim.y, newDim.x);
	CopyDeviceToHost(dout, img.data, img.nzyxdim);
	FreeDevice(din);
	FreeDevice(dout);
}




void Substract_GPU(MultidimArray<float> &img, MultidimArray<float> &substrahend) {
	assert(substrahend.nzyxdim == img.nzyxdim && "Shapes should match");
	float* din = MallocDeviceFromHost(img.data, img.nzyxdim);
	float* dSubtrahends = MallocDeviceFromHost(substrahend.data, substrahend.nzyxdim);

	SubtractFromSlices(din, dSubtrahends, din, img.nzyxdim, 1);
}