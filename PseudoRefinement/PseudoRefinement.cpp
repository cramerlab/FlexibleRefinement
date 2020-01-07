#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;




#include "PseudoRefinement.h"


__declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(float* projections, float* angles, int3 dims, float *atomCenters, float *atomWeights, float rAtoms, unsigned int nAtoms) {


	PseudoProjector *proj = new PseudoProjector(dims, atomCenters, atomWeights, rAtoms, nAtoms);
	MultidimArray< float> IExp = MultidimArray<float>(dims.z, dims.y, dims.x);
	memcpy(IExp.data, projections, dims.x*dims.y*dims.z * sizeof(float));
	return proj;
}


__declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj,  float* output, float* output_nrm, float3 angles, float shiftX, float shiftY, unsigned int batch) {


	proj->project_Pseudo(output, output_nrm, angles, shiftX, shiftY, PSEUDO_FORWARD);

}

__declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	return proj->ART_multi_Image_step(Iexp, angles, shiftX, shiftY, numImages);
}

__declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, float* outp) {
	std::copy(proj->atomWeight.begin(), proj->atomWeight.begin() + proj->atomWeight.size(), outp);

}
