#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;




#include "PseudoRefinement.h"


__declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(int3 dims, float *atomCenters, float *atomWeights, float rAtoms, unsigned int nAtoms) {


	PseudoProjector *proj = new PseudoProjector(dims, atomCenters, atomWeights, rAtoms, nAtoms);

	return proj;
}


__declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj,  float* output, float* output_nrm, float3 angles, float shiftX, float shiftY, unsigned int batch) {
	proj->project_Pseudo(output, output_nrm, angles, shiftX, shiftY, PSEUDO_FORWARD);
}

__declspec(dllexport) void __stdcall GetProjectionCTF(PseudoProjectorPTR proj,  float* output, float* output_nrm,  DOUBLE *gaussTables, DOUBLE *gaussTables2, float3 angles, float shiftX, float shiftY, unsigned int batch) {
	proj->project_PseudoCTF(output, output_nrm, gaussTables, gaussTables2, angles, shiftX, shiftY, PSEUDO_FORWARD);
}

__declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	return proj->ART_multi_Image_step(Iexp, angles, shiftX, shiftY, numImages);
}

__declspec(dllexport) float __stdcall DoARTStepCTF(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	return proj->ART_multi_Image_step(Iexp, angles, gaussTables, gaussTables2, shiftX, shiftY, numImages);
}

__declspec(dllexport) void __stdcall getGaussianTableFull(float * table, DOUBLE sigma, int interpoints) {
	DOUBLE sigma4 = 4 * sigma;
	Matrix1D<DOUBLE> gaussianProjectionTable = Matrix1D<DOUBLE>(CEIL(sigma4*sqrt(2) * interpoints));
	int origin = CEIL(sigma4*sqrt(2) * interpoints) / 2;
	gaussianProjectionTable.coreDeallocate();
	gaussianProjectionTable.vdata = table;
	gaussianProjectionTable.destroyData = false;
	FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
		gaussianProjectionTable(i) = gaussian1D((DOUBLE)(i-origin) / interpoints, sigma);
}

__declspec(dllexport) void __stdcall convolve(DOUBLE * img, DOUBLE * ctf, DOUBLE * outp, int3 dims)
{

	MultidimArray<DOUBLE> ctfMat = MultidimArray<DOUBLE>(dims.z, dims.y, dims.x/2+1);
	MultidimArray<DOUBLE> outMat = MultidimArray<DOUBLE>(1, dims.y, dims.x);
	ctfMat.destroyData = false;
	ctfMat.data = ctf;
	outMat.destroyData = false;
	FourierTransformer ft;
	ft.setThreadsNumber(1);
	MultidimArray<DOUBLE> mat = MultidimArray<DOUBLE>(1, dims.y, dims.x);
	mat.destroyData = false;
	MultidimArray<Complex> matFT = MultidimArray<Complex>(1, dims.y, dims.x / 2 + 1);
	for (size_t k = 0; k < dims.z; k++)
	{
		mat.data = img + k * (dims.x*dims.y);
		outMat.data = outp + k * (dims.x*dims.y);
		ft.FourierTransform(mat, matFT, false);
		FOR_ALL_ELEMENTS_IN_ARRAY2D(matFT)
		{
			auto tmp = A2D_ELEM(matFT, i, j);
			auto tmp2 = A3D_ELEM(ctfMat, k, i, j);
			A2D_ELEM(matFT, i, j) = A2D_ELEM(matFT, i, j) * A3D_ELEM(ctfMat, k, i, j);
		}
		ft.inverseFourierTransform(matFT, outMat);

	}
}

void convolveImage(MultidimArray<DOUBLE> imgMat, MultidimArray<DOUBLE> ctfMat, MultidimArray<DOUBLE> outMat, int3 dims)
{



	FourierTransformer ft;
	ft.setThreadsNumber(1);

	MultidimArray<Complex> matFT = MultidimArray<Complex>(1, dims.y, dims.x / 2 + 1);
	for (size_t k = 0; k < dims.z; k++)
	{
		ft.FourierTransform(imgMat, matFT, false);
		FOR_ALL_ELEMENTS_IN_ARRAY2D(matFT)
		{
			auto tmp = A2D_ELEM(matFT, i, j);
			auto tmp2 = A3D_ELEM(ctfMat, k, i, j);
			A2D_ELEM(matFT, i, j) = A2D_ELEM(matFT, i, j) * A3D_ELEM(ctfMat, k, i, j);
		}
		ft.inverseFourierTransform(matFT, outMat);

	}
}

__declspec(dllexport) float __stdcall DoARTStepMoved(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, float *atomPositions, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector<Matrix1D<DOUBLE>>  prev = proj->atomPosition;

	std::vector<Matrix1D<DOUBLE>>  newAtomPosition = std::vector<Matrix1D<DOUBLE>>();
	newAtomPosition.reserve(prev.size());


	for (size_t i = 0; i < prev.size(); i++)
	{
		Matrix1D<DOUBLE> tmp = Matrix1D<DOUBLE>(3);
		XX(tmp) = atomPositions[i * 3];
		YY(tmp) = atomPositions[i * 3 + 1];
		ZZ(tmp) = atomPositions[i * 3 + 2];
		newAtomPosition.push_back(tmp);

	}

	proj->atomPosition = newAtomPosition;
	float ret = proj->ART_multi_Image_step(Iexp, angles, shiftX, shiftY, numImages);
	proj->atomPosition = prev;
	return ret;
}

__declspec(dllexport) float __stdcall DoARTStepMovedCTF(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, float *atomPositions, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector<Matrix1D<DOUBLE>>  prev = proj->atomPosition;

	std::vector<Matrix1D<DOUBLE>>  newAtomPosition = std::vector<Matrix1D<DOUBLE>>();
	newAtomPosition.reserve(prev.size());


	for (size_t i = 0; i < prev.size(); i++)
	{
		Matrix1D<DOUBLE> tmp = Matrix1D<DOUBLE>(3);
		XX(tmp) = atomPositions[i * 3];
		YY(tmp) = atomPositions[i * 3 + 1];
		ZZ(tmp) = atomPositions[i * 3 + 2];
		newAtomPosition.push_back(tmp);

	}

	proj->atomPosition = newAtomPosition;
	float ret = proj->ART_multi_Image_step(Iexp, angles, GaussTables, GaussTables2, shiftX, shiftY, numImages);
	proj->atomPosition = prev;
	return ret;
}

__declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, float* outp) {
	std::copy(proj->atomWeight.begin(), proj->atomWeight.begin() + proj->atomWeight.size(), outp);

}
