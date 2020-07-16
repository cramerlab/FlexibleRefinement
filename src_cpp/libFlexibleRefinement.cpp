#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;




#include "libFlexibleRefinement.h"


__declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(int3 dims, RDOUBLE *atomCenters, RDOUBLE *atomWeights, RDOUBLE rAtoms, unsigned int nAtoms) {


	PseudoProjector *proj = new PseudoProjector(dims, atomCenters, atomWeights, rAtoms, nAtoms);

	return proj;
}


__declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj, RDOUBLE* output, RDOUBLE* output_nrm, float3 angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int batch) {
	proj->project_Pseudo(output, output_nrm, angles, shiftX, shiftY, PSEUDO_FORWARD);
}

__declspec(dllexport) void __stdcall GetProjectionCTF(PseudoProjectorPTR proj, RDOUBLE* output, RDOUBLE* output_nrm,  RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE border, float3 angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int batch) {
	proj->project_PseudoCTF(output, output_nrm, gaussTables, gaussTables2, border, angles, shiftX, shiftY, PSEUDO_FORWARD);
}

__declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	return proj->ART_multi_Image_step(Iexp, angles, shiftX, shiftY, numImages);
}

__declspec(dllexport) float __stdcall DoARTStepCTF(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	return proj->ART_multi_Image_step(Iexp, angles, gaussTables, gaussTables2, border, shiftX, shiftY, numImages);
}

__declspec(dllexport) void __stdcall getGaussianTableFull(RDOUBLE * table, RDOUBLE sigma, int interpoints) {
	RDOUBLE sigma4 = 4 * sigma;
	Matrix1D<RDOUBLE> gaussianProjectionTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(2) * interpoints));
	int origin = CEIL(sigma4*sqrt(2) * interpoints) / 2;
	gaussianProjectionTable.coreDeallocate();
	gaussianProjectionTable.vdata = table;
	gaussianProjectionTable.destroyData = false;
	FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
		gaussianProjectionTable(i) = gaussian1D((RDOUBLE)(i-origin) / interpoints, sigma);
}

__declspec(dllexport) void __stdcall convolve(RDOUBLE * img, RDOUBLE * ctf, RDOUBLE * outp, int3 dims)
{

	MultidimArray<RDOUBLE> ctfMat = MultidimArray<RDOUBLE>(dims.z, dims.y, dims.x/2+1);
	MultidimArray<RDOUBLE> outMat = MultidimArray<RDOUBLE>(1, dims.y, dims.x);
	ctfMat.destroyData = false;
	ctfMat.data = ctf;
	outMat.destroyData = false;
	FourierTransformer ft;
	ft.setThreadsNumber(1);
	MultidimArray<RDOUBLE> mat = MultidimArray<RDOUBLE>(1, dims.y, dims.x);
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

void convolveImage(MultidimArray<RDOUBLE> imgMat, MultidimArray<RDOUBLE> ctfMat, MultidimArray<RDOUBLE> outMat, int3 dims)
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

__declspec(dllexport) float __stdcall DoARTStepMoved(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE *atomPositions, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	std::vector<Matrix1D<RDOUBLE>>  prev = proj->atomPositions;

	std::vector<Matrix1D<RDOUBLE>>  newAtomPosition = std::vector<Matrix1D<RDOUBLE>>();
	newAtomPosition.reserve(prev.size());


	for (size_t i = 0; i < prev.size(); i++)
	{
		Matrix1D<RDOUBLE> tmp = Matrix1D<RDOUBLE>(3);
		XX(tmp) = atomPositions[i * 3];
		YY(tmp) = atomPositions[i * 3 + 1];
		ZZ(tmp) = atomPositions[i * 3 + 2];
		newAtomPosition.push_back(tmp);

	}

	proj->atomPositions = newAtomPosition;
	float ret = proj->ART_multi_Image_step(Iexp, angles, shiftX, shiftY, numImages);
	proj->atomPositions = prev;
	return ret;
}

__declspec(dllexport) float __stdcall DoARTStepMovedCTF(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE *atomPositions, RDOUBLE * GaussTables, RDOUBLE * GaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	std::vector<Matrix1D<RDOUBLE>>  prev = proj->atomPositions;

	std::vector<Matrix1D<RDOUBLE>>  newAtomPosition = std::vector<Matrix1D<RDOUBLE>>();
	newAtomPosition.reserve(prev.size());


	for (size_t i = 0; i < prev.size(); i++)
	{
		Matrix1D<RDOUBLE> tmp = Matrix1D<RDOUBLE>(3);
		XX(tmp) = atomPositions[i * 3];
		YY(tmp) = atomPositions[i * 3 + 1];
		ZZ(tmp) = atomPositions[i * 3 + 2];
		newAtomPosition.push_back(tmp);

	}

	proj->atomPositions = newAtomPosition;
	float ret = proj->ART_multi_Image_step(Iexp, angles, GaussTables, GaussTables2, border, shiftX, shiftY, numImages);
	proj->atomPositions = prev;
	return ret;
}


__declspec(dllexport) float __stdcall DoARTStepMovedCTF_DB(PseudoProjectorPTR proj, RDOUBLE * Iexp, RDOUBLE * Itheo, RDOUBLE * Icorr, RDOUBLE * Idiff, float3 * angles, RDOUBLE *atomPositions, RDOUBLE * GaussTables, RDOUBLE * GaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	std::vector<Matrix1D<RDOUBLE>>  prev = proj->atomPositions;

	std::vector<Matrix1D<RDOUBLE>>  newAtomPosition = std::vector<Matrix1D<RDOUBLE>>();
	newAtomPosition.reserve(prev.size());


	for (size_t i = 0; i < prev.size(); i++)
	{
		Matrix1D<RDOUBLE> tmp = Matrix1D<RDOUBLE>(3);
		XX(tmp) = atomPositions[i * 3];
		YY(tmp) = atomPositions[i * 3 + 1];
		ZZ(tmp) = atomPositions[i * 3 + 2];
		newAtomPosition.push_back(tmp);

	}

	proj->atomPositions = newAtomPosition;
	float ret = proj->ART_multi_Image_step_DB(Iexp, Itheo, Icorr, Idiff, angles, GaussTables, GaussTables2, border, shiftX, shiftY, numImages);
	proj->atomPositions = prev;
	return ret;
}

__declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, RDOUBLE* outp) {
	std::copy(proj->atomWeight.begin(), proj->atomWeight.begin() + proj->atomWeight.size(), outp);

}
