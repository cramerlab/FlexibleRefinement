


#include "PseudoProjector.h"
#include <filesystem>
#include "cxxopts.hpp"
#include <string>
#include "metadata_table.h"
#include "readMRC.h"
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>
#include <cuda_runtime_api.h>
#include "gpu_project.cuh"
#include "Projector.h"
#define WRITE_PROJECTIONS

namespace fs = std::filesystem;
enum Algo { ART = 0, SIRT = 1 };

struct params {

	std::string inputPDB;
	std::string inputStar;
};
void defineParams(cxxopts::Options &options)
{

	options.add_options()
		("i,input", "Input", cxxopts::value<std::string>(), "Input PDB file of atoms")
		("o,output", "rootname", cxxopts::value<std::string>(), "Rootname for output")
		("sigma", "s", cxxopts::value<RDOUBLE>()->default_value("1.5"), "Sigma of Gaussians used")
		("initialSeeds", "N", cxxopts::value<size_t>()->default_value("300"), "Initial number of Atoms")
		("growSeeds", "percentage", cxxopts::value<size_t>()->default_value("30"), "Percentage of growth, At each iteration the smallest percentage/2 pseudoatoms will be removed, and percentage new pseudoatoms will be created.")
		("filterInput", "f", cxxopts::value<RDOUBLE>(), "Low-pass filter input using this threshold")
		("stop", "p", cxxopts::value<RDOUBLE>()->default_value("0.001"), "Stop criterion (0<p<1) for inner iterations. At each iteration the current number of gaussians will be optimized until the average error does not decrease at least this amount relative to the previous iteration.")
		("targetError", "p", cxxopts::value<RDOUBLE>()->default_value("0.02"), "Finish when the average representation error is below this threshold (in percentage; by default, 2%)")
		("dontAllowMovement", "true", cxxopts::value<bool>()->default_value("false"), "Don't allow pseudoatoms to move")
		("dontAllowIntensity", "f", cxxopts::value<RDOUBLE>()->default_value("0.01"), "Don't allow pseudoatoms to change intensity. f determines the fraction of intensity")
		("intensityColumn", "s", cxxopts::value<std::string>()->default_value("Bfactor"), "Where to write the intensity in the PDB file")
		("Nclosest", "N", cxxopts::value<size_t>()->default_value("3"), "N closest atoms, it is used only for the distance histogram")
		("minDistance", "d", cxxopts::value<RDOUBLE>()->default_value("0.001"), "Minimum distance between two pseudoatoms (in Angstroms). Set it to -1 to disable")
		("penalty", "p", cxxopts::value<RDOUBLE>()->default_value("10"), "Penalty for overshooting")
		("sampling_rate", "Ts", cxxopts::value<RDOUBLE>()->default_value("1"), "Sampling rate Angstroms/pixel")
		("dontScale", "true", cxxopts::value<bool>()->default_value("false"), "Don't scale atom weights in the PDB")
		("binarize", "threshold", cxxopts::value<RDOUBLE>()->default_value("0.5"), "Binarize the volume")
		("thr", "t", cxxopts::value<size_t>()->default_value("1"), "Number of Threads")
		("mask", "mask_type", cxxopts::value<std::string>(), "Which mask type to use. Options are real_file and binary_file")
		("maskfile", "f", cxxopts::value<std::string>(), "Path of mask file")
		("center", "c", cxxopts::value<std::vector<RDOUBLE>>()->default_value("0,0,0"), "Center of Mask")
		("v,verbose", "v", cxxopts::value<int>()->default_value("0"), "Verbosity Level");
}

params readParams(cxxopts::ParseResult &res) {
	
	return { "","" };

}


void weightProjections(MultidimArray<RDOUBLE> &projections, float3 * angles, int3 dims)
{
	int maxR = dims.x / 2;
	float* Weights = new float[maxR];

	for (int r = 0; r < maxR; r++)
	{
		float Sum = 0;

		float3 Point = { (float)r, 0, 0 };

		for (int a = 0; a < projections.zdim; a++)
		{
			glm::vec3 Normal = glm::transpose(Matrix3Euler(angles[a])) * glm::vec3(0, 0, 1);
			float Dist = abs(Normal.x*Point.x + Normal.y*Point.y + Normal.z*Point.z);
			Sum += std::max((float)0, 1 - Dist);
		}

		Weights[r] = 1.0f / std::max(1.0f, Sum);
	}

	MultidimArray<float> Weights2D(dims.y, dims.x / 2 + 1);

	int i = 0;
	for (int y = 0; y < dims.y; y++)
	{
		int yy = y < dims.y / 2 + 1 ? y : y - dims.y;

		for (int x = 0; x < dims.x / 2 + 1; x++)
		{
			int xx = x;
			float r = (float)sqrt(xx * xx + yy * yy);
			Weights2D.data[i++] = Weights[std::min(maxR - 1, (int)std::round(r))];
		}
	}

	if(false){
		MRCImage<float> weightIm(Weights2D);
		weightIm.writeAs<float>("D:\\EMD\\9233\\TomoReconstructions\\projectionsWeighting_weights2D.mrc");
	}
	idxtype batch = 1024;

	int3 sliceDim = { projections.xdim, projections.ydim, 1 };
	int3 sliceFTDim = { projections.xdim / 2 + 1, projections.ydim, 1 };
	int sliceFTElements = sliceFTDim.x*sliceFTDim.y;
	int ndims = DimensionCount(sliceDim);
	int sliceElements = projections.xdim * projections.ydim;
	for (int idx = 0; idx < projections.zdim;) {
		int batchElements = std::min(projections.zdim - idx, batch);
		float * d_projections;
		cudaErrchk(cudaMalloc(&d_projections, Elements(sliceDim)*batchElements * sizeof(float)));

		cudaErrchk(cudaMemcpy(d_projections, projections.data + sliceElements * idx, projections.yxdim*batchElements*sizeof(float), cudaMemcpyHostToDevice));
		float * d_weights = MallocDeviceFromHost(Weights2D.data, Weights2D.yxdim);
		tcomplex * d_fft;
		cudaErrchk(cudaMalloc(&d_fft, sliceFTElements*batchElements * sizeof(*d_fft)));
		d_FFTR2C(d_projections, d_fft, ndims, sliceDim, batchElements);
		if(false){
			MultidimArray<float> fft(batchElements, dims.y, dims.x / 2 + 1);
			tcomplex * fft_complex = (tcomplex*)malloc(sliceFTElements*sizeof(tcomplex)*batchElements);
			cudaErrchk(cudaMemcpy(fft_complex, d_fft, sizeof(tcomplex)*sliceFTElements*batchElements, cudaMemcpyDeviceToHost));
			for (size_t i = 0; i < sliceFTElements*batchElements; i++)
			{
				fft.data[i] = fft_complex[i].x;
			}
			MRCImage<float> fftIm(fft);
			fftIm.writeAs<float>("D:\\EMD\\9233\\TomoReconstructions\\projectionsWeighting_projectionsFFT.mrc", true);
		}
		d_ComplexMultiplyByVector(d_fft, d_weights, d_fft, sliceFTElements, batchElements);
		if (false) {
			MultidimArray<float> fft(batchElements, dims.y, dims.x / 2 + 1);
			tcomplex * fft_complex = (tcomplex*)malloc(sliceFTElements * sizeof(tcomplex)*batchElements);
			cudaErrchk(cudaMemcpy(fft_complex, d_fft, sizeof(tcomplex)*sliceFTElements*batchElements, cudaMemcpyDeviceToHost));
			for (size_t i = 0; i < sliceFTElements*batchElements; i++)
			{
				fft.data[i] = fft_complex[i].x;
			}
			MRCImage<float> fftIm(fft);
			fftIm.writeAs<float>("D:\\EMD\\9233\\TomoReconstructions\\projectionsWeighting_projectionsFFT_weighted.mrc", true);
		}
		d_OwnIFFTC2R(d_fft, d_projections, ndims, sliceDim, batchElements, true);
		if (false) {
			MultidimArray<float> projTemp(batchElements, dims.y, dims.x);

			cudaErrchk(cudaMemcpy(projTemp.data, d_projections, sizeof(float)*sliceElements*batchElements, cudaMemcpyDeviceToHost));

			MRCImage<float> projIm(projTemp);
			projIm.writeAs<float>("D:\\EMD\\9233\\TomoReconstructions\\projectionsWeighting_projectionsWeighted.mrc", true);
		}

		cudaErrchk(cudaMemcpy(projections.data+ sliceElements*idx, d_projections, sliceElements*batchElements * sizeof(float), cudaMemcpyDeviceToHost));
		cudaErrchk(cudaFree(d_fft));
		cudaErrchk(cudaFree(d_projections));
		cudaErrchk(cudaFree(d_weights));
		idx += batchElements;
	}

}



idxtype readProjections(FileName starFileName, MultidimArray<RDOUBLE> &projections, float3 **angles, bool shuffle=false)
{
	MetaDataTable MD;
	try {
		long ret = MD.read(starFileName);
	}
	catch (RelionError Err) {
		std::cout << "Could not read file" << std::endl << Err.msg << std::endl;
		exit(EXIT_FAILURE);
	}
	idxtype numProj = MD.numberOfObjects();
	*angles = (float3 *)malloc(sizeof(float3)*numProj);
	auto rng = std::default_random_engine{};
	std::vector<idxtype> idxLookup;
	idxLookup.reserve(numProj);
	for (idxtype i = 0; i < numProj; i++)
		idxLookup.emplace_back(i);
	if(shuffle)
	    std::shuffle(std::begin(idxLookup), std::end(idxLookup), rng);
	
	{// read projections from file
		bool isInit = false;
		FileName imageName;
		FileName prevImageName = "";
		char imageName_cstr[1000];
		MRCImage<RDOUBLE> im;
		idxtype idx = 0;
		int num;
		FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD) {
			MD.getValue(EMDL_IMAGE_NAME, imageName);
			idxtype randomI = idxLookup[idx];	// Write to random position in projections to avoid any bias
			MD.getValue(EMDL_ORIENT_ROT, (*angles)[randomI].x);
			MD.getValue(EMDL_ORIENT_TILT, (*angles)[randomI].y);
			MD.getValue(EMDL_ORIENT_PSI, (*angles)[randomI].z);
			(*angles)[randomI].x = ToRad((*angles)[randomI].x);
			(*angles)[randomI].y = ToRad((*angles)[randomI].y);
			(*angles)[randomI].z = ToRad((*angles)[randomI].z);
			sscanf(imageName.c_str(), "%d@%s", &num, imageName_cstr);
			imageName = imageName_cstr;
			if (imageName != prevImageName) {
				im = MRCImage<RDOUBLE>::readAs(imageName);
				if (!isInit) {
					projections.resize(numProj, im().ydim, im().xdim);
					isInit = true;
				}
			}
			prevImageName = imageName;
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(im()) {
				DIRECT_A3D_ELEM(projections, randomI, i, j) = im(num - 1, i, j);
			}
			idx++;
		}
	}
	return numProj;
}

void multiplyByCtf(MultidimArray<float>& projections, MultidimArray<float> ctf) {
	idxtype batch = 1024;
	int3 sliceDim = { projections.xdim, projections.ydim, 1 };
	int2 projDim = { projections.xdim, projections.ydim };
	int3 sliceFTDim = { projections.xdim / 2 + 1, projections.ydim, 1 };
	int sliceFTElements = sliceFTDim.x*sliceFTDim.y;
	int ndims = DimensionCount(sliceDim);
	int sliceElements = projections.xdim * projections.ydim;

	tfloat3 * forward_shifts = (tfloat3 *)malloc(batch * sizeof(*forward_shifts));
	tfloat3 * back_shifts = (tfloat3 *)malloc(batch * sizeof(*forward_shifts));
	for (size_t i = 0; i < batch; i++)
	{
		forward_shifts[i] = { projDim.x / 2, projDim.y / 2, 1 };
		back_shifts[i] = { -projDim.x / 2, -projDim.y / 2, 1 };
	}
	for (int idx = 0; idx < projections.zdim;) {
		int batchElements = std::min(projections.zdim - idx, batch);
		float * d_projections = MallocDeviceFromHost(projections.data + sliceElements * idx, projections.yxdim*batchElements);
		float * d_ctf = MallocDeviceFromHost(ctf.data + sliceFTElements * idx, Elements(sliceFTDim) * batchElements);
		tcomplex * d_fft;
		cudaMalloc(&d_fft, sliceFTElements*batchElements * sizeof(*d_fft));
		d_FFTR2C(d_projections, d_fft, ndims, sliceDim, batchElements);
		d_Shift(d_fft, d_fft, sliceDim, forward_shifts, false, batchElements);
		d_ComplexMultiplyByVector(d_fft, d_ctf, d_fft, Elements(sliceFTDim) * batchElements, 1);
		d_Shift(d_fft, d_fft, sliceDim, back_shifts, false, batchElements);
		d_IFFTC2R(d_fft, d_projections, ndims, sliceDim, batchElements);
		cudaMemcpy(projections.data + idx*sliceElements, d_projections, sizeof(*d_projections)*Elements(sliceDim)*batchElements, cudaMemcpyDeviceToHost);
		cudaFree(d_projections);
		cudaFree(d_ctf);
		cudaFree(d_fft);
		idx += batchElements;
	}
}

float * do_CTFReconstruction(MultidimArray<float> &projections, int numANgles, float3 * angles, int3 dims) {
	idxtype batch = 1024;
	int3 sliceDim = { dims.x, dims.y, 1 };
	int2 projDim = { dims.x, dims.y };
	int ndims = DimensionCount(sliceDim);
	int sliceElements = Elements2(projDim);
	tfloat3 *h_shifts = (tfloat3 *)malloc(batch * sizeof(*h_shifts));
	Projector proj(dims, 2);
	for (size_t i = 0; i < batch; i++)
	{
		h_shifts[i] = { -sliceDim.x / 2, -sliceDim.y / 2, 0 };
	}
	tcomplex * ctf  = (tcomplex *) malloc(sizeof(*ctf)*projections.zyxdim);
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(projections) {
		ctf[k*YXSIZE(projections) + (i * XSIZE(projections)) + j] = { DIRECT_A3D_ELEM(projections, k, i, j)*DIRECT_A3D_ELEM(projections, k, i, j), 0.0f };
	}
	for (int idx = 0; idx < numANgles;) {
		int batchElements = std::min((int)(numANgles - idx), (int)batch);
		float * d_weights;
		cudaErrchk(cudaMalloc(&d_weights, ElementsFFT(sliceDim)*batchElements*sizeof(*d_weights)));
		gtom::d_ValueFill(d_weights, ElementsFFT(sliceDim)*batchElements, 1.0f);
		tcomplex * d_ctf;
		cudaErrchk(cudaMalloc(&d_ctf, ElementsFFT(sliceDim)*batchElements * sizeof(*d_ctf)));
		cudaErrchk(cudaMemcpy(d_ctf, (void *)(ctf+idx*ElementsFFT(sliceDim)), sizeof(*d_ctf)*batchElements*ElementsFFT(sliceDim), cudaMemcpyHostToDevice));
		if(false){
			float * d_abs_ctf;
			cudaErrchk(cudaMalloc(&d_abs_ctf, ElementsFFT(sliceDim)*batchElements * sizeof(*d_abs_ctf)));
			d_Abs(d_ctf, d_abs_ctf, ElementsFFT(sliceDim)*batchElements);
			
			MultidimArray<float> abs_ctf(batchElements, dims.y, ElementsFFT1(dims.x));
			cudaMemcpy(abs_ctf.data, d_abs_ctf, ElementsFFT(sliceDim)*batchElements * sizeof(*d_abs_ctf), cudaMemcpyDeviceToHost);
			MRCImage<float> abs_ctfIm(abs_ctf);
			abs_ctfIm.writeAs<float>(std::string("D:\\EMD\\9233\\TomoReconstructions\\") + "2.0_readCTF_" + std::to_string(idx) + ".mrc");
		}
		//d_Shift(d_ctf, d_ctf, sliceDim, (tfloat3*)h_shifts, false, batchElements);
		proj.BackProject(d_ctf, d_weights, sliceDim, angles + idx, batchElements, { 1,1,0 });
		cudaErrchk(cudaFree(d_ctf));
		cudaErrchk(cudaFree(d_weights));
		idx += batchElements;
	}
	float * d_ctfReconstruction = proj.d_Reconstruct(true, "C1");
	return d_ctfReconstruction;
}

void do_reconstruction(MultidimArray<float> projections, float3 * angles, int3 dims, bool isCtf = false) {
	idxtype batch = 1024;
	int3 sliceDim = { projections.xdim, projections.ydim, 1 };
	int ndims = DimensionCount(sliceDim);
	tfloat3 *h_shifts = (tfloat3 *)malloc(batch * sizeof(*h_shifts));
	Projector proj(dims, 2);
	for (size_t i = 0; i < batch; i++)
	{
		h_shifts[i] = { sliceDim.x / 2, sliceDim.y / 2, 0 };
	}
	for (int idx = 0; idx < projections.zdim;) {
		int batchElements = std::min(projections.zdim - idx, batch);
		float * d_projections = MallocDeviceFromHost(projections.data + Elements(sliceDim)* idx, projections.yxdim*batchElements);
		float * d_weights = MallocDevice(ElementsFFT(sliceDim)*batchElements);
		gtom::d_ValueFill(d_weights, ElementsFFT(sliceDim)*batchElements, 1.0f);
		tcomplex * d_fft;
		cudaMalloc(&d_fft, ElementsFFT(sliceDim)*batchElements * sizeof(*d_fft));
		d_FFTR2C(d_projections, d_fft, ndims, sliceDim, batchElements);
		d_Shift(d_fft, d_fft, sliceDim, (tfloat3*)h_shifts, false, batchElements);
		proj.BackProject(d_fft, d_weights, sliceDim, angles + idx, batchElements, { 1,1,0 });
		cudaFree(d_projections);
		cudaFree(d_weights);
		cudaFree(d_fft);
		idx += batchElements;

	}
	float * d_reconstruction = proj.d_Reconstruct(false, "C1");
	gtom::d_NormMonolithic(d_reconstruction, d_reconstruction, Elements(dims), T_NORM_MEAN01STD, 1);
	{
		MultidimArray<float> reconstruction(dims.z, dims.y, dims.x);
		cudaMemcpy(reconstruction.data, d_reconstruction, sizeof(float)*Elements(dims), cudaMemcpyDeviceToHost);

		MRCImage<float> reconIm(reconstruction);
		reconIm.writeAs<float>("D:\\EMD\\9233\\TomoReconstructions\\2.0_convolved_fourierSpaceRecon.mrc", true);
	}
	cudaFree(d_reconstruction);
}

float * do_reconstruction(MultidimArray<float> projections, MultidimArray<float> &ctf, float3 * angles, int3 dims, bool isCtf=false) {
	idxtype batch = 1024;
	int3 sliceDim = { projections.xdim, projections.ydim, 1 };
	int ndims = DimensionCount(sliceDim);
	tfloat3 *h_shifts = (tfloat3 *)malloc(batch * sizeof(*h_shifts));
	Projector proj(dims, 2);
	for (size_t i = 0; i < batch; i++)
	{
		h_shifts[i] = {sliceDim.x/2, sliceDim.y/2, 0};
	}
	for (int idx = 0; idx < projections.zdim;) {
		int batchElements = std::min(projections.zdim - idx, batch);
		float * d_projections = MallocDeviceFromHost(projections.data + Elements(sliceDim)* idx, projections.yxdim*batchElements);
		float * d_weights = MallocDeviceFromHost(ctf.data + ElementsFFT(sliceDim)*idx, ElementsFFT(sliceDim)*batchElements);
		tcomplex * d_fft;
		cudaMalloc(&d_fft, ElementsFFT(sliceDim)*batchElements * sizeof(*d_fft));
		d_FFTR2C(d_projections, d_fft, ndims, sliceDim, batchElements);
		d_Shift(d_fft, d_fft, sliceDim, (tfloat3*)h_shifts, false, batchElements);
		proj.BackProject(d_fft, d_weights, sliceDim, angles + idx, batchElements, { 1,1,0 });
		cudaFree(d_projections);
		cudaFree(d_weights);
		cudaFree(d_fft);
		idx += batchElements;

	}
	float * d_reconstruction = proj.d_Reconstruct(false, "C1");
	return d_reconstruction;
}

void ctfTest() {
	cudaErrchk(cudaSetDevice(1));
	cudaErrchk(cudaDeviceSynchronize());
	// Define Parameters
	idxtype N = 800000;
	Algo algo = SIRT;
	FileName pixsize = "2.0";
	RDOUBLE super = 4.0;
	bool writeProjections = false;	//Wether or not to write out projections before and after each iteration
	idxtype numThreads = 24;
	omp_set_num_threads(numThreads);
	idxtype numIt = 3;

	FileName starFileName = "D:\\EMD\\9233\\emd_9233_Scaled_2.0.projections_tomo_convolved-fromAtoms.star";

	FileName refFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".mrc";
	FileName refMaskFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_mask.mrc";
	FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_" + std::to_string(N / 1000) + "k.pdb";		//PDB File containing pseudo atom coordinates
	//FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_2.0_largeMask.pdb.pdb";		//PDB File containing pseudo atom coordinates
	FileName fnOut = "D:\\EMD\\9233\\TomoReconstructions\\2.0_convolvedFromAtoms";
	bool CTFWeighting = true;
	FileName ctfFile = "D:\\EMD\\9233\\Projections_2.0_tomo\\projections_tomo_ctf.mrc";

	//Read Images
	MRCImage<RDOUBLE> origVol = MRCImage<RDOUBLE>::readAs(refFileName);
	MRCImage<RDOUBLE> origMasked = MRCImage<RDOUBLE>::readAs(refFileName);
	MRCImage<RDOUBLE> Mask = MRCImage<RDOUBLE>::readAs(refMaskFileName);
	MRCImage<RDOUBLE> ctf = MRCImage<RDOUBLE>::readAs(ctfFile);
	origMasked.setData(origMasked()*Mask());
	origMasked.writeAs<float>(refFileName.withoutExtension() + "_masked.mrc", true);
	int3 refDims = { origVol().xdim, origVol().ydim, origVol().zdim };

	float3 *angles;
	MultidimArray<RDOUBLE> projections;
	idxtype numProj = readProjections(starFileName, projections, &angles, false);
	//numProj = 2048;


	if (CTFWeighting)
		multiplyByCtf(projections, ctf());



	MultidimArray<float> CTF = ctf();
	CTF *= CTF;		//CTF^2

	MultidimArray<RDOUBLE> realCTF;

	realspaceCTF(CTF, realCTF, refDims);

	MRCImage<RDOUBLE> realspaceCTFIm(realCTF);
	realspaceCTFIm.writeAs<float>(fnOut + "_realspaceCTFs.mrc");

	cudaErrchk(cudaDeviceSynchronize());
	if (writeProjections)
	{
		MRCImage<RDOUBLE> projectionsIM(projections);
		projectionsIM.writeAs<float>(fnOut + "_readProjections.mrc", true);
	}
	//weightProjections(projections, angles, refDims);
	cudaErrchk(cudaDeviceSynchronize());
	if (writeProjections)
	{
		MRCImage<RDOUBLE> projectionsIM(projections);
		projectionsIM.writeAs<float>(fnOut + "_readProjectionsWeighted.mrc", true);
	}



	std::vector<float3> StartAtoms;
	std::vector<RDOUBLE> RefIntensities;
	idxtype NAtoms = Pseudoatoms::readAtomsFromFile(pdbFileName, StartAtoms, RefIntensities, N);
	std::vector<RDOUBLE> StartIntensities;
	StartIntensities.resize(RefIntensities.size(), (RDOUBLE)0);



	PseudoProjector proj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (float *)StartAtoms.data(), (RDOUBLE *)StartIntensities.data(), super, NAtoms);
	proj.lambdaART = 0.001 / projections.zdim;
	//PseudoProjector ctfProj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (float *)StartAtoms.data(), (RDOUBLE *)StartIntensities.data(), super, NAtoms);
	//ctfProj.lambdaART = 0.1 / projections.zdim;
	PseudoProjector RefProj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (float *)StartAtoms.data(), (RDOUBLE *)RefIntensities.data(), super, NAtoms);

	MRCImage<RDOUBLE> *RefImage = RefProj.create3DImage(super);
	RefImage->writeAs<float>(fnOut + "_ref3DIm.mrc", true);

	MultidimArray<float> refArray = RefImage->operator()();
	delete RefImage;
	cudaErrchk(cudaDeviceSynchronize());


	// Do Iterative reconstruction
	for (size_t itIdx = 0; itIdx < numIt; itIdx++)
	{
		if (algo == SIRT) {
			//proj.SIRT(projections, CTF, angles, numProj, NULL, NULL, NULL, NULL, 0, 0);
			//ctfProj.SIRT(realCTF, angles, numProj, 0, 0);
			if (true)
			{
				// Debug variant that writes out correction images
				MultidimArray<RDOUBLE> Itheo, Icorr, Idiff, Inorm;
				proj.SIRT(projections, CTF, angles, numProj, &Itheo, &Icorr, &Idiff, &Inorm, 0.0, 0.0);
				MRCImage<RDOUBLE> imItheo(Itheo);
				imItheo.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Itheo.mrc", true);
				MRCImage<RDOUBLE> imIcorr(Icorr);
				imIcorr.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Icorr.mrc", true);
				MRCImage<RDOUBLE> imIdiff(Idiff);
				imIdiff.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Idiff.mrc", true);
				MRCImage<RDOUBLE> imInorm(Inorm);
				imInorm.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Inorm.mrc", true);
			}
		}

		MRCImage<RDOUBLE> *after1Itoversample = proj.create3DImage(super);
		after1Itoversample->writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + ".mrc", true);
		writeFSC(origVol(), (*after1Itoversample)(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + ".fsc");
		writeFSC(origMasked(), (*after1Itoversample)() * Mask(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_masked.fsc");
		delete after1Itoversample;
		/*
		MRCImage<RDOUBLE> *ctfReconstructionIm = ctfProj.create3DImage(super);
		ctfReconstructionIm->writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_ctfRecon.mrc", true);
		{
			MultidimArray<float> ctfReconstruction = ctfReconstructionIm->operator()();
			float * d_ctfReconstruction;
			cudaErrchk(cudaMalloc(&d_ctfReconstruction, Elements(refDims) * sizeof(*d_ctfReconstruction)));
			cudaErrchk(cudaMemcpy(d_ctfReconstruction, ctfReconstruction.data, Elements(refDims) * sizeof(*d_ctfReconstruction), cudaMemcpyHostToDevice));

			tcomplex * d_fftctfReconstruction;
			cudaErrchk(cudaMalloc(&d_fftctfReconstruction, ElementsFFT(refDims) * sizeof(*d_fftctfReconstruction)));
			float * d_absfftctfReconstruction;
			cudaErrchk(cudaMalloc(&d_absfftctfReconstruction, ElementsFFT(refDims) * sizeof(*d_absfftctfReconstruction)));

			d_FFTR2C(d_ctfReconstruction, d_fftctfReconstruction, DimensionCount(refDims), refDims, 1);
			float R = refDims.x / 2 + 1;
			tfloat3 center = { 0,0,0 };
			int3 size = { ElementsFFT1(refDims.x), refDims.y, refDims.z };
			d_SphereMaskFT(d_fftctfReconstruction, d_fftctfReconstruction, refDims, R, 1);
			d_IFFTC2R(d_fftctfReconstruction, d_ctfReconstruction, DimensionCount(refDims), refDims, 1, true);
			MultidimArray<float> ctfReconstructionMasked(ctfReconstruction.zdim, ctfReconstruction.ydim, ctfReconstruction.xdim);
			cudaErrchk(cudaMemcpy(ctfReconstructionMasked.data, d_ctfReconstruction, Elements(refDims) * sizeof(*d_ctfReconstruction), cudaMemcpyDeviceToHost));
			MRCImage<float> ctfReconstructionMaskedIm(ctfReconstructionMasked);
			ctfReconstructionMaskedIm.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_ctfReconMasked.mrc", true);
			d_Abs(d_fftctfReconstruction, d_absfftctfReconstruction, ElementsFFT(refDims));

			float * d_finalReconstruction;
			cudaErrchk(cudaMalloc(&d_finalReconstruction, Elements(refDims) * sizeof(*d_finalReconstruction)));
			cudaErrchk(cudaMemcpy(d_finalReconstruction, after1Itoversample->data.data, Elements(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyHostToDevice));
			tcomplex *d_fft;
			cudaErrchk(cudaMalloc(&d_fft, ElementsFFT(refDims) * sizeof(*d_fft)));
			d_FFTR2C(d_finalReconstruction, d_fft, DimensionCount(refDims), refDims, 1);
			d_ComplexDivideSafeByVector(d_fft, d_absfftctfReconstruction, d_fft, ElementsFFT(refDims), 1);
			d_SphereMaskFT(d_fft, d_fft, refDims, R, 1);
			d_IFFTC2R(d_fft, d_finalReconstruction, DimensionCount(refDims), refDims, 1);

			MultidimArray<float> finalReconstructionCTFReconstructionWeighted(ctfReconstruction.zdim, ctfReconstruction.ydim, ctfReconstruction.xdim);
			cudaErrchk(cudaMemcpy(finalReconstructionCTFReconstructionWeighted.data, d_finalReconstruction, Elements(refDims) * sizeof(float), cudaMemcpyDeviceToHost));
			MRCImage<float> finalReconstructionCTFReconstructionWeightedIm(finalReconstructionCTFReconstructionWeighted);
			finalReconstructionCTFReconstructionWeightedIm.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_finalReconstructionCTFReconstructionWeightedIm.mrc", true);
			writeFSC(origMasked(), finalReconstructionCTFReconstructionWeightedIm() * Mask(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_finalReconstructionCTFReconstructionWeightedIm.fsc");
			MultidimArray<float> fftctfReconstruction(ctfReconstruction.zdim, ctfReconstruction.ydim, ctfReconstruction.xdim / 2 + 1);
			cudaErrchk(cudaMemcpy(fftctfReconstruction.data, d_absfftctfReconstruction, ElementsFFT(refDims) * sizeof(*d_absfftctfReconstruction), cudaMemcpyDeviceToHost));
			MRCImage<float> fftctfReconstructionIm(fftctfReconstruction);
			fftctfReconstructionIm.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_fftctfRecon.mrc", true);
			cudaErrchk(cudaFree(d_ctfReconstruction));
			cudaErrchk(cudaFree(d_fftctfReconstruction));
			cudaErrchk(cudaFree(d_absfftctfReconstruction));
		}

		if (false) {

			float * h_3DCTF = (float *)malloc(ElementsFFT(refDims) * sizeof(*h_3DCTF));

			float * d_3DCTF = do_CTFReconstruction(CTF, numProj, angles, refDims); //Reconstructed 3D CTF
			{
				cudaErrchk(cudaMemcpy(h_3DCTF, d_3DCTF, ElementsFFT(refDims) * sizeof(*d_3DCTF), cudaMemcpyDeviceToHost));
				{
					MultidimArray<float> CTF3D(refDims.z, refDims.y, ElementsFFT1(refDims.x));
					cudaErrchk(cudaMemcpy(CTF3D.data, d_3DCTF, ElementsFFT(refDims) * sizeof(*d_3DCTF), cudaMemcpyDeviceToHost));
					MRCImage<float> CTF3DIm(CTF3D);
					CTF3DIm.writeAs<float>(fnOut + "_reconstructed3DCTF.mrc", true);
				}
			}


			MultidimArray<float> h_reconstructionWeighted;
			{
				float *d_reconstructionWeighted = NULL; //Reconstruction using fourier backProjection and CTF as weights
				float *d_reconstructionUnweighted = NULL;
				// Create Reconstructions with fourier back projection
				MultidimArray<float> CTF_ones(CTF.zdim, CTF.ydim, CTF.xdim);
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(CTF_ones) {
					DIRECT_A3D_ELEM(CTF_ones, k, i, j) = 1.0f;
				}
				d_reconstructionUnweighted = do_reconstruction(projections, CTF_ones, angles, refDims);
				d_reconstructionWeighted = do_reconstruction(projections, CTF, angles, refDims);

				//Write out Reconstructions
				h_reconstructionWeighted = MultidimArray<float>(refDims.z, refDims.y, refDims.x);
				cudaMemcpy(h_reconstructionWeighted.data, d_reconstructionWeighted, sizeof(float)*Elements(refDims), cudaMemcpyDeviceToHost);
				cudaFree(d_reconstructionWeighted);
			}
			if (true)
			{
				float *d_reconstructionWeighted = NULL; //Reconstruction using fourier backProjection and CTF as weights
				float *d_reconstructionUnweighted = NULL;
				// Create Reconstructions with fourier back projection
				MultidimArray<float> CTF_ones(CTF.zdim, CTF.ydim, CTF.xdim);
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(CTF_ones) {
					DIRECT_A3D_ELEM(CTF_ones, k, i, j) = 1.0f;
				}
				d_reconstructionUnweighted = do_reconstruction(projections, CTF_ones, angles, refDims);
				d_reconstructionWeighted = do_reconstruction(projections, CTF, angles, refDims);

				//Write out Reconstructions
				h_reconstructionWeighted = MultidimArray<float>(refDims.z, refDims.y, refDims.x);
				cudaMemcpy(h_reconstructionWeighted.data, d_reconstructionWeighted, sizeof(float)*Elements(refDims), cudaMemcpyDeviceToHost);

				MRCImage<float> reconWeightedIm(h_reconstructionWeighted);
				reconWeightedIm.writeAs<float>(fnOut + "_fourierReconWeighted.mrc", true);


				MultidimArray<float> h_reconstructionUnweighted(refDims.z, refDims.y, refDims.x);
				cudaMemcpy(h_reconstructionUnweighted.data, d_reconstructionUnweighted, sizeof(float)*Elements(refDims), cudaMemcpyDeviceToHost);

				MRCImage<float> reconUnweightedIm(h_reconstructionUnweighted);
				reconUnweightedIm.writeAs<float>(fnOut + "_fourierRecon.mrc", true);

				writeFSC(h_reconstructionUnweighted*Mask(), origMasked(), fnOut + "_fourierRecon_masked.fsc");
				writeFSC(h_reconstructionWeighted*Mask(), origMasked(), fnOut + "_fourierReconWeighted_masked.fsc");

				//Take FFT of reconstructed volumes
				tcomplex *d_fft_unweighted, *d_fft_weighted;
				float * d_absfft_unweighted, *d_absfft_weighted;
				cudaErrchk(cudaMalloc(&d_absfft_unweighted, ElementsFFT(refDims) * sizeof(*d_fft_unweighted)));
				cudaErrchk(cudaMalloc(&d_absfft_weighted, ElementsFFT(refDims) * sizeof(*d_fft_unweighted)));
				cudaErrchk(cudaMalloc(&d_fft_unweighted, ElementsFFT(refDims) * sizeof(*d_fft_unweighted)));
				cudaErrchk(cudaMalloc(&d_fft_weighted, ElementsFFT(refDims) * sizeof(*d_fft_weighted)));
				d_FFTR2C(d_reconstructionUnweighted, d_fft_unweighted, DimensionCount(refDims), refDims, 1);
				d_FFTR2C(d_reconstructionWeighted, d_fft_weighted, DimensionCount(refDims), refDims, 1);
				d_Abs(d_fft_unweighted, d_absfft_unweighted, ElementsFFT(refDims));
				d_Abs(d_fft_weighted, d_absfft_weighted, ElementsFFT(refDims));

				//Different Measures

					// 1. Calculate what 3D CTF should look like by dividing both reconstructions
				float* d_own3DCTF;
				cudaErrchk(cudaMalloc(&d_own3DCTF, ElementsFFT(refDims) * sizeof(*d_own3DCTF)));

				d_DivideByVector(d_absfft_unweighted, d_absfft_weighted, d_own3DCTF, ElementsFFT(refDims), 1);
				MultidimArray<float> CTF3D(refDims.z, refDims.y, ElementsFFT1(refDims.x));
				cudaErrchk(cudaMemcpy(CTF3D.data, d_own3DCTF, ElementsFFT(refDims) * sizeof(*d_own3DCTF), cudaMemcpyDeviceToHost));
				MRCImage<float> CTF3DIm(CTF3D);
				CTF3DIm.writeAs<float>(fnOut + "_divided3DCTF.mrc", true);

				// 2. Divide unweighted reconstruction by ctf^2
				tcomplex * d_fft_ownWeighted;
				float * d_reconstructionOwnWeighting;
				cudaErrchk(cudaMalloc(&d_fft_ownWeighted, ElementsFFT(refDims) * sizeof(*d_fft_ownWeighted)));
				cudaErrchk(cudaMalloc(&d_reconstructionOwnWeighting, Elements(refDims) * sizeof(*d_reconstructionOwnWeighting)));

				d_ComplexDivideSafeByVector(d_fft_unweighted, d_own3DCTF, d_fft_ownWeighted, ElementsFFT(refDims), 1);
				d_IFFTC2R(d_fft_ownWeighted, d_reconstructionOwnWeighting, DimensionCount(refDims), refDims, 1);

				MultidimArray<float> h_reconstructionOwnWeighting(refDims.z, refDims.y, refDims.x);
				cudaErrchk(cudaMemcpy(h_reconstructionOwnWeighting.data, d_reconstructionOwnWeighting, Elements(refDims) * sizeof(float), cudaMemcpyDeviceToHost));
				MRCImage<float> im_reconstructionOwnWeighting(h_reconstructionOwnWeighting);
				im_reconstructionOwnWeighting.writeAs<float>(fnOut + "_fourierReconOwnWeighting.mrc", true);

				// 3. Weigh using reconstructed 3D CTF
				tfloat3 *h_shifts = (tfloat3 *)malloc(1 * sizeof(*h_shifts));

				h_shifts[0] = { refDims.x / 2, refDims.y / 2, refDims.z/2 };


				cudaErrchk(cudaFree(d_absfft_unweighted));
				cudaErrchk(cudaFree(d_absfft_weighted));
				cudaErrchk(cudaFree(d_fft_weighted));
				cudaErrchk(cudaFree(d_fft_unweighted));
				cudaErrchk(cudaFree(d_own3DCTF));
				cudaErrchk(cudaFree(d_reconstructionWeighted));
				cudaErrchk(cudaFree(d_reconstructionUnweighted));

				cudaErrchk(cudaFree(d_fft_ownWeighted));
				cudaErrchk(cudaFree(d_reconstructionOwnWeighting));
			}

			{
				if (CTFWeighting)
				{
					//Divide by CTF^2
					float * d_finalReconstruction;
					cudaErrchk(cudaMalloc(&d_finalReconstruction, Elements(refDims) * sizeof(*d_finalReconstruction)));
					cudaErrchk(cudaMemcpy(d_finalReconstruction, after1Itoversample->data.data, Elements(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyHostToDevice));
					tcomplex *d_fft;
					cudaErrchk(cudaMalloc(&d_fft, ElementsFFT(refDims) * sizeof(*d_fft)));

					float *d_absfft;
					cudaErrchk(cudaMalloc(&d_absfft, ElementsFFT(refDims) * sizeof(*d_absfft)));

					float * d_fourierReconstruction;
					cudaErrchk(cudaMalloc(&d_fourierReconstruction, Elements(refDims) * sizeof(*d_fourierReconstruction)));
					cudaErrchk(cudaMemcpy(d_fourierReconstruction, h_reconstructionWeighted.data, Elements(refDims) * sizeof(*d_fourierReconstruction), cudaMemcpyHostToDevice));
					tcomplex * d_fftFourierReconstruction;
					cudaErrchk(cudaMalloc(&d_fftFourierReconstruction, ElementsFFT(refDims) * sizeof(*d_fftFourierReconstruction)));
					float * d_absfftFourierReconstruction;
					cudaErrchk(cudaMalloc(&d_absfftFourierReconstruction, ElementsFFT(refDims) * sizeof(*d_absfftFourierReconstruction)));

					d_FFTR2C(d_finalReconstruction, d_fft, DimensionCount(refDims), refDims, 1);
					d_FFTR2C(d_fourierReconstruction, d_fftFourierReconstruction, DimensionCount(refDims), refDims, 1);

					d_Abs(d_fft, d_absfft, ElementsFFT(refDims));
					d_Abs(d_fftFourierReconstruction, d_absfftFourierReconstruction, ElementsFFT(refDims));

					float *d_exp3DCTF;
					cudaErrchk(cudaMalloc(&d_exp3DCTF, ElementsFFT(refDims) * sizeof(*d_exp3DCTF)));
					d_DivideSafeByVector(d_absfft, d_absfftFourierReconstruction, d_exp3DCTF, ElementsFFT(refDims), 1);

					MultidimArray<float>  h_exp3DCTF(refDims.z, refDims.x, ElementsFFT1(refDims.x));
					cudaErrchk(cudaMemcpy(h_exp3DCTF.data, d_exp3DCTF, ElementsFFT(refDims) * sizeof(*h_exp3DCTF.data), cudaMemcpyDeviceToHost));
					MRCImage<float> exp3DCTFIm(h_exp3DCTF);
					exp3DCTFIm.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_fftquotient3DCTF.mrc", true);

					float *d_3DCTF;
					cudaErrchk(cudaMalloc(&d_3DCTF, ElementsFFT(refDims) * sizeof(*d_3DCTF)));
					cudaErrchk(cudaMemcpy(d_3DCTF, h_3DCTF, ElementsFFT(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyHostToDevice));
					//d_MaxOp(d_3DCTF, 1e-2f, d_3DCTF, ElementsFFT(refDims));
					d_ComplexDivideSafeByVector(d_fft, d_3DCTF, d_fft, ElementsFFT(refDims), 1);
					//d_ComplexDivideSafeByVector(d_fft, d_3DCTF, d_fft, ElementsFFT(refDims), 1);

					d_IFFTC2R(d_fft, d_finalReconstruction, DimensionCount(refDims), refDims, 1);
					cudaErrchk(cudaMemcpy(after1Itoversample->data.data, d_finalReconstruction, Elements(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyDeviceToHost));
					cudaErrchk(cudaFree(d_fft));
					cudaErrchk(cudaFree(d_finalReconstruction));
					cudaErrchk(cudaFree(d_3DCTF));
				}
				after1Itoversample->writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_weighted.mrc", true);
				writeFSC(origVol(), (*after1Itoversample)(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_weighted.fsc");
				writeFSC(origMasked(), (*after1Itoversample)() * Mask(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_weighted_masked.fsc");
			}

			delete after1Itoversample;


			MRCImage<RDOUBLE> *finalReconstructionImage = proj.create3DImage(super);
			if (CTFWeighting)
			{
				//Divide by CTF^2
				float * d_finalReconstruction;
				cudaErrchk(cudaMalloc(&d_finalReconstruction, Elements(refDims) * sizeof(*d_finalReconstruction)));
				cudaErrchk(cudaMemcpy(d_finalReconstruction, finalReconstructionImage->data.data, Elements(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyHostToDevice));
				tcomplex *d_fftFinalReconstruction;
				cudaErrchk(cudaMalloc(&d_fftFinalReconstruction, ElementsFFT(refDims) * sizeof(*d_fftFinalReconstruction)));

				float *d_absfftFinalReconstruction;
				cudaErrchk(cudaMalloc(&d_absfftFinalReconstruction, ElementsFFT(refDims) * sizeof(*d_absfftFinalReconstruction)));

				float * d_fourierReconstruction;
				cudaErrchk(cudaMalloc(&d_fourierReconstruction, Elements(refDims) * sizeof(*d_fourierReconstruction)));
				cudaErrchk(cudaMemcpy(d_fourierReconstruction, h_reconstructionWeighted.data, Elements(refDims) * sizeof(*d_fourierReconstruction), cudaMemcpyHostToDevice));
				tcomplex * d_fftFourierReconstruction;
				cudaErrchk(cudaMalloc(&d_fftFourierReconstruction, ElementsFFT(refDims) * sizeof(*d_fftFourierReconstruction)));
				float * d_absfftFourierReconstruction;
				cudaErrchk(cudaMalloc(&d_absfftFourierReconstruction, ElementsFFT(refDims) * sizeof(*d_absfftFourierReconstruction)));



				d_FFTR2C(d_finalReconstruction, d_fftFinalReconstruction, DimensionCount(refDims), refDims, 1);
				d_FFTR2C(d_fourierReconstruction, d_fftFourierReconstruction, DimensionCount(refDims), refDims, 1);

				d_Abs(d_fftFinalReconstruction, d_absfftFinalReconstruction, ElementsFFT(refDims));
				d_Abs(d_fftFourierReconstruction, d_absfftFourierReconstruction, ElementsFFT(refDims));

				float *d_exp3DCTF;
				cudaErrchk(cudaMalloc(&d_exp3DCTF, ElementsFFT(refDims) * sizeof(*d_exp3DCTF)));
				d_DivideSafeByVector(d_absfftFourierReconstruction, d_absfftFinalReconstruction, d_exp3DCTF, ElementsFFT(refDims), 1);

				MultidimArray<float>  h_exp3DCTF(refDims.z, refDims.x, ElementsFFT1(refDims.x));
				cudaErrchk(cudaMemcpy(h_exp3DCTF.data, d_exp3DCTF, ElementsFFT(refDims) * sizeof(*h_exp3DCTF.data), cudaMemcpyDeviceToHost));
				MRCImage<float> exp3DCTFIm(h_exp3DCTF);
				exp3DCTFIm.writeAs<float>(fnOut + "_fftquotient3DCTF.mrc", true);

				float *d_3DCTF;
				cudaErrchk(cudaMalloc(&d_3DCTF, ElementsFFT(refDims) * sizeof(*d_3DCTF)));
				cudaErrchk(cudaMemcpy(d_3DCTF, h_3DCTF, ElementsFFT(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyHostToDevice));
				//d_MaxOp(d_3DCTF, 1e-2f, d_3DCTF, ElementsFFT(refDims));
				d_ComplexDivideSafeByVector(d_fftFinalReconstruction, d_3DCTF, d_fftFinalReconstruction, ElementsFFT(refDims), 1);
				//d_ComplexDivideSafeByVector(d_fft, d_3DCTF, d_fft, ElementsFFT(refDims), 1);
				float R = refDims.x / 2 + 1;
				tfloat3 center = { 0,0,0 };
				int3 size = { ElementsFFT1(refDims.x), refDims.y, refDims.z };

				d_SphereMaskFT(d_finalReconstruction, d_finalReconstruction, refDims, R, 1);
				d_IFFTC2R(d_fftFinalReconstruction, d_finalReconstruction, DimensionCount(refDims), refDims, 1);

				tcomplex * d_fftOwnDivided;
				cudaErrchk(cudaMalloc(&d_fftOwnDivided, ElementsFFT(refDims) * sizeof(*d_fftOwnDivided)));
				d_ComplexDivideSafeByVector(d_fftFinalReconstruction, d_exp3DCTF, d_fftOwnDivided, ElementsFFT(refDims), 1);

				float * d_ownDivided;
				cudaErrchk(cudaMalloc(&d_ownDivided, Elements(refDims) * sizeof(*d_ownDivided)));

				d_IFFTC2R(d_fftOwnDivided, d_ownDivided, DimensionCount(refDims), refDims, 1);

				MultidimArray<float> h_ownDivided(refDims.z, refDims.y, refDims.x);
				cudaMemcpy(h_ownDivided.data, d_ownDivided, Elements(refDims) * sizeof(float), cudaMemcpyDeviceToHost);

				MRCImage<float> ownDividedIm(h_ownDivided);
				ownDividedIm.writeAs<float>(fnOut + "_ownWeighted.mrc");

				cudaErrchk(cudaMemcpy(finalReconstructionImage->data.data, d_finalReconstruction, Elements(refDims) * sizeof(*d_finalReconstruction), cudaMemcpyDeviceToHost));
				cudaErrchk(cudaFree(d_fftFinalReconstruction));
				cudaErrchk(cudaFree(d_finalReconstruction));
				cudaErrchk(cudaFree(d_3DCTF));
			}
			finalReconstructionImage->writeAs<float>(fnOut + "_finalReconstruction.mrc", true);
			writeFSC(origVol(), (*finalReconstructionImage)(), fnOut + "_finalReconstruction.fsc");
			writeFSC(origMasked(), (*finalReconstructionImage)() * Mask(), fnOut + "_finalReconstruction_masked.fsc");
		}*/
	}
}
unsigned seed = 42;
MultidimArray<RDOUBLE> applyRandomNoise(MultidimArray<RDOUBLE> &projections, MultidimArray<RDOUBLE> &maskProjections, RDOUBLE snr) {


	float minSignal = 100000;
	float maxSignal = -100000;
	double mean = 0.0;
	double maskSum = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projections) {
		RDOUBLE val = DIRECT_MULTIDIM_ELEM(projections, n);
		RDOUBLE maskVal = DIRECT_MULTIDIM_ELEM(maskProjections, n);
		minSignal = std::min(minSignal, val);
		maxSignal = std::max(maxSignal, val);
		mean += maskVal * val;
		maskSum += maskVal;
	}
	mean /= maskSum;
	double stdDevSig = 0.0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(projections) {
		RDOUBLE maskVal = DIRECT_MULTIDIM_ELEM(maskProjections, n);
		stdDevSig += maskVal*pow(DIRECT_MULTIDIM_ELEM(projections, n) - mean, 2);

	}
	stdDevSig = sqrt(stdDevSig/maskSum);
	double stdDevNoise = stdDevSig / snr;
	MultidimArray<RDOUBLE> noise(projections.zdim, projections.ydim, projections.xdim);
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<RDOUBLE> distribution(0, mean / snr);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(noise) {
		DIRECT_MULTIDIM_ELEM(noise, n) = distribution(generator);
	}
	projections += noise;
	return noise;
}


int main(int argc, char** argv) {

	int deviceCount = 0;
	cudaErrchk(cudaGetDeviceCount(&deviceCount));
	idxtype maxMem = 0;
	int maxmemDevice = -1;
	for (int i = 0; i < deviceCount; i++)
	{
		cudaErrchk(cudaSetDevice(i));
		idxtype GPU_FREEMEM;
		idxtype GPU_MEMLIMIT;
		cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);
		if (GPU_FREEMEM > maxMem) {
			maxmemDevice = i;
			maxMem = GPU_FREEMEM;
		}
	}
	if (maxmemDevice == -1) {
		std::cerr << "There was no device that had memory left out of " << deviceCount << " devices" << std::endl;
		return EXIT_FAILURE;
	}
	cudaErrchk(cudaSetDevice(maxmemDevice));
	cudaErrchk(cudaDeviceSynchronize());
	// Define Parameters
	idxtype N = 800000;
	Algo algo = SIRT;
	FileName pixsize = "2.0";
	RDOUBLE super = 4.0;
	bool writeProjections = false;	//Wether or not to write out projections before and after each iteration
	idxtype numThreads = 24;
	omp_set_num_threads(numThreads);
	idxtype numIt = 3;

	FileName projectionsOneStarFileName = "D:\\EMD\\9233\\Movement_Analysis\\original_proj.star";
	FileName tsvOneFileName = "D:\\EMD\\9233\\Movement_Analysis\\original.tsv";

	bool haveSecondFile = true;
	FileName projectionsTwoStarFileName = "D:\\EMD\\9233\\Movement_Analysis\\original_proj.star";
	FileName tsvTwoFileName = "D:\\EMD\\9233\\Movement_Analysis\\original.tsv";

	FileName refFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".mrc";
	FileName refMaskFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_mask.mrc";
	//PDB File containing pseudo atom coordinates
	//FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_2.0_largeMask.pdb.pdb";		//PDB File containing pseudo atom coordinates
	for (int multiply = 1; multiply < 10; multiply++) {
		FileName outdir = "D:\\EMD\\9233\\Movement_Analysis\\Reconstruction_Single_" + std::to_string(multiply) + "_with_weighting\\";
		fs::create_directories(outdir.c_str());

		FileName fnOut = outdir + "recon";

		//Read Images
		MRCImage<RDOUBLE> origVol = MRCImage<RDOUBLE>::readAs(refFileName);
		MRCImage<RDOUBLE> origMasked = MRCImage<RDOUBLE>::readAs(refFileName);
		MRCImage<RDOUBLE> Mask = MRCImage<RDOUBLE>::readAs(refMaskFileName);
		origMasked.setData(origMasked()*Mask());
		origMasked.writeAs<float>(refFileName.withoutExtension() + "_masked.mrc", true);
		int3 refDims = { origVol().xdim, origVol().ydim, origVol().zdim };

		float3 *anglesOne;
		MultidimArray<RDOUBLE> projectionsOne;
		idxtype numProjOne = readProjections(projectionsOneStarFileName, projectionsOne, &anglesOne, false);
		MultidimArray<RDOUBLE> CTFOne(projectionsOne.zdim, projectionsOne.ydim, projectionsOne.xdim / 2 + 1);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(CTFOne) {
			DIRECT_MULTIDIM_ELEM(CTFOne, n) = 1.0;
		}

		float3 *anglesTwo;
		MultidimArray<RDOUBLE> projectionsTwo;
		idxtype numProjTwo = 0;
		if (haveSecondFile) {

			numProjTwo = readProjections(projectionsTwoStarFileName, projectionsTwo, &anglesTwo, false);
			float3 *newAnglesTwo = (float3*)malloc(multiply * numProjTwo * sizeof(float3));
			MultidimArray<RDOUBLE> newProjectionsTwo(multiply*numProjTwo, projectionsTwo.ydim, projectionsTwo.xdim);
			for (size_t i = 0; i < multiply; i++)
			{
				memcpy(newAnglesTwo + numProjTwo * i, anglesTwo, numProjTwo * sizeof(float3));
				memcpy(newProjectionsTwo.data + projectionsTwo.nzyxdim * i, projectionsTwo.data, projectionsTwo.nzyxdim * sizeof(RDOUBLE));
			}
			numProjTwo *= multiply;
			free(anglesTwo);
			anglesTwo = newAnglesTwo;
			projectionsTwo = newProjectionsTwo;
		}
		//numProj = 2048;
		MultidimArray<RDOUBLE> CTFTwo(projectionsOne.zdim, projectionsOne.ydim, projectionsOne.xdim / 2 + 1);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(CTFTwo) {
			DIRECT_MULTIDIM_ELEM(CTFTwo, n) = 1.0;
		}

		idxtype numProj = numProjOne + numProjTwo;

		MultidimArray<RDOUBLE> projections(numProj, projectionsOne.ydim, projectionsOne.xdim);
		MultidimArray<RDOUBLE> CTF(numProj, CTFOne.ydim, CTFOne.xdim);
		float3 * angles = (float3*)malloc(numProj * sizeof(*angles));
		memcpy(projections.data, projectionsOne.data, projectionsOne.zyxdim * sizeof(*projections.data));
		memcpy(CTF.data, CTFOne.data, CTFOne.zyxdim * sizeof(*projections.data));

		memcpy(angles, anglesOne, numProjOne * sizeof(*angles));
		if (haveSecondFile) {
			memcpy(projections.data + projectionsOne.zyxdim, projectionsTwo.data, projectionsTwo.zyxdim * sizeof(*projections.data));
			memcpy(CTF.data + CTFOne.zyxdim, CTFTwo.data, CTFTwo.zyxdim * sizeof(*projections.data));
			memcpy(angles + numProjOne, anglesTwo, numProjTwo * sizeof(*angles));
		}


		int *positionMatching = (int*)malloc(sizeof(int)*numProj);
		for (size_t i = 0; i < numProjOne; i++)
		{
			positionMatching[i] = 0;
		}
		for (size_t i = numProjOne; i < numProj; i++)
		{
			positionMatching[i] = 0;
		}
		cudaErrchk(cudaDeviceSynchronize());
		if (writeProjections)
		{
			MRCImage<RDOUBLE> projectionsIM(projections);
			projectionsIM.writeAs<float>(fnOut + "_readProjections.mrc", true);
			MRCImage<RDOUBLE> projectionsIMOne(projectionsOne);
			projectionsIMOne.writeAs<float>(fnOut + "_readProjectionsOne.mrc", true);
			if (haveSecondFile)
			{
				MRCImage<RDOUBLE> projectionsIMTwo(projectionsTwo);
				projectionsIMTwo.writeAs<float>(fnOut + "_readProjectionsTwo.mrc", true);
			}
		}

		cudaErrchk(cudaDeviceSynchronize());


		Pseudoatoms *Atoms = Pseudoatoms::ReadTsvFile(tsvOneFileName);

		if (haveSecondFile) {
			Pseudoatoms *AtomsTwo = Pseudoatoms::ReadTsvFile(tsvTwoFileName);

			Atoms->addAlternativeOrientation((float*)AtomsTwo->alternativePositions[0], AtomsTwo->NAtoms);
			delete AtomsTwo;
		}


		int3 projDim = make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim));
		PseudoProjector RefProj(projDim, Atoms, super);

		for (size_t i = 0; i < Atoms->NAtoms; i++)
		{
			Atoms->AtomWeights[i] = 1;
		}
		MultidimArray < RDOUBLE> projectionMask;
		if (!fs::exists(fnOut + "_projMask.mrc")) {
			PseudoProjector MaskProj(projDim, Atoms, 1.0);

			MaskProj.createMask(angles, positionMatching, NULL, projectionMask, numProj, 0.0, 0.0);

			projectionMask.binarize(0.1);
			{
				MRCImage im(projectionMask);
				im.writeAs<float>(fnOut + "_projMask.mrc");
			}
		}
		else {
			MRCImage<float> im = MRCImage<float>::readAs(fnOut + "_projMask.mrc");
			projectionMask = im();
		}
		for (size_t i = 0; i < Atoms->NAtoms; i++)
		{
			Atoms->AtomWeights[i] = 0;
		}

		MultidimArray<RDOUBLE> unnoised = projections;

		float snrs[] = { 0.1 };
		for (float snr : snrs)
		{
			double lambda = 0.1;
			PseudoProjector proj(projDim, Atoms, super);
			proj.lambdaART = lambda / projections.zdim;
			projections = unnoised;
			if (snr != std::numeric_limits<float>::infinity()) {
				MultidimArray<RDOUBLE> noise = applyRandomNoise(projections, projectionMask, snr);
				fnOut = outdir + "lb_" + std::to_string(lambda) + "_snr_" + std::to_string(snr) + "\\recon";
				fs::create_directories((outdir + "lb_" + std::to_string(lambda) + "_snr_" + std::to_string(snr)).c_str());
				if (true)
				{
					MRCImage<RDOUBLE> projectionsIM(projections);
					projectionsIM.writeAs<float>(fnOut + "_readProjectionsnoised.mrc", true);
					MRCImage<RDOUBLE> noiseIm(noise);
					noiseIm.writeAs<float>(fnOut + "_noise.mrc", true);
				}
			}
			else {
				fnOut = outdir + "lb_" + std::to_string(lambda) + "_snr_inf\\recon";
				fs::create_directories((outdir + "lb_" + std::to_string(lambda) + "_snr_" + std::to_string(snr)).c_str());
				if (true)
				{
					MRCImage<RDOUBLE> projectionsIM(projections);
					projectionsIM.writeAs<float>(fnOut + "_readProjectionsnoised.mrc", true);
					MultidimArray<RDOUBLE> noise;
					noise.resize(projections.zdim, projections.ydim, projections.xdim);
					MRCImage<RDOUBLE> noiseIm(noise);
					noiseIm.writeAs<float>(fnOut + "_noise.mrc", true);
				}
			}
			weightProjections(projections, angles, refDims);

			if (writeProjections)
			{
				MRCImage<RDOUBLE> projectionsIM(projections);
				projectionsIM.writeAs<float>(fnOut + "_readProjectionsWeighted.mrc", true);
			}

			//PseudoProjector ctfProj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (float *)StartAtoms.data(), (RDOUBLE *)StartIntensities.data(), super, NAtoms);
			//ctfProj.lambdaART = 0.1 / projections.zdim;


			MRCImage<RDOUBLE> *RefImage = RefProj.create3DImage(super);
			RefImage->writeAs<float>(fnOut + "_ref3DIm.mrc", true);

			MultidimArray<float> refArray = RefImage->operator()();
			delete RefImage;
			cudaErrchk(cudaDeviceSynchronize());



			// Do Iterative reconstruction
			for (size_t itIdx = 0; itIdx < numIt; itIdx++)
			{
				if (algo == SIRT) {
					proj.SIRT(projections, angles, positionMatching, numProj, NULL, NULL, NULL, NULL, 0, 0);
					//ctfProj.SIRT(realCTF, angles, numProj, 0, 0);
					if (false)
					{
						// Debug variant that writes out correction images
						MultidimArray<RDOUBLE> Itheo, Icorr, Idiff, Inorm;
						proj.SIRT(projections, angles, positionMatching, numProj, &Itheo, &Icorr, &Idiff, &Inorm, 0.0, 0.0);
						MRCImage<RDOUBLE> imItheo(Itheo);
						imItheo.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Itheo.mrc", true);
						MRCImage<RDOUBLE> imIcorr(Icorr);
						imIcorr.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Icorr.mrc", true);
						MRCImage<RDOUBLE> imIdiff(Idiff);
						imIdiff.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Idiff.mrc", true);
						MRCImage<RDOUBLE> imInorm(Inorm);
						imInorm.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_Inorm.mrc", true);
					}
				}

				MRCImage<RDOUBLE> *after1Itoversample = proj.create3DImage(super);
				after1Itoversample->writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + ".mrc", true);
				writeFSC(origVol(), (*after1Itoversample)(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + ".fsc");
				writeFSC(origMasked(), (*after1Itoversample)() * Mask(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled" + std::to_string((int)super) + "_masked.fsc");
				delete after1Itoversample;
			}
		}
		delete Atoms;
		free(angles);
		free(anglesOne);
		free(anglesTwo);
	}
}