#include "liblionImports.h"
#include "glm/glm.hpp"
#include "Relion.cuh"
#include "src/projector.h"
#include "metadata_table.h"
#include "pseudoatoms.h"
#include "readMRC.h"
#include "volume_to_pseudoatoms.h"
#include <random>
#include "GPU_Project.cuh"
#include <direct.h>
#include <filesystem>

#include <omp.h>

namespace fs = std::filesystem;

static inline void outputDeviceAsImage(float *d_data, int3 Dims, FileName outName, bool isFT = false) {
	MultidimArray<float> h_data(Dims.z, Dims.y, isFT ? (Dims.x / 2 + 1) : Dims.x);
	cudaErrchk(cudaMemcpy(h_data.data, d_data, isFT ? ElementsFFT(Dims) : Elements(Dims) * sizeof(*d_data), cudaMemcpyDeviceToHost));
	MRCImage<float> h_im(h_data);
	h_im.writeAs<float>(outName, true);
}


void projectVolume(FileName outname, MultidimArray<RDOUBLE> &ref, float3 *angles, int N, int projdim, int oversampling) {
	int3 dims = make_int3(ref.xdim);
	relion::MultidimArray<float> vol;
	vol.initZeros(dims.z, dims.y, dims.x);
	for (uint i = 0; i < Elements(dims); i++)
		vol.data[i] = ref.data[i];
	relion::MultidimArray<float> dummy;

	int3 projDim = make_int3(projdim, projdim, 1);

	int Oversampled = 2 * (oversampling * (dims.x / 2) + 1) + 1;
	int3 DimsOversampled = make_int3(Oversampled, Oversampled, Oversampled);
	int3 projectordims = make_int3(DimsOversampled.x / 2 + 1, DimsOversampled.y, DimsOversampled.z);

	float *flat_initialized = (float*)malloc(sizeof(float)*Elements(projectordims) * 2);
	InitProjector(dims, oversampling, vol.data, flat_initialized, projdim);

	float2 *h_initialized = (float2*)malloc(sizeof(float2)*Elements(projectordims));


	for (uint i = 0; i < Elements(projectordims); i++) {
		((float2*)h_initialized)[i] = make_float2(flat_initialized[2*i], flat_initialized[2*i+1]);
	}

	float2 * d_initialized;
	cudaErrchk(cudaMalloc(&d_initialized, Elements(projectordims) * sizeof(*h_initialized)));
	cudaErrchk(cudaMemcpy(d_initialized, h_initialized, Elements(projectordims)*sizeof(*d_initialized), cudaMemcpyHostToDevice));

	uint64_t h_textureid[2];
	uint64_t h_arrayid[2];
	CreateTexture3DComplex(d_initialized, projectordims, h_textureid, h_arrayid, false);
	cudaErrchk(cudaPeekAtLastError())
	cudaTex t_DataRe = h_textureid[0];
	cudaTex t_DataIm = h_textureid[1];
	cudaTex a_DataRe = h_arrayid[0];
	cudaTex a_DataIm = h_arrayid[1];

	tcomplex * d_proj;
	tfloat * d_proj_real;
	tfloat * d_proj_real_remapped;
	cudaErrchk(cudaMalloc(&d_proj, (projdim / 2 + 1)*projdim*N * sizeof(*d_proj)));
	cudaErrchk(cudaMalloc(&d_proj_real, projdim*projdim*N * sizeof(*d_proj_real)));
	cudaErrchk(cudaMalloc(&d_proj_real_remapped, projdim*projdim*N * sizeof(*d_proj_real_remapped)));

	ProjectForwardTex(t_DataRe, t_DataIm, d_proj, make_int3(Oversampled), make_int2(projdim, projdim), angles, oversampling, N);

	cudaErrchk(cudaPeekAtLastError())
	d_IFFTC2R(d_proj, d_proj_real, DimensionCount(projDim), projDim, N, false);
	cudaErrchk(cudaPeekAtLastError())
	d_RemapFull2FullFFT(d_proj_real, d_proj_real_remapped, projDim, N);
	cudaErrchk(cudaPeekAtLastError())
	MultidimArray<RFLOAT> projections(N, projdim, projdim);
	cudaMemcpy(projections.data, d_proj_real_remapped, projections.nzyxdim * sizeof(float), cudaMemcpyDeviceToHost);
	
	MRCImage<float> h_projim(projections);
	h_projim.writeAs<float>(outname, true);

	cudaErrchk(cudaFree(d_proj));
	cudaErrchk(cudaFree(d_proj_real));
	cudaErrchk(cudaFree(d_proj_real_remapped));
	DestroyTexture(t_DataRe, a_DataRe);
	DestroyTexture(t_DataIm, a_DataIm);
}

void writeProjectionsToDisk(Pseudoatoms &atoms, float3* angles, idxtype numAngles, float super, int3 Dims, FileName outname) {

	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms.NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, atoms.AtomPositions.data(), atoms.NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms.NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms.AtomWeights.data(), atoms.NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)1024);	//Hard limit of elementsPerBatch instead of calculating

	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

	float * d_superProjections;
	float * d_projections;

	tcomplex * d_projectionsBatchFFT;

	cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
	cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
	cudaErrchk(cudaMalloc(&d_projectionsBatchFFT, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(tcomplex)));

	// Forward project in batches
	for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
	{
		idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
		float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
		int3 batchProjDim = Dims;
		batchProjDim.z = batch;

		int3 batchSuperProjDim = Dims * super;
		batchSuperProjDim.z = batch;
		float3 * h_angles = angles + startIm;

		RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, atoms.NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
		cudaErrchk(cudaPeekAtLastError());
		if (false)
			outputDeviceAsImage(d_superProjections, batchSuperProjDim, outname + std::string("_d_superProjectionsBatch_it") + std::to_string(startIm) + ".mrc", false);
		//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
		d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch, NULL, d_projectionsBatchFFT);
		if (false)
			outputDeviceAsImage(d_projectionsBatch, batchProjDim, outname + std::string("_d_ProjectionsBatch_it") + std::to_string(startIm) + ".mrc", false);
	}

	MultidimArray<float> h_data(numAngles, Dims.y, Dims.x);
	cudaErrchk(cudaMemcpy(h_data.data, d_projections, h_data.nzyxdim * sizeof(*d_projections), cudaMemcpyDeviceToHost));
	MRCImage<float> h_im(h_data);
	h_im.writeAs<float>(outname, true);

	cudaErrchk(cudaFree(d_superProjections));
	cudaErrchk(cudaFree(d_projections));
	cudaErrchk(cudaFree(d_projectionsBatchFFT));
	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_atomIntensities));
	cufftDestroy(planForward);
	cufftDestroy(planBackward);
}


idxtype readProjections(FileName starFileName, MultidimArray<RDOUBLE> &projections, float3 **angles, bool shuffle = false)
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
	if (shuffle)
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

static void createDir(std::string Path) {

	if (!fs::is_directory(Path)) { // Check if src folder exists
		fs::create_directories(Path); // create src folder
	}
}





int main(int argc, char** argv) {

	createDir("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\AtomGraphs");
	createDir("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output");
	omp_set_num_threads(20);
	cudaSetDevice(1);


	MRCImage<float> im = MRCImage<float>::readAs("D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0.mrc");
	MRCImage<float> softMask = MRCImage<float>::readAs("D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0_softMask.mrc");
	/*
	{
		MultidimArray<RFLOAT> rastered;

		for (size_t i = 0; i < 10; i++)
		{
			//Atoms.RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
			MRCImage input = MRCImage<float>::readAs("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\weighting_false_4.000000_1\\8.mrc");
			rastered = input();
			MultidimArray<RDOUBLE> fsc;
			MetaDataTable MDfsc;
			myGetFSC(input()*softMask(), im()*softMask(), fsc);
			for (size_t k = 0; k < XSIZE(fsc); k++)
			{
				std::cout << DIRECT_A1D_ELEM(fsc, k) << std::endl;
			}
		}
	}
	return 0;*/
	






	int3 dims = { im().xdim, im().ydim, im().zdim };
	//float super = 4.0;


	//int N = 600000;
	int N_series[11] = { 600000, 1000000, 1000, 10000, 20000, 30000, 50000, 100000, 200000, 500000,  800000 };
	float Super_series[3] = { 4, 1, 2 };
	//float3 *angles;
	//MultidimArray<RDOUBLE> refProjections;
	//idxtype numProj = readProjections("D:\\EMD\\9233\\emd_9233_Scaled_2.0.projections_uniform.star", refProjections, &angles, false);
	std::vector<float3> vec_angles = GetHealpixAnglesRad(1);
	idxtype numProj = vec_angles.size();
	char bufferN[100];
	char bufferSuper[100];
	char bufferOutput[200];
	projectVolume("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\refProjections.mrc", im(), vec_angles.data(), numProj, dims.x, 2);
	FILE *logfileFSC, *logfileMaskedFSC;
	bool weighting = false;
	if (false) {
		logfileFSC = fopen("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\fsc_values.tsv", "w");
		logfileMaskedFSC = fopen("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\fsc_masked_values.tsv", "w");
		fprintf(logfileFSC, "weighting\tN\tSuper\titeration\tWave_Number\tFSC\n");


		fprintf(logfileMaskedFSC, "weighting\tN\tSuper\titeration\tWave_Number\tFSC\n");
		for each (int N in N_series)
		{
			for each (float super in Super_series)
			{
				//std::cout << "Thread " << omp_get_thread_num() << " working on " << i << " " << super << std::endl;
				//continue;
				sprintf(bufferN, "%d", (int)N);

				sprintf(bufferSuper, "%lf", super);
				sprintf(bufferOutput, "D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\AtomGraphs\\emd_9233_Scaled_2.0_%dk_%d.pdb", N / 1000, (int)super);
				ProgVolumeToPseudoatoms initializer(23, new char*[23]{ "consecutive_rastering","-i", "D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0.mrc", "-o", bufferOutput, "--initialSeeds", bufferN, "--oversampling", bufferSuper, "--mask", "binary_file",
					"--maskfile", "D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0_zylinderMask.mrc", "--InterpolateValues", "true", "--dontAllowMovement",  "true", "--dontAllowIntensity", "true", "--dontAllowNumberChange", "false", "--thr", "20" });

				initializer.run();
				Pseudoatoms Atoms = initializer.Atoms;

				MultidimArray<RFLOAT> rastered;
				FileName outdir = "D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\";

				outdir = outdir + "weighting_" + (weighting ? "true" : "false") + "_" + std::to_string(super) + "_" + std::to_string(N / 1000) + "\\";
				fs::create_directories(outdir.c_str());

				for (size_t i = 0; i < 10; i++)
				{
					//Atoms.RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
					MRCImage input = MRCImage<float>::readAs(outdir + std::to_string(i) + ".mrc");
					rastered = input();
					{
						MRCImage<float> out(rastered);
						out.writeAs<float>(outdir + std::to_string(i) + ".mrc");
					}

					auto fsc = writeFSC(im()*softMask(), rastered*softMask(), std::string(outdir) + std::to_string(i) + "_fsc_softMasked.star");
					for (size_t j = 0; j < XSIZE(fsc); j++)
					{
						fprintf(logfileMaskedFSC, "%d\t%d\t%d\t%d\t%d\t%f\n", weighting ? 1 : 0, N, (int)(super), i, j, DIRECT_A1D_ELEM(fsc, j));
					}
					fflush(logfileMaskedFSC);

					fsc = writeFSC(im(), rastered, std::string(outdir) + std::to_string(i) + "_fsc.star");
					for (size_t j = 0; j < XSIZE(fsc); j++)
					{
						fprintf(logfileFSC, "%d\t%d\t%d\t%d\t%d\t%f\n", weighting ? 1 : 0, N, (int)(super), i, j, DIRECT_A1D_ELEM(fsc, j));
					}
					fflush(logfileFSC);

					//writeProjectionsToDisk(Atoms, vec_angles.data(), numProj, super, dims, outdir + std::to_string(i) + "_proj.mrc");					
					Atoms.IntensityFromVolume(rastered, 4.0);
				}
			}
		}
		fclose(logfileFSC);
		fclose(logfileMaskedFSC);
	

	logfileFSC = fopen("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\fsc_values_with_weighting.tsv", "w");
	logfileMaskedFSC = fopen("D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\fsc_masked_values_with_weighting.tsv", "w");

	weighting = true;
	for (int i = 0; i < 11; i++)
	{
		int  N = N_series[i];

		for each (float super in Super_series)
		{
			std::cout << "Thread " << omp_get_thread_num() << " working on " << i << " " << super << std::endl;
			//continue;
			sprintf(bufferN, "%d", (int)N);

			sprintf(bufferSuper, "%lf", super);
			sprintf(bufferOutput, "D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\AtomGraphs\\emd_9233_Scaled_2.0_%dk_%d.pdb", N / 1000, (int)super);
			ProgVolumeToPseudoatoms initializer(23, new char*[23]{ "consecutive_rastering","-i", "D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0.mrc", "-o", bufferOutput, "--initialSeeds", bufferN, "--oversampling", bufferSuper, "--mask", "binary_file",
				"--maskfile", "D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0_zylinderMask.mrc", "--InterpolateValues", "true", "--dontAllowMovement",  "true", "--dontAllowIntensity", "true", "--dontAllowNumberChange", "false", "--thr", "20" });

			initializer.run();
			Pseudoatoms Atoms = initializer.Atoms;

			MultidimArray<RFLOAT> rastered;
			FileName outdir = "D:\\FlexibleRefinementResults\\Results\\Consecutive_Rastering\\Output\\";

			outdir = outdir + "weighting_" + (weighting ? "true" : "false") + "_" + std::to_string(super) + "_" + std::to_string(N / 1000) + "\\";
			fs::create_directories(outdir.c_str());

			for (size_t i = 0; i < 10; i++)
			{
				Atoms.RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
				{
					MRCImage<float> out(rastered);
					out.writeAs<float>(outdir + std::to_string(i) + ".mrc", true);
				}

				auto fsc = writeFSC(im()*softMask(), rastered*softMask(), std::string(outdir) + std::to_string(i) + "_fsc_softMasked.star");
				for (size_t j = 0; j < XSIZE(fsc); j++)
				{
					fprintf(logfileMaskedFSC, "%d\t%d\t%d\t%d\t%d\t%f\n", weighting ? 1 : 0, N, (int)(super), i, j, DIRECT_A1D_ELEM(fsc, j));
				}
				fflush(logfileMaskedFSC);

				fsc = writeFSC(im(), rastered, std::string(outdir) + std::to_string(i) + "_fsc.star");
				for (size_t j = 0; j < XSIZE(fsc); j++)
				{
					fprintf(logfileFSC, "%d\t%d\t%d\t%d\t%d\t%f\n", weighting ? 1 : 0, N, (int)(super), i, j, DIRECT_A1D_ELEM(fsc, j));
				}
				fflush(logfileFSC);

				writeProjectionsToDisk(Atoms, vec_angles.data(), numProj, super, dims, outdir + std::to_string(i) + "_proj.mrc");
				Atoms.IntensityFromVolume(rastered, 4.0);
			}
		}
	}
	}
}

