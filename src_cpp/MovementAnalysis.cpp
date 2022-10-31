#include "liblionImports.h"
#include "AtomMover.h"
#include "metadata_table.h"
#include "pseudoatoms.h"
#include <filesystem>
#include "readMRC.h"
#include "volume_to_pseudoatoms.h"
#include <random>
#include "GPU_Project.cuh"
#include <direct.h>
#include "ADAM_Solver.h"
#include <omp.h>
#include <io.h>
#include "ProjectionReader.h"

namespace fs = std::filesystem;

static inline void outputDeviceAsImage(float *d_data, int3 Dims, FileName outName, bool isFT = false) {
	MultidimArray<float> h_data(Dims.z, Dims.y, isFT ? (Dims.x / 2 + 1) : Dims.x);
	cudaErrchk(cudaMemcpy(h_data.data, d_data, isFT ? ElementsFFT(Dims) : Elements(Dims) * sizeof(*d_data), cudaMemcpyDeviceToHost));
	MRCImage<float> h_im(h_data);
	h_im.writeAs<float>(outName, true);
}
void writeProjectionsToDisk(Pseudoatoms *Atoms, float3* angles, idxtype numAngles, float super, int3 Dims, FileName outname) {

	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, Atoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, Atoms->AtomPositions.data(), Atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, Atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, Atoms->AtomWeights.data(), Atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

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

		RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, Atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
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

/*
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
*/

int main(int argc, char** argv) {
	omp_set_num_threads(20);

	// Parameters
	FileName inVol = "D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0.mrc"; 
	FileName inMask = "D:\\FlexibleRefinementResults\\input\\emd_9233_Scaled_2.0_softMask.mrc";
	FileName inProj = "D:\\FlexibleRefinementResults\\input\\projectionsTomo_order4\\emd_9233_Scaled_2.0.projections_tomo.star";
	FileName outdirBase = "D:\\FlexibleRefinementResults\\input\\MovementAnalysis\\";

	std::vector<RFLOAT> steps = { 0.25, 0.33, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0 };
	float diff = 3.0;
	int N = 500000;
	bool weighting = false;
	float super = 4;
	std::vector<int> its_per_step = { 500, 500, 500, 500, 500, 500, 500, 300 };



	MRCImage<float> im = MRCImage<float>::readAs(inVol);
	MRCImage<float> mask = MRCImage<float>::readAs(inMask);
	if (!fs::exists(outdirBase.c_str())) {
		fs::create_directories(outdirBase.c_str());
	}
	int3 dims = { im().xdim, im().ydim, im().zdim };
	//float super = 4.0;



	//float3 *angles;
	MultidimArray<RDOUBLE> refProjections;
	ProjectionSet * refProjectionSet = readProjectionsAndAngles(inProj, 1);
	char bufferN[100];
	char bufferSuper[100];
	char bufferIn[1000];
	char bufferMask[1000];
	sprintf(bufferN, "%d", N);
	sprintf(bufferIn, "%s", inVol.c_str());
	sprintf(bufferMask, "%s", inMask.c_str());
	sprintf(bufferSuper, "%lf", super);
	ProgVolumeToPseudoatoms initializer(21, new char*[21]{ "consecutive_rastering","-i", bufferIn, "-o", "foo", "--initialSeeds", bufferN, "--oversampling", bufferSuper, "--mask", "binary_file",
		"--maskfile", bufferMask, "--InterpolateValues", "true", "--dontAllowMovement",  "true", "--dontAllowIntensity", "true", "--dontAllowNumberChange", "false" });

	MultidimArray<RFLOAT> rastered;
	Pseudoatoms * Atoms;
	
	if (fs::exists((outdirBase + "original.tsv").c_str())){
		Atoms = Pseudoatoms::ReadTsvFile(outdirBase + "original.tsv");
		Atoms->initGrid(dims, 0.2);
	}
	else {
		initializer.run();
		Atoms = new Pseudoatoms(initializer.Atoms);
		Atoms->initGrid(dims, 0.2);
		Atoms->writeTsvFile(outdirBase + "original.tsv");
	}
	if (!fs::exists(outdirBase + "original.mrc")){
		Atoms->RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
		{
			MRCImage<float> out(rastered);
			out.writeAs<float>(outdirBase + "original.mrc");
		}
		writeFSC(im()*mask(), rastered*mask(), std::string(outdirBase) + "original_fsc_masked.star");
		writeFSC(im(), rastered, std::string(outdirBase) + "original_fsc.star");
		writeProjectionsToDisk(Atoms, refProjectionSet->angles, refProjectionSet->numProj, super, dims, outdirBase + "original_proj.mrc");
	}
	outdirBase += "non_linear_";

	std::vector<MultidimArray<RFLOAT>> refVolumes;
	for (auto step : steps) {
		MultidimArray<RFLOAT> refVol;
		Atoms->RasterizeToVolume(refVol, dims, step, false, weighting);
		refVolumes.emplace_back(refVol);
	}

	//for (float diff : diffList) {
	{
		FileName outdir = outdirBase + "ordered_movement_" + std::to_string(diff) + "_weighting_" + (weighting ? "true" : "false") + "_" + std::to_string(super) + "_" + std::to_string(N / 1000) + "\\";
		_mkdir(outdir.c_str());

		static std::default_random_engine e;
		e.seed(42); //Reproducible results for testing
		static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1

		for (size_t i = 0; i < Atoms->NAtoms; i++)
		{
			//float3 distance = { (dis(e) * 2 - 1)*diff,(dis(e) * 2 - 1)*diff, (dis(e) * 2 - 1)*diff };
			float3 distance = { diff * pow(Atoms->AtomPositions[i].z / dims.z,2),diff * pow(Atoms->AtomPositions[i].z / dims.z,2) ,0 };
			Atoms->AtomPositions[i] = Atoms->AtomPositions[i] + distance;
		}

		Atoms->RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
		{
			MRCImage<float> out(rastered);
			out.writeAs<float>(outdir + "moved.mrc");
		}
		writeProjectionsToDisk(Atoms, refProjectionSet->angles, refProjectionSet->numProj, super, dims, outdir + "moved_proj.mrc");
		writeFSC(im()*mask(), rastered*mask(), std::string(outdir) + "moved_fsc_masked.star");
		writeFSC(im(), rastered, std::string(outdir) + "moved_fsc.star");
		//writeProjectionsToDisk(Atoms, angles, numProj, super, dims, outdir + "moved_proj.mrc");
		Atoms->writeTsvFile(outdir + "moved.tsv");

		//Try to correct

		for (int i = 0; i < steps.size(); i++) {
			AtomMover mover(Atoms, refVolumes[i], dims, steps[i], false, weighting, 0.01);
			ADAM_Solver adam_solver(0.001);
			int numIt = its_per_step[i];

			//lbfgs_solver.run(mover, 500);
			adam_solver.run(mover, numIt, outdir + "moved_" + std::to_string(numIt) + "_" + std::to_string(i) + ".log");

			Atoms->RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
			{
				MRCImage<float> out(rastered);
				out.writeAs<float>(outdir + "moved_" + std::to_string(numIt) + "_" + std::to_string(i) + ".mrc");
			}
			writeFSC(im()*mask(), rastered*mask(), std::string(outdir) + "moved_" + std::to_string(numIt) + "_" + std::to_string(i) + "_fsc_masked.star");
			writeFSC(im(), rastered, std::string(outdir) + "moved_" + std::to_string(numIt) + "_" + std::to_string(i) + "_fsc.star");
			//writeProjectionsToDisk(Atoms, refProjectionSet->angles, refProjectionSet->numProj, super, dims, outdir + "moved_" + std::to_string(numIt) + "_" + std::to_string(i) + "_proj.mrc");
			Atoms->writeTsvFile(outdir + "moved_" + std::to_string(numIt) + "_" + std::to_string(i) + ".tsv");
		}

		/*
		//First using 1 ref
		{
			AtomMover mover(Atoms, refVolume1, dims, 1.0, false, weighting, 0.01);
			ADAM_Solver adam_solver;
			int numIt = 300;

			//lbfgs_solver.run(mover, 500);
			adam_solver.run(mover, numIt, outdir + "moved_" + std::to_string(numIt) + "_1.log");

			Atoms->RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
			{
				MRCImage<float> out(rastered);
				out.writeAs<float>(outdir + "moved_" + std::to_string(numIt) + "_1.mrc");
			}
			writeFSC(im()*mask(), rastered*mask(), std::string(outdir) + "moved_" + std::to_string(numIt) + "_1_fsc_masked.star");
			writeFSC(im(), rastered, std::string(outdir) + "moved_" + std::to_string(numIt) + "_1_fsc.star");
			writeProjectionsToDisk(Atoms, angles, numProj, super, dims, outdir + "moved_" + std::to_string(numIt) + "_1_proj.mrc");
			Atoms->writeTsvFile(outdir + "moved_" + std::to_string(numIt) + "_1.tsv");
		}

		//First using 2 ref
		{
			AtomMover mover(Atoms, refVolume2, dims, 2.0, false, weighting, 0.01);
			ADAM_Solver adam_solver;
			int numIt = 300;

			//lbfgs_solver.run(mover, 500);
			adam_solver.run(mover, numIt, outdir + "moved_" + std::to_string(numIt) + "_2.log");

			Atoms->RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
			{
				MRCImage<float> out(rastered);
				out.writeAs<float>(outdir + "moved_" + std::to_string(numIt) + "_2.mrc");
			}
			writeFSC(im()*mask(), rastered*mask(), std::string(outdir) + "moved_" + std::to_string(numIt) + "_2_fsc_masked.star");
			writeFSC(im(), rastered, std::string(outdir) + "moved_" + std::to_string(numIt) + "_2_fsc.star");
			writeProjectionsToDisk(Atoms, angles, numProj, super, dims, outdir + "moved_" + std::to_string(numIt) + "_2_proj.mrc");
			Atoms->writeTsvFile(outdir + "moved_" + std::to_string(numIt) + "_2.tsv");
		}

		{
			AtomMover mover(Atoms, refVolume4, dims, 4.0, false, weighting, 0.01);
			ADAM_Solver adam_solver;
			int numIt = 300;

			//lbfgs_solver.run(mover, 500);
			adam_solver.run(mover, numIt, outdir + "moved_" + std::to_string(numIt) + "_4.log");

			Atoms->RasterizeToVolume(rastered, { (int)im().xdim, (int)im().ydim, (int)im().zdim }, super, true, weighting);
			{
				MRCImage<float> out(rastered);
				out.writeAs<float>(outdir + "moved_" + std::to_string(numIt) + "_4.mrc");
			}
			writeFSC(im()*mask(), rastered*mask(), std::string(outdir) + "moved_" + std::to_string(numIt) + "_4_fsc_masked.star");
			writeFSC(im(), rastered, std::string(outdir) + "moved_" + std::to_string(numIt) + "_4_fsc.star");
			writeProjectionsToDisk(Atoms, angles, numProj, super, dims, outdir + "moved_50_4_proj.mrc");
			Atoms->writeTsvFile(outdir + "moved_" + std::to_string(numIt) + "_4.tsv");
		}
		*/
	}
	
}
