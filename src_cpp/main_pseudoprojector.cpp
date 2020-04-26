


#include "PseudoProjector.h"
#include "cxxopts.hpp"
#include <string>
#include "metadata_table.h"
#include "readMRC.h"
#include <stdio.h>
#include <algorithm>
#include <random>
#include <omp.h>
#define WRITE_PROJECTIONS
struct params {

	std::string inputPDB;
	std::string inputStar;
};
void defineParams(cxxopts::Options &options)
{

	options.add_options()
		("i,input", "Input", cxxopts::value<std::string>(), "Input PDB file of atoms")
		("o,output", "rootname", cxxopts::value<std::string>(), "Rootname for output")
		("sigma", "s", cxxopts::value<DOUBLE>()->default_value("1.5"), "Sigma of Gaussians used")
		("initialSeeds", "N", cxxopts::value<size_t>()->default_value("300"), "Initial number of Atoms")
		("growSeeds", "percentage", cxxopts::value<size_t>()->default_value("30"), "Percentage of growth, At each iteration the smallest percentage/2 pseudoatoms will be removed, and percentage new pseudoatoms will be created.")
		("filterInput", "f", cxxopts::value<DOUBLE>(), "Low-pass filter input using this threshold")
		("stop", "p", cxxopts::value<DOUBLE>()->default_value("0.001"), "Stop criterion (0<p<1) for inner iterations. At each iteration the current number of gaussians will be optimized until the average error does not decrease at least this amount relative to the previous iteration.")
		("targetError", "p", cxxopts::value<DOUBLE>()->default_value("0.02"), "Finish when the average representation error is below this threshold (in percentage; by default, 2%)")
		("dontAllowMovement", "true", cxxopts::value<bool>()->default_value("false"), "Don't allow pseudoatoms to move")
		("dontAllowIntensity", "f", cxxopts::value<DOUBLE>()->default_value("0.01"), "Don't allow pseudoatoms to change intensity. f determines the fraction of intensity")
		("intensityColumn", "s", cxxopts::value<std::string>()->default_value("Bfactor"), "Where to write the intensity in the PDB file")
		("Nclosest", "N", cxxopts::value<size_t>()->default_value("3"), "N closest atoms, it is used only for the distance histogram")
		("minDistance", "d", cxxopts::value<DOUBLE>()->default_value("0.001"), "Minimum distance between two pseudoatoms (in Angstroms). Set it to -1 to disable")
		("penalty", "p", cxxopts::value<DOUBLE>()->default_value("10"), "Penalty for overshooting")
		("sampling_rate", "Ts", cxxopts::value<DOUBLE>()->default_value("1"), "Sampling rate Angstroms/pixel")
		("dontScale", "true", cxxopts::value<bool>()->default_value("false"), "Don't scale atom weights in the PDB")
		("binarize", "threshold", cxxopts::value<DOUBLE>()->default_value("0.5"), "Binarize the volume")
		("thr", "t", cxxopts::value<size_t>()->default_value("1"), "Number of Threads")
		("mask", "mask_type", cxxopts::value<std::string>(), "Which mask type to use. Options are real_file and binary_file")
		("maskfile", "f", cxxopts::value<std::string>(), "Path of mask file")
		("center", "c", cxxopts::value<std::vector<DOUBLE>>()->default_value("0,0,0"), "Center of Mask")
		("v,verbose", "v", cxxopts::value<int>()->default_value("0"), "Verbosity Level");
}

params readParams(cxxopts::ParseResult &res) {
	
	return { "","" };

}



int main(int argc, char** argv) {

	cxxopts::Options options(argv[0], " - example command line options");
	options
		.positional_help("[optional args]")
		.show_positional_help();
	defineParams(options);
	try
	{
		cxxopts::ParseResult result = options.parse(argc, argv);
		readParams(result);
	}
	catch (const cxxopts::OptionException& e)
	{
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}
	idxtype batchSize = 64;
	FileName starFileName = "D:\\EMPIAR\\10168\\emd_4180_res7.projectionsConv2_uniform.star";
	FileName pdbFileName = "D:\\EMPIAR\\10168\\emd_4180_res7_5k.pdb";		//PDB File containing pseudo atom coordinates
	FileName fnOut = pdbFileName.withoutExtension() +"_conv_bs" + std::to_string(batchSize);
	idxtype numThreads = 24;
	omp_set_num_threads(numThreads);

	bool writeProjections = false;	//Wether or not to write out projections before and after each iteration

	std::ifstream ifs(pdbFileName);

	std::string line;

	
	DOUBLE sigma = 0.0;
	std::vector<float3> atoms;
	std::vector<DOUBLE> atompositions;
	std::vector<DOUBLE> intensities;
	atoms.reserve(10000);
	atompositions.reserve(10000);
	intensities.reserve(10000);
	while (std::getline(ifs, line)) {
		if (line.rfind("REMARK fixedGaussian") != std::string::npos) {
			sscanf(line.c_str(), "REMARK fixedGaussian %lf\n",&sigma);
		}
		if (line.rfind("ATOM") != std::string::npos) {
			float3 atom;
			DOUBLE intensity;
			sscanf(line.c_str(), "ATOM\t%*d\tDENS\tDENS\t%*d\t%f\t%f\t%f\t%lf\tDENS", &(atom.x), &(atom.y), &(atom.z), &intensity);
			intensities.emplace_back(intensity);
			atoms.emplace_back(atom);
			atompositions.emplace_back(atom.x);
			atompositions.emplace_back(atom.y);
			atompositions.emplace_back(atom.z);
		}
	}

	MetaDataTable MD;
	try {
		long ret = MD.read(starFileName);
	}
	catch (RelionError Err){
		std::cout << "Could not read file" << std::endl << Err.msg << std::endl;
		return EXIT_FAILURE;
	}
	FileName imageName;
	FileName prevImageName = "";
	char imageName_cstr[1000];

	int num;
	idxtype numProj = MD.numberOfObjects();
	
	
	float3 *angles = (float3 *)malloc(sizeof(float3)*numProj);

	idxtype idx = 0;
	MRCImage<DOUBLE> im;
	MultidimArray<DOUBLE> projections;
	
	
	bool isInit = false;
	
	auto rng = std::default_random_engine{};
	std::vector<idxtype> idxLookup;
	idxLookup.reserve(numProj);
	for (idxtype i = 0; i < numProj; i++)
		idxLookup.emplace_back(i);
	std::shuffle(std::begin(idxLookup), std::end(idxLookup), rng);
	
	FOR_ALL_OBJECTS_IN_METADATA_TABLE(MD) {
		MD.getValue(EMDL_IMAGE_NAME, imageName);
		idxtype randomI = idxLookup[idx];	// Write to random position in projections to avoid any bias
		MD.getValue(EMDL_ORIENT_ROT, angles[randomI].x);
		MD.getValue(EMDL_ORIENT_TILT, angles[randomI].y);
		MD.getValue(EMDL_ORIENT_PSI, angles[randomI].z);

		sscanf(imageName.c_str(), "%d@%s", &num, imageName_cstr);
		imageName = imageName_cstr;
		if (imageName != prevImageName) {
			im = MRCImage<DOUBLE>::readAs(imageName);
			if (!isInit) {
				projections.resize(numProj, im().ydim, im().xdim);
				isInit = true;
			}
		}
		prevImageName = imageName;
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(im()) {
			DIRECT_A3D_ELEM(projections, randomI, i, j) = im(num-1, i, j);
		}
		idx++;
	}

#ifdef WRITE_PROJECTIONS
	MRCImage<DOUBLE> projectionsIM(projections);
	projectionsIM.writeAs<float>(starFileName.withoutExtension() + ".mrc", true);
#endif



	PseudoProjector proj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (DOUBLE *)atompositions.data(), intensities.data(), sigma, atoms.size());
	proj.lambdaART = 0.01;

	MRCImage<DOUBLE> *initialRep = proj.create3DImage();
	initialRep->writeAs<float>(fnOut + "_initialRep.mrc", true);
	delete initialRep;

	MRCImage<DOUBLE> *initialRepOversampled = proj.create3DImage(2.0);
	initialRepOversampled->writeAs<float>(fnOut + "_initialRep_oversampled2.mrc", true);
	delete initialRepOversampled;
#ifdef WRITE_PROJECTIONS
	MultidimArray<DOUBLE> pseudoProjectionsData(numProj, projections.ydim, projections.xdim);
#pragma omp parallel for
	for (int n = 0; n < numProj; n++) {
		proj.project_Pseudo(pseudoProjectionsData.data+pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
	}
	
	MRCImage<DOUBLE> pseudoProjections(pseudoProjectionsData);
	pseudoProjections.writeAs<float>(fnOut + "_initial_pseudoProjections.mrc", true);
	pseudoProjectionsData.coreDeallocate();
	pseudoProjectionsData.coreAllocate();
#endif
	if (batchSize > 1)
	{
		for (int batchidx = 0; batchidx < (int)(numProj / ((float)batchSize) + 0.5); batchidx++) {
			int numLeft = std::min((int)numProj - batchidx * batchSize, batchSize);
			MultidimArray<DOUBLE> slice = projections.getZSlices(batchidx * batchSize, batchidx * batchSize + numLeft);
			if (numLeft != 0) {
				proj.ART_batched(slice, numLeft, angles + batchidx * batchSize, 0.0, 0.0);
			}

		}
	}
	else
	{
		proj.ART_multi_Image_step(projections.data, angles, 0.0, 0.0, numProj);
	}

	
	MRCImage<DOUBLE> *after1Itoversample = proj.create3DImage(3.0);
	after1Itoversample->writeAs<float>(fnOut + "_it1_oversampled3.mrc", true);
	delete after1Itoversample;

	MRCImage<DOUBLE> * after1It = proj.create3DImage(1.0);
	after1It->writeAs<float>(fnOut + "_it1.mrc", true);
	delete after1It;

	if (writeProjections) {
#pragma omp parallel for
		for (int n = 0; n < numProj; n++) {
			proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
		}

		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_1stit_pseudoProjections.mrc", true);
		pseudoProjectionsData.coreDeallocate();
		pseudoProjectionsData.coreAllocate();
	}

	if (batchSize > 1)
	{
		for (int batchidx = 0; batchidx < (int)(numProj / ((float)batchSize) + 0.5); batchidx++) {
			int numLeft = std::min((int)numProj - batchidx * batchSize, batchSize);
			MultidimArray<DOUBLE> slice = projections.getZSlices(batchidx * batchSize, batchidx * batchSize + numLeft);
			if (numLeft != 0) {
				proj.ART_batched(slice, numLeft, angles + batchidx * batchSize, 0.0, 0.0);
			}

		}
	}
	else
	{
		proj.ART_multi_Image_step(projections.data, angles, 0.0, 0.0, numProj);
	}
	
	MRCImage<DOUBLE> *after2Itoversample = proj.create3DImage(2.0);
	after2Itoversample->writeAs<float>(fnOut + "_it2_oversampled2.mrc");
	delete after2Itoversample;

	MRCImage<DOUBLE> *after2It = proj.create3DImage(1.0);
	after2It->writeAs<float>(fnOut + "_it2.mrc", true);
	delete after2It;
	if (writeProjections) {
#pragma omp parallel for
		for (int n = 0; n < numProj; n++) {

			proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
		}
		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_2ndit_pseudoProjections.mrc", true);
		pseudoProjectionsData.coreDeallocate();
		pseudoProjectionsData.coreAllocate();
	}

	if (batchSize > 1)
	{
		for (int batchidx = 0; batchidx < (int)(numProj / ((float)batchSize) + 0.5); batchidx++) {
			int numLeft = std::min((int)numProj - batchidx * batchSize, batchSize);
			MultidimArray<DOUBLE> slice = projections.getZSlices(batchidx * batchSize, batchidx * batchSize + numLeft);
			if (numLeft != 0) {
				proj.ART_batched(slice, numLeft, angles + batchidx * batchSize, 0.0, 0.0);
			}

		}
	}
	else
	{
		proj.ART_multi_Image_step(projections.data, angles, 0.0, 0.0, numProj);
	}

	MRCImage<DOUBLE> *after3Itoversample = proj.create3DImage(2.0);
	after3Itoversample->writeAs<float>(fnOut + "_it3_oversampled2.mrc", true);
	delete after3Itoversample;

	MRCImage<DOUBLE> *after3It = proj.create3DImage(1.0);
	after3It->writeAs<float>(fnOut + "_it3.mrc", true);
	delete after3It;

	if (writeProjections) {
#pragma omp parallel for
		for (int n = 0; n < numProj; n++) {
			proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
		}

		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_3rdit_pseudoProjections.mrc");
	}

	proj.writePDB(fnOut + "_final.pdb");
}