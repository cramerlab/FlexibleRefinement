


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






void doNonMoved() {
	
	std::vector<int> nList = { 30000, 40000, 50000, 60000, 75000, 85000 };
	//for (auto N : nList)
	{
		idxtype N = 75000;
		FileName pixsize = "1.5";
		idxtype batchSize = 64;
		idxtype numIt = 15;
		FileName starFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".projections_uniform.star";
		FileName refFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".mrc";
		FileName refMaskFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_mask.mrc";
		FileName refReconFileName = starFileName.withoutExtension() + ".WARP_recon.mrc";
		//FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_" + std::to_string(N / 1000) + "k.pdb";		//PDB File containing pseudo atom coordinates
		FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15.pdb";		//PDB File containing pseudo atom coordinates
		
		FileName fnOut = pdbFileName.withoutExtension() + "_bs" + std::to_string(batchSize);
		MRCImage<DOUBLE> origVol = MRCImage<DOUBLE>::readAs(refFileName);
		MRCImage<DOUBLE> origMasked = MRCImage<DOUBLE>::readAs(refFileName);
		MRCImage<DOUBLE> Mask = MRCImage<DOUBLE>::readAs(refMaskFileName);
		origMasked.setData(origMasked()*Mask());
		origMasked.writeAs<float>(refFileName.withoutExtension() + "_masked.mrc", true);

		MRCImage<DOUBLE> refReconVol = MRCImage<DOUBLE>::readAs(refReconFileName);

		writeFSC(origVol(), refReconVol(), refReconFileName.withoutExtension() + ".fsc");
		writeFSC(origMasked(), refReconVol()*Mask(), refReconFileName.withoutExtension() + "_masked.fsc");

		idxtype numThreads = 24;
		omp_set_num_threads(numThreads);

		bool writeProjections = false;	//Wether or not to write out projections before and after each iteration

		std::ifstream ifs(pdbFileName);

		std::string line;


		DOUBLE sigma = 0.0;
		std::vector<float3> StartAtoms;
		std::vector<DOUBLE> atompositions;
		std::vector<DOUBLE> intensities;
		StartAtoms.reserve(N);
		atompositions.reserve(N);
		intensities.reserve(N);
		while (std::getline(ifs, line)) {
			if (line.rfind("REMARK fixedGaussian") != std::string::npos) {
				sscanf(line.c_str(), "REMARK fixedGaussian %lf\n", &sigma);
			}
			if (line.rfind("ATOM") != std::string::npos) {
				float3 atom;
				DOUBLE intensity;
				sscanf(line.c_str(), "ATOM\t%*d\tDENS\tDENS\t%*d\t%f\t%f\t%f\t%lf\tDENS", &(atom.x), &(atom.y), &(atom.z), &intensity);
				intensities.emplace_back(intensity);
				StartAtoms.emplace_back(atom);
				atompositions.emplace_back(atom.x);
				atompositions.emplace_back(atom.y);
				atompositions.emplace_back(atom.z);
			}
		}

		MetaDataTable MD;
		try {
			long ret = MD.read(starFileName);
		}
		catch (RelionError Err) {
			std::cout << "Could not read file" << std::endl << Err.msg << std::endl;
			exit(EXIT_FAILURE);
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
				DIRECT_A3D_ELEM(projections, randomI, i, j) = im(num - 1, i, j);
			}
			idx++;
		}

		if (writeProjections) {
			MRCImage<DOUBLE> projectionsIM(projections);
			projectionsIM.writeAs<float>(starFileName.withoutExtension() + ".mrc", true);
		}




		PseudoProjector proj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (DOUBLE *)atompositions.data(), intensities.data(), sigma, StartAtoms.size());
		proj.lambdaART = 0.01;
		
		//Make movements

		DOUBLE zMin = std::numeric_limits<DOUBLE>::max();
		DOUBLE zMax = std::numeric_limits<DOUBLE>::lowest();
		for (idxtype n = 0; n < proj.atoms.AtomPositions.size(); n++) {
			zMin = std::min(zMin, (double)proj.atoms.AtomPositions[n](2));
			zMax = std::max(zMax, (double)proj.atoms.AtomPositions[n](2));
		}
		double lower_bound = -1.0;
		double upper_bound = 1.0;
		std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
		std::default_random_engine re;
		for (idxtype tmp = 0; tmp < 10; tmp++)
		{


			double xShift = unif(re);
			double yShift = sqrt(1 - pow(xShift, 2));
			if (unif(re) < 0.0) {
				yShift *= -1;
			}
			double r = unif(re) * 5 + 10;
			for (idxtype n = 0; n < proj.atoms.AtomPositions.size(); n++) {
				proj.atoms.AtomPositions[n].vdata[0] = StartAtoms[n].x + pow(proj.atoms.AtomPositions[n](2) - zMin, 1.5) / (pow(zMax - zMin, 1.5)) * r * xShift;
				proj.atoms.AtomPositions[n].vdata[1] = StartAtoms[n].y + pow(proj.atoms.AtomPositions[n](2) - zMin, 1.5) / (pow(zMax - zMin, 1.5)) * r * yShift;
				//std::cout << "z=" << proj.atomPositions[n](2) << " shift by " << (proj.atomPositions[n](2) - zMin, 2) / (pow(zMax - zMin, 2)) * r * xShift << " , " << (proj.atomPositions[n](2) - zMin, 2) / (pow(zMax - zMin, 2)) * r * yShift << std::endl;
			}
			MRCImage<DOUBLE> *movedIm = proj.create3DImage(2.0);
			movedIm->writeAs<float>("D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\" + std::to_string(tmp) + ".mrc", true);
			delete movedIm;
			proj.writePDB("D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\" + std::to_string(tmp));

		}
		
		/*Make movements*/
		/*
		DOUBLE zMin = std::numeric_limits<DOUBLE>::max();
		DOUBLE zMax = std::numeric_limits<DOUBLE>::lowest();
		for (idxtype n = 0; n < proj.atomPositions.size(); n++) {
			zMin = std::min(zMin, (double)proj.atomPositions[n](2));
			zMax = std::max(zMax, (double)proj.atomPositions[n](2));
		}
		double lower_bound = -1.0;
		double upper_bound = 1.0;
		std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
		std::default_random_engine re;
		for (idxtype tmp = 0; tmp < 10; tmp++)
		{


			double xShift = unif(re);
			double yShift = sqrt(1 - pow(xShift, 2));
			if (unif(re) < 0.0) {
				yShift *= -1;
			}
			double r = unif(re) * 5 + 10;
			for (idxtype n = 0; n < proj.atomPositions.size(); n++) {
				proj.atomPositions[n].vdata[0] = StartAtoms[n].x + pow(proj.atomPositions[n](2) - zMin, 1.5) / (pow(zMax - zMin, 1.5)) * r * xShift;
				proj.atomPositions[n].vdata[1] = StartAtoms[n].y + pow(proj.atomPositions[n](2) - zMin, 1.5) / (pow(zMax - zMin, 1.5)) * r * yShift;
				//std::cout << "z=" << proj.atomPositions[n](2) << " shift by " << (proj.atomPositions[n](2) - zMin, 2) / (pow(zMax - zMin, 2)) * r * xShift << " , " << (proj.atomPositions[n](2) - zMin, 2) / (pow(zMax - zMin, 2)) * r * yShift << std::endl;
			}
			MRCImage<DOUBLE> *movedIm = proj.create3DImage(2.0);
			movedIm->writeAs<float>("D:\\EMD\\9233\\moving\\" + std::to_string(tmp) + ".mrc", true);
			delete movedIm;
			proj.writePDB("D:\\EMD\\9233\\moving\\" + std::to_string(tmp));

		}*/
		/*
		MRCImage<DOUBLE> *initialRep = proj.create3DImage();
		initialRep->writeAs<float>(fnOut + "_initialRep.mrc", true);
		delete initialRep;
		*/
		MRCImage<DOUBLE> *initialRepOversampled = proj.create3DImage(2.0);
		initialRepOversampled->writeAs<float>(fnOut + "_initialRep_oversampled2.mrc", true);
		writeFSC(origVol(), (*initialRepOversampled)(), fnOut + "_initialRep_oversampled2.fsc");
		writeFSC(origMasked(), (*initialRepOversampled)()*Mask(), fnOut + "_initialRep_oversampled2_masked.fsc");
		delete initialRepOversampled;

		MultidimArray<DOUBLE> pseudoProjectionsData;
		MRCImage<DOUBLE> pseudoProjections;
		if (writeProjections) {
			pseudoProjectionsData.resize(numProj, projections.ydim, projections.xdim);
#pragma omp parallel for
			for (int n = 0; n < numProj; n++) {
				proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
			}

			pseudoProjections.setData(pseudoProjectionsData);
			pseudoProjections.writeAs<float>(fnOut + "_initial_pseudoProjections.mrc", true);
			pseudoProjectionsData.coreDeallocate();
			pseudoProjectionsData.coreAllocate();
		}

		// Do Iterative reconstruction
		for (size_t itIdx = 0; itIdx < numIt; itIdx++)
		{


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


			MRCImage<DOUBLE> *after1Itoversample = proj.create3DImage(2.0);
			after1Itoversample->writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled2.mrc", true);
			writeFSC(origVol(), (*after1Itoversample)(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled2.fsc");
			writeFSC(origMasked(), (*after1Itoversample)()*Mask(), fnOut + "_it" + std::to_string(itIdx + 1) + "_oversampled2_masked.fsc");
			delete after1Itoversample;

			MRCImage<DOUBLE> * after1It = proj.create3DImage(1.0);
			after1It->writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + ".mrc", true);
			delete after1It;
			proj.writePDB(fnOut + "_it" + std::to_string(itIdx + 1) + "");
			if (writeProjections) {
#pragma omp parallel for
				for (int n = 0; n < numProj; n++) {
					proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
				}

				pseudoProjections.setData(pseudoProjectionsData);
				pseudoProjections.writeAs<float>(fnOut + "_" + std::to_string(itIdx + 1) + "stit_pseudoProjections.mrc", true);
				pseudoProjectionsData.coreDeallocate();
				pseudoProjectionsData.coreAllocate();
			}
		}
	}
	/*
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
	writeFSC(origVol(), (*after2Itoversample)(), fnOut + "_it2_oversampled2.fsc");
	delete after2Itoversample;

	MRCImage<DOUBLE> *after2It = proj.create3DImage(1.0);
	after2It->writeAs<float>(fnOut + "_it2.mrc", true);
	delete after2It;
	proj.writePDB(fnOut + "_it2");
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
	writeFSC(origVol(), (*after3Itoversample)(), fnOut + "_it3_oversampled2.fsc");
	delete after3Itoversample;

	MRCImage<DOUBLE> *after3It = proj.create3DImage(1.0);
	after3It->writeAs<float>(fnOut + "_it3.mrc", true);
	delete after3It;
	proj.writePDB(fnOut + "_it3");
	if (writeProjections) {
#pragma omp parallel for
		for (int n = 0; n < numProj; n++) {
			proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
		}

		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_3rdit_pseudoProjections.mrc");
	}*/
}


enum Algo {ART=0, SIRT=1};

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
	
	doNonMoved();
	exit(1);
	
	Algo algorithm = SIRT;
	idxtype batchSize = 64;
	idxtype numIt = 10;
	idxtype numMovements = 9;
	idxtype projPerMovement = 1024;
	FileName starFileName = "D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\projections_uniform.star";
	FileName refFileName = "D:\\EMD\\9233\\emd_9233_Scaled_1.5.mrc";
	FileName refPDB = "D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k.pdb";
	FileName refReconFileName = starFileName.withoutExtension() + ".WARP_recon.mrc";
	FileName startPDB = "D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15.pdb";
	//FileName refPDB = "D:\\EMD\\9233\\emd_9233_Scaled_1.5_40k_bs64_it3.pdb";
	FileName pdbFileDirName = "D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\";		//PDB File containing pseudo atom coordinates
	FileName fnOut = pdbFileDirName + "out_" + (algorithm == ART ? "ART" : "SIRT") + "_lbd0.0001_bs" + std::to_string(batchSize);
	MRCImage<DOUBLE> origVol = MRCImage<DOUBLE>::readAs(refFileName);
	MRCImage<DOUBLE> Mask = MRCImage<DOUBLE>::readAs(refFileName.withoutExtension() + "_mask.mrc");
	MRCImage<DOUBLE> origVolMasked = MRCImage<DOUBLE>::readAs(refFileName);
	origVolMasked.setData(Mask()*origVol());
	MRCImage<DOUBLE> refReconVol = MRCImage<DOUBLE>::readAs(refReconFileName);

	writeFSC(origVol(), refReconVol(), refReconFileName.withoutExtension() + ".fsc");
	writeFSC(origVolMasked(), refReconVol()*Mask(), refReconFileName.withoutExtension() + "_masked.fsc");


	idxtype numThreads = 24;

	omp_set_num_threads(numThreads);

	bool writeProjections = false;	//Wether or not to write out projections before and after each iteration

	DOUBLE sigma = 0.0;
	std::vector<float3> StartAtoms;
	std::vector<DOUBLE> StartAtompositionsFlat;
	std::vector<DOUBLE> StartIntensities;

	std::vector< Matrix1D<DOUBLE> > StartAtomPositionsMatrix;
	/*Reading initial PDB*/
	{
		std::ifstream ifs(startPDB);

		std::string line;
		StartAtoms.reserve(75000);
		StartAtompositionsFlat.reserve(75000);
		StartIntensities.reserve(75000);
		StartAtomPositionsMatrix.reserve(75000);
		while (std::getline(ifs, line)) {
			if (line.rfind("REMARK fixedGaussian") != std::string::npos) {
				sscanf(line.c_str(), "REMARK fixedGaussian %lf\n", &sigma);
			}
			if (line.rfind("ATOM") != std::string::npos) {
				float3 atom;
				DOUBLE intensity;
				sscanf(line.c_str(), "ATOM\t%*d\tDENS\tDENS\t%*d\t%f\t%f\t%f\t%lf\tDENS", &(atom.x), &(atom.y), &(atom.z), &intensity);
				StartIntensities.emplace_back(intensity);
				StartAtoms.emplace_back(atom);
				StartAtompositionsFlat.emplace_back(atom.x);
				StartAtompositionsFlat.emplace_back(atom.y);
				StartAtompositionsFlat.emplace_back(atom.z);
				Matrix1D<DOUBLE> a(3);
				a(0) = atom.x;
				a(1) = atom.y;
				a(2) = atom.z;
				StartAtomPositionsMatrix.push_back(a);

			}
		}
	}

	std::vector<float3> RefAtoms;
	std::vector<DOUBLE> RefAtompositionsFlat;
	std::vector<DOUBLE> RefIntensities;
	std::vector<Matrix1D<DOUBLE>> RefAtompositionsMatrix;
	//Read ref pdb
	{
		std::ifstream ifs(refPDB);

		std::string line;
		RefAtoms.reserve(50000);
		RefAtompositionsFlat.reserve(50000);
		RefIntensities.reserve(50000);
		RefAtompositionsMatrix.reserve(50000);
		while (std::getline(ifs, line)) {
			if (line.rfind("REMARK fixedGaussian") != std::string::npos) {
				sscanf(line.c_str(), "REMARK fixedGaussian %lf\n", &sigma);
			}
			if (line.rfind("ATOM") != std::string::npos) {
				float3 atom;
				DOUBLE intensity;
				sscanf(line.c_str(), "ATOM\t%*d\tDENS\tDENS\t%*d\t%f\t%f\t%f\t%lf\tDENS", &(atom.x), &(atom.y), &(atom.z), &intensity);
				RefIntensities.emplace_back(intensity);
				RefAtoms.emplace_back(atom);
				RefAtompositionsFlat.emplace_back(atom.x);
				RefAtompositionsFlat.emplace_back(atom.y);
				RefAtompositionsFlat.emplace_back(atom.z);
				Matrix1D<DOUBLE> a(3);
				a(0) = atom.x;
				a(1) = atom.y;
				a(2) = atom.z;
				RefAtompositionsMatrix.push_back(a);

			}
		}
	}

	/* reading moved pdbs*/
	std::vector<std::vector< Matrix1D<DOUBLE> >> atomPosition;
	for (idxtype i = 0; i < numMovements; i++) {
		atomPosition.push_back(std::vector<Matrix1D<DOUBLE>>());

		std::ifstream ifs(pdbFileDirName + std::to_string(i) + ".pdb");

		std::string line;

		while (std::getline(ifs, line)) {
			if (line.rfind("REMARK fixedGaussian") != std::string::npos) {
				sscanf(line.c_str(), "REMARK fixedGaussian %lf\n", &sigma);
			}
			if (line.rfind("ATOM") != std::string::npos) {
				float3 atom;
				DOUBLE intensity;
				sscanf(line.c_str(), "ATOM\t%*d\tDENS\tDENS\t%*d\t%f\t%f\t%f\t%lf\tDENS", &(atom.x), &(atom.y), &(atom.z), &intensity);

				Matrix1D<DOUBLE> a(3);
				a(0) = atom.x;
				a(1) = atom.y;
				a(2) = atom.z;
				atomPosition[i].emplace_back(a);
			}
		}
	}
	MetaDataTable MD;
	try {
		long ret = MD.read(starFileName);
	}
	catch (RelionError Err) {
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
	//std::shuffle(std::begin(idxLookup), std::end(idxLookup), rng);
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
			DIRECT_A3D_ELEM(projections, randomI, i, j) = im(num - 1, i, j);
		}
		idx++;
	}



	if (writeProjections) {
		MRCImage<DOUBLE> projectionsIM(projections);
		projectionsIM.writeAs<float>(starFileName.withoutExtension() + ".mrc", true);
	}


	PseudoProjector proj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (DOUBLE *)StartAtompositionsFlat.data(), RefIntensities.data(), sigma, StartAtoms.size());
	PseudoProjector RefProj(make_int3((int)(projections.xdim), (int)(projections.xdim), (int)(projections.xdim)), (DOUBLE *)RefAtompositionsFlat.data(), StartIntensities.data(), sigma, RefAtoms.size());
	proj.lambdaART = 0.0001;


	
	proj.setAtomPositions(RefAtompositionsMatrix);
	/*
	MRCImage<DOUBLE> *initialRep = proj.create3DImage();
	initialRep->writeAs<float>(fnOut + "_initialRep.mrc", true);
	delete initialRep;
	*/
	MRCImage<DOUBLE> *initialRepOversampled = proj.create3DImage(2.0);
	initialRepOversampled->writeAs<float>(fnOut + "_initialRep_oversampled2.mrc", true);
	writeFSC(origVol(), (*initialRepOversampled)(), fnOut + "_initialRep_oversampled2.fsc");
	writeFSC(origVolMasked(), (*initialRepOversampled)()*Mask(), fnOut + "_initialRep_oversampled2_masked.fsc");
	delete initialRepOversampled;
	/*
	MRCImage<DOUBLE> *RefRepOversampled = RefProj.create3DImage(2.0);
	RefRepOversampled->writeAs<float>(fnOut + "_RefRep_oversampled2.mrc", true);
	writeFSC(origVol(), (*RefRepOversampled)(), fnOut + "_RefRep_oversampled2.fsc");
	writeFSC(origVolMasked(), Mask()*(*RefRepOversampled)(), fnOut + "_RefRep_oversampled2_masked.fsc");
	delete RefRepOversampled;
	*/



	MultidimArray<DOUBLE> pseudoProjectionsData;
	MRCImage<DOUBLE> pseudoProjections;





	std::vector<projecction> precalc;
	if (algorithm == SIRT) {
		for (idxtype n = 0; n < atomPosition.size(); n++) {
			std::vector<float3> ang;
			for (size_t i = 0; i < projPerMovement; i++)
			{
				ang.emplace_back(angles[n*projPerMovement + i]);
			}
			idxtype zStart = n * projPerMovement;
			idxtype zEnd = (n + 1)*projPerMovement;
			MultidimArray<DOUBLE> slice;
			slice.xdim = projections.xdim;
			slice.ydim = projections.ydim;
			slice.zdim = std::min(projections.zdim, (long)zEnd) - zStart;

			slice.yxdim = projections.yxdim;
			slice.nzyxdim = slice.xdim * slice.ydim * slice.zdim;
			slice.nzyxdimAlloc = slice.nzyxdim;
			slice.destroyData = false;
			slice.xinit = projections.xinit;
			slice.yinit = projections.yinit;
			slice.data = &(DIRECT_A3D_ELEM(projections, zStart, 0, 0));
			slice.destroyData = false;
			proj.addToPrecalcs(precalc, slice, ang, &(atomPosition[n]), 0, 0);
		}
	}


	if (writeProjections) {
		pseudoProjectionsData.initZeros(numProj, projections.ydim, projections.xdim);

		idxtype positionIdx = 0;
		for (idxtype batchIdx = 0; batchIdx < numProj;) {

			proj.setAtomPositions(atomPosition[positionIdx]);
			positionIdx++;
#pragma omp parallel for
			for (int n = 0; n < projPerMovement; n++) {
				//proj.setAtomPositions(RefAtompositionsMatrix);
				proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*(batchIdx + n), NULL, angles[batchIdx + n], 0.0, 0.0, PSEUDO_FORWARD);
			}
			batchIdx += projPerMovement;
		}

		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_initial_pseudoProjections.mrc", true);
		pseudoProjectionsData.coreDeallocate();
		pseudoProjectionsData.coreAllocate();
		pseudoProjectionsData.initZeros(numProj, projections.ydim, projections.xdim);
	}

	if (writeProjections) {

//		idxtype positionIdx = 0;
//		for (idxtype batchIdx = 0; batchIdx < numProj;) {
//			/*
//			RefProj.setAtomPositions(atomPosition[positionIdx]);
//			MRCImage<DOUBLE> *RefRepMoved = RefProj.create3DImage(2.0);
//			RefRepMoved->writeAs<float>(fnOut + "_RefRepMoved_" + std::to_string(positionIdx) + "_oversampled2.mrc", true);
//
//			delete RefRepMoved;
//			*/
//			positionIdx++;
//#pragma omp parallel for
//			for (int n = 0; n < projPerMovement; n++) {
//				//proj.setAtomPositions(RefAtompositionsMatrix);
//				RefProj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*(batchIdx + n), NULL, angles[batchIdx + n], 0.0, 0.0, PSEUDO_FORWARD);
//			}
//			batchIdx += projPerMovement;
//		}
#pragma omp parallel
		{
			MultidimArray<DOUBLE> ITheo(projections.ydim, projections.xdim);
			ITheo.coreDeallocate();
			MultidimArray<DOUBLE> INorm(projections.ydim, projections.xdim);
			ITheo.destroyData = false;
#pragma omp for
			for (int n = 0; n < precalc.size(); n++)
			{
				ITheo.data = pseudoProjectionsData.data + pseudoProjectionsData.yxdim * n;

				RefProj.project_Pseudo(ITheo, INorm, precalc[n].atomPositons, precalc[n].Euler, 0.0, 0.0, PSEUDO_FORWARD);

			}
		}
		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_ref_pseudoProjections.mrc", true);
		pseudoProjectionsData.coreDeallocate();
		pseudoProjectionsData.coreAllocate();
		pseudoProjectionsData.initZeros(numProj, projections.ydim, projections.xdim);
	}





	{

		MultidimArray<DOUBLE> Itheo(projections.xdim, projections.ydim);
		MultidimArray<DOUBLE> Icorr(projections.xdim, projections.ydim);
		std::shuffle(std::begin(idxLookup), std::end(idxLookup), rng);
		for (size_t tmp = 0; tmp < 5; tmp++)
		{
			Itheo.initZeros();
			idxtype idx = idxLookup[tmp];
			proj.project_Pseudo(Itheo, Icorr, precalc[idx].atomPositons, precalc[idx].Euler, 0.0, 0.0, PSEUDO_FORWARD);
			DOUBLE sumProj = 0.0;
			DOUBLE sumRef = 0.0;
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Itheo) {
				sumProj += DIRECT_A3D_ELEM(Itheo, k, i, j);
				sumRef += DIRECT_A3D_ELEM(projections, idx, i, j);
			}
			std::cout << "sumProj: " << sumProj << std::endl;
			std::cout << "sumRef: " << sumRef << std::endl;
			DOUBLE corrFactor = sumRef / sumProj;
			for (size_t i = 0; i < proj.atoms.AtomWeights.size(); i++)
			{
				proj.atoms.AtomWeights[i] = proj.atoms.AtomWeights[i] * corrFactor;
			}
		}
	}

	for (idxtype itIdx = 0; itIdx < numIt; itIdx++) {
		idxtype processed = 0;
		idxtype positionIdx = 0;
		// precalculate intensity offset

		// ART
		if (algorithm == ART) {
			if (batchSize > 1)
			{
				for (int batchidx = 0; batchidx < (int)(numProj / ((float)batchSize) + 0.5); batchidx++) {
					if (processed % (projPerMovement) == 0) {
						proj.setAtomPositions(atomPosition[positionIdx]);
						positionIdx++;
					}
					int numLeft = std::min((int)numProj - batchidx * batchSize, batchSize);

					MultidimArray<DOUBLE> slice = projections.getZSlices(batchidx * batchSize, batchidx * batchSize + numLeft);
					if (numLeft != 0) {
						proj.ART_batched(slice, numLeft, angles + batchidx * batchSize, 0.0, 0.0);
					}
					processed += numLeft;

				}
			}
			else
			{
				proj.ART_multi_Image_step(projections.data, angles, 0.0, 0.0, numProj);
			}
		}
		
		//SIRT
		if (algorithm == SIRT) {
			if (writeProjections) {
				MultidimArray<DOUBLE> Itheo, Icorr, Idiff, Inorm;
				MRCImage<DOUBLE> im;
				proj.SIRT_from_precalc(precalc, Itheo, Icorr, Idiff, Inorm, 0.0, 0.0);
				im.setData(Itheo);
				im.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_Itheo.mrc");
				im.setData(Icorr);
				im.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_Icorr.mrc");
				im.setData(Idiff);
				im.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_Idiff.mrc");
				im.setData(Inorm);
				im.writeAs<float>(fnOut + "_it" + std::to_string(itIdx + 1) + "_Inorm.mrc");
			}
			else
				proj.SIRT_from_precalc(precalc, 0, 0);

			proj.lambdaART = proj.lambdaART / sqrt(2);
		}
		proj.setAtomPositions(RefAtompositionsMatrix);
		MRCImage<DOUBLE> *after1Itoversample = proj.create3DImage(3.0);
		after1Itoversample->writeAs<float>(fnOut + "_it" + std::to_string(itIdx+1)+ "_oversampled3.mrc", true);
		writeFSC(origVol(), (*after1Itoversample)(), fnOut + "_it" + std::to_string(itIdx+1) + "_oversampled3.fsc");
		writeFSC(origVolMasked(), Mask()*(*after1Itoversample)(), fnOut + "_it" + std::to_string(itIdx+1) + "_oversampled3_masked.fsc");
		delete after1Itoversample;

		//MRCImage<DOUBLE> * after1It = proj.create3DImage(1.0);
		//after1It->writeAs<float>(fnOut + "_it" + std::to_string(itIdx) + ".mrc", true);
		//delete after1It;
		proj.writePDB(fnOut + "_it" + std::to_string(itIdx+1));

		if (false && writeProjections) {

			idxtype positionIdx = 0;
			for (idxtype batchIdx = 0; batchIdx < numProj;) {

				proj.setAtomPositions(atomPosition[positionIdx]);
				positionIdx++;
#pragma omp parallel for
				for (int n = 0; n < 1024; n++) {
					//proj.setAtomPositions(RefAtompositionsMatrix);
					proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*(batchIdx + n), NULL, angles[batchIdx + n], 0.0, 0.0, PSEUDO_FORWARD);
				}
				batchIdx += 1024;
			}

			pseudoProjections.setData(pseudoProjectionsData);
			pseudoProjections.writeAs<float>(fnOut + "_" + std::to_string(itIdx+1) + "stit_pseudoProjections.mrc", true);
			pseudoProjectionsData.coreDeallocate();
			pseudoProjectionsData.coreAllocate();
			pseudoProjectionsData.initZeros(numProj, projections.ydim, projections.xdim);
		}
	}
	/*
	processed = 0;
	positionIdx = 0;
	if (batchSize > 1)
	{
		for (int batchidx = 0; batchidx < (int)(numProj / ((float)batchSize) + 0.5); batchidx++) {
			if (processed % 1024 == 0) {
				proj.setAtomPositions(atomPosition[positionIdx]);
				positionIdx++;
			}
			int numLeft = std::min((int)numProj - batchidx * batchSize, batchSize);
			MultidimArray<DOUBLE> slice = projections.getZSlices(batchidx * batchSize, batchidx * batchSize + numLeft);
			if (numLeft != 0) {
				proj.ART_batched(slice, numLeft, angles + batchidx * batchSize, 0.0, 0.0);
			}
			processed += numLeft;

		}
	}
	else
	{
		proj.ART_multi_Image_step(projections.data, angles, 0.0, 0.0, numProj);
	}
	proj.setAtomPositions(StartAtomPositions);
	MRCImage<DOUBLE> *after2Itoversample = proj.create3DImage(2.0);
	after2Itoversample->writeAs<float>(fnOut + "_it2_oversampled2.mrc");
	writeFSC(origVol(), (*after2Itoversample)(), fnOut + "_it2_oversampled2.fsc");
	delete after2Itoversample;

	MRCImage<DOUBLE> *after2It = proj.create3DImage(1.0);
	after2It->writeAs<float>(fnOut + "_it2.mrc", true);
	delete after2It;
	proj.writePDB(fnOut + "_it2");
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
	processed = 0;
	positionIdx = 0;
	if (batchSize > 1)
	{
		for (int batchidx = 0; batchidx < (int)(numProj / ((float)batchSize) + 0.5); batchidx++) {
			if (processed % 1024 == 0) {
				proj.setAtomPositions(atomPosition[positionIdx]);
				positionIdx++;
			}
			int numLeft = std::min((int)numProj - batchidx * batchSize, batchSize);
			MultidimArray<DOUBLE> slice = projections.getZSlices(batchidx * batchSize, batchidx * batchSize + numLeft);
			if (numLeft != 0) {
				proj.ART_batched(slice, numLeft, angles + batchidx * batchSize, 0.0, 0.0);
			}
			processed += numLeft;

		}
	}
	else
	{
		proj.ART_multi_Image_step(projections.data, angles, 0.0, 0.0, numProj);
	}
	proj.setAtomPositions(StartAtomPositions);
	MRCImage<DOUBLE> *after3Itoversample = proj.create3DImage(2.0);
	after3Itoversample->writeAs<float>(fnOut + "_it3_oversampled2.mrc", true);
	writeFSC(origVol(), (*after3Itoversample)(), fnOut + "_it3_oversampled2.fsc");
	delete after3Itoversample;

	MRCImage<DOUBLE> *after3It = proj.create3DImage(1.0);
	after3It->writeAs<float>(fnOut + "_it3.mrc", true);
	delete after3It;
	proj.writePDB(fnOut + "_it3");
	if (writeProjections) {
#pragma omp parallel for
		for (int n = 0; n < numProj; n++) {
			proj.project_Pseudo(pseudoProjectionsData.data + pseudoProjectionsData.yxdim*n, NULL, angles[n], 0.0, 0.0, PSEUDO_FORWARD);
		}

		pseudoProjections.setData(pseudoProjectionsData);
		pseudoProjections.writeAs<float>(fnOut + "_3rdit_pseudoProjections.mrc");
	}
	*/


	
}