
#include <cstdlib>
#include "readMRC.h"
#include "pseudoatoms.h"

int main(char** argv, int argc) {

	idxtype N = 800000; //Estimate of number of Atoms
	FileName pixsize = "2.0";
	float super = 4.0f;
	FileName starFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".projections_uniform_combined.distorted_10.star";


	FileName refFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + ".mrc";
	FileName refMaskFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_mask.mrc";
	FileName pdbFileName = "D:\\EMD\\9233\\emd_9233_Scaled_" + pixsize + "_" + std::to_string(N / 1000) + "k.pdb";		//PDB File containing pseudo atom coordinates
	FileName fnOut = "D:\\EMD\\9233\\AtomFitting\\";
	/*
	MRCImage<float> referenceVolume(refFileName);
	int3 Dims = { referenceVolume().xdim, referenceVolume().ydim, referenceVolume().zdim };

	int3 superDims = { (int)(Dims.z * super + 0.5), (int)(Dims.y * super + 0.5), (int)(Dims.x * super + 0.5) };
	MultidimArray<float> superRefVolume = referenceVolume();
	ResizeMapGPU(superRefVolume, superDims);
	std::vector<float3> AtomPositions;
	std::vector<DOUBLE> AtomIntensities;

	idxtype NAtoms = Pseudoatoms::readAtomsFromFile(pdbFileName, AtomPositions, AtomIntensities, N); //actually read number of atoms
	*/
	int3 Dims = { 10, 10, 10 };
	float* pseudoPositions = new float[3];
	float* pseudoweights = new float[1];
	pseudoPositions[0] = pseudoPositions[1] = pseudoPositions[2] = 5.125;
	pseudoweights[0] = 1;

	Pseudoatoms atoms(pseudoPositions, pseudoweights, 1, ATOM_INTERPOLATE);
	//Pseudoatoms atoms(((float*)AtomPositions.data()), ((float*)AtomIntensities.data()), NAtoms, ATOM_INTERPOLATE);

	MultidimArray<float> SuperRefVolume;
	atoms.RasterizeToVolume(SuperRefVolume, Dims, super, false);
	{
		MRCImage<float> refIm(SuperRefVolume);
		refIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\SuperRefVol.mrc", true);
	}

	MultidimArray<float> RefVolume;
	atoms.RasterizeToVolume(RefVolume, Dims, super);
	{
		MRCImage<float> refIm(RefVolume);
		refIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\RefVol.mrc", true);
	}
	atoms.AtomPositions[0].x =5.100;


	MultidimArray<float> SuperMovedRefVolume;
	atoms.RasterizeToVolume(SuperMovedRefVolume, Dims, super, false);
	{
		MRCImage<float> refIm(SuperMovedRefVolume);
		refIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\SuperMovedRefVolume.mrc", true);
	}

	MultidimArray<float> MovedRefVolume;
	atoms.RasterizeToVolume(MovedRefVolume, Dims, super);
	{
		MRCImage<float> refIm(MovedRefVolume);
		refIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\MovedRefVolume.mrc", true);
	}


	atoms.MoveAtoms(SuperRefVolume, Dims, super, true);


	return EXIT_SUCCESS;
}