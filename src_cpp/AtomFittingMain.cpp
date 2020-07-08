
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

	MRCImage<float> referenceVolume(refFileName);
	int3 Dims = { referenceVolume().xdim, referenceVolume().ydim, referenceVolume().zdim };
	int3 superDims = { (int)(Dims.z * super + 0.5), (int)(Dims.y * super + 0.5), (int)(Dims.x * super + 0.5) };
	MultidimArray<float> superRefVolume = referenceVolume();
	ResizeMapGPU(superRefVolume, superDims);
	std::vector<float3> AtomPositions;
	std::vector<DOUBLE> AtomIntensities;

	idxtype NAtoms = Pseudoatoms::readAtomsFromFile(pdbFileName, AtomPositions, AtomIntensities, N); //actually read number of atoms

	Pseudoatoms atoms(((float*)AtomPositions.data()), ((float*)AtomIntensities.data()), NAtoms, ATOM_INTERPOLATE);

	MultidimArray<float> rasteredVolume;
	atoms.RasterizeToVolume(rasteredVolume, Dims, super, true);

	MultidimArray<float> RefVolume = referenceVolume();

	MultidimArray<float> DiffVolume = RefVolume;
	Substract_GPU(DiffVolume, rasteredVolume);
	{
		MRCImage<float> diffIm(DiffVolume);
		diffIm.writeAs<float>(fnOut + "diffIm.mrc", true);
	}


	return EXIT_SUCCESS;
}