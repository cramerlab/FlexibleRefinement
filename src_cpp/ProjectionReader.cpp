#include "ProjectionReader.h"
#include "liblionImports.h"
#include "metadata_table.h"
#include <random>
#include "Types.h"
#include <filesystem>
#include "readMRC.h"
#include "PseudoProjector.h"

using namespace relion;
namespace fs = std::filesystem;

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

	fs::path parentPath = starFileName.c_str();
	parentPath = parentPath.parent_path();
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
			for (int i = 0; i < imageName.size(); i++) {
				if (imageName[i] == '/') {
					imageName[i] = '\\';
				}
			}
			if (imageName != prevImageName) {
				im = MRCImage<RDOUBLE>::readAs((parentPath / imageName.c_str()).string());
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


ProjectionSet * readProjectionsAndAngles(FileName starFileName, int multiply) {

	ProjectionSet *output = new ProjectionSet();
	MultidimArray<RDOUBLE> projections;
	float3 * angles = NULL;


	int numProj = readProjections(starFileName, projections, &angles, false);

	output->angles = (float3*)malloc(multiply * numProj * sizeof(float3));
	output->images.resizeNoCopy(multiply*numProj, projections.ydim, projections.xdim);
	for (size_t i = 0; i < multiply; i++)
	{
		memcpy(output->angles + numProj * i, angles, numProj * sizeof(float3));
		memcpy(output->images.data + projections.nzyxdim * i, projections.data, projections.nzyxdim * sizeof(RDOUBLE));
	}
	numProj *= multiply;
	free(angles);
	projections.coreDeallocate();
	output->numProj = numProj;
	return output;
}

ProjectionSet * getDummyCTF(ProjectionSet *referenceSet) {
	ProjectionSet *output = new ProjectionSet();

	output->images.resize(referenceSet->images.zdim, referenceSet->images.ydim, referenceSet->images.xdim / 2 + 1);
	output->numProj = referenceSet->numProj;
	output->angles = (float3 *)malloc(sizeof(*output->angles)*output->numProj);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(output->images) {
		DIRECT_MULTIDIM_ELEM(output->images, n) = 1.0;
	}
	return output;
}

ProjectionSet * combineProjectionSets(ProjectionSet *firstSet, ProjectionSet *secondSet, ProjectionSet *thirdSet) {
	ProjectionSet *output = new ProjectionSet();
	output->numProj = firstSet->numProj + secondSet->numProj, +thirdSet->numProj;
	output->images.resize(output->numProj, firstSet->images.ydim, firstSet->images.xdim);
	output->angles = (float3 *)malloc(sizeof(*output->angles)*output->numProj);

	int copied = 0;
	if (firstSet->numProj > 0) {
		memcpy(output->images.data, firstSet->images.data, firstSet->images.zyxdim * sizeof(*firstSet->images.data));
		memcpy(output->angles, firstSet->angles, firstSet->numProj * sizeof(*output->angles));
		copied += firstSet->numProj;
	}
	if (secondSet->numProj > 0) {
		memcpy(output->images.data + copied * output->images.yxdim, secondSet->images.data, secondSet->images.zyxdim * sizeof(*secondSet->images.data));
		memcpy(output->angles + copied, secondSet->angles, secondSet->numProj * sizeof(*output->angles));
		copied += secondSet->numProj;
	}
	if (thirdSet->numProj > 0) {
		memcpy(output->images.data + copied * output->images.yxdim, thirdSet->images.data, thirdSet->images.zyxdim * sizeof(*thirdSet->images.data));
		memcpy(output->angles + copied, thirdSet->angles, thirdSet->numProj * sizeof(*output->angles));
		copied += thirdSet->numProj;
	}
	return output;
}

ProjectionSet * getMaskSet(FileName maskFileName, ProjectionSet * projectionSet, Pseudoatoms *atoms) {

	ProjectionSet * maskSet = new ProjectionSet();

	for (size_t i = 0; i < atoms->NAtoms; i++)
	{
		atoms->AtomWeights[i] = 1;
	}
	if (!fs::exists(maskFileName.c_str())) {
		int3 maskDim = make_int3(projectionSet->images.xdim, projectionSet->images.ydim, 1);
		PseudoProjector MaskProj(maskDim, atoms, 1.0);

		MaskProj.createMask(projectionSet->angles, projectionSet->positionMatching, NULL, maskSet->images, projectionSet->numProj, 0.0, 0.0);

		maskSet->images.binarize(0.1);
		{
			MRCImage im(maskSet->images);
			im.writeAs<float>(maskFileName);
		}
	}
	else {
		MRCImage<float> im = MRCImage<float>::readAs(maskFileName);
		maskSet->images = im();
	}

	return maskSet;
}
