#pragma once
#ifndef PROJECTION_READER_H
#define PROJECTION_READER_H

#include "liblionImports.h"
#include "Types.h"
#include "Pseudoatoms.h"

struct ProjectionSet {
	float3 * angles;
	relion::MultidimArray<RDOUBLE> images;
	int numProj;
	int *positionMatching;
	ProjectionSet() :angles(NULL), images(), numProj(0), positionMatching(NULL) {}
};


idxtype readProjections(relion::FileName starFileName, relion::MultidimArray<RDOUBLE> &projections, float3 **angles, bool shuffle);

ProjectionSet * readProjectionsAndAngles(relion::FileName starFileName, int multiply = 1);

ProjectionSet * getDummyCTF(ProjectionSet *referenceSet);

ProjectionSet * combineProjectionSets(ProjectionSet *firstSet, ProjectionSet *secondSet, ProjectionSet *thirdSet);

ProjectionSet * getMaskSet(relion::FileName maskFileName, ProjectionSet * projectionSet, Pseudoatoms *atoms);

#endif