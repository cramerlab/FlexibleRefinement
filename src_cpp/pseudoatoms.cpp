#include "pseudoatoms.h"
using namespace relion;

void Pseudoatoms::RasterizeToVolume(MultidimArray<DOUBLE>& vol, int3 Dims, DOUBLE super)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		vol.resizeNoCopy((long)(Dims.z * super + 0.5), (long)(Dims.y * super + 0.5), (long)(Dims.x * super + 0.5));


		MultidimArray<DOUBLE> Weights((long)(Dims.z * super + 0.5), (long)(Dims.y * super + 0.5), (long)(Dims.x * super + 0.5));
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			Matrix1D<DOUBLE> superPos = AtomPositions[p] * super;

			int X0 = (int)XX(superPos);
			DOUBLE ix = XX(superPos) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)YY(superPos);
			DOUBLE iy = YY(superPos) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)ZZ(superPos);
			DOUBLE iz = ZZ(superPos) - Z0;
			int Z1 = Z0 + 1;

			DOUBLE v0 = 1.0f - iz;
			DOUBLE v1 = iz;

			DOUBLE v00 = (1.0f - iy) * v0;
			DOUBLE v10 = iy * v0;
			DOUBLE v01 = (1.0f - iy) * v1;
			DOUBLE v11 = iy * v1;

			DOUBLE v000 = (1.0f - ix) * v00;
			DOUBLE v100 = ix * v00;
			DOUBLE v010 = (1.0f - ix) * v10;
			DOUBLE v110 = ix * v10;
			DOUBLE v001 = (1.0f - ix) * v01;
			DOUBLE v101 = ix * v01;
			DOUBLE v011 = (1.0f - ix) * v11;
			DOUBLE v111 = ix * v11;

			A3D_ELEM(vol, Z0, Y0, X0) += AtomWeights[p] * v000;
			A3D_ELEM(vol, Z0, Y0, X1) += AtomWeights[p] * v001;
			A3D_ELEM(vol, Z0, Y1, X0) += AtomWeights[p] * v010;
			A3D_ELEM(vol, Z0, Y1, X1) += AtomWeights[p] * v011;

			A3D_ELEM(vol, Z1, Y0, X0) += AtomWeights[p] * v100;
			A3D_ELEM(vol, Z1, Y0, X1) += AtomWeights[p] * v101;
			A3D_ELEM(vol, Z1, Y1, X0) += AtomWeights[p] * v110;
			A3D_ELEM(vol, Z1, Y1, X1) += AtomWeights[p] * v111;

			A3D_ELEM(Weights, Z0, Y0, X0) += v000;
			A3D_ELEM(Weights, Z0, Y0, X1) += v001;
			A3D_ELEM(Weights, Z0, Y1, X0) += v010;
			A3D_ELEM(Weights, Z0, Y1, X1) += v011;

			A3D_ELEM(Weights, Z1, Y0, X0) += v100;
			A3D_ELEM(Weights, Z1, Y0, X1) += v101;
			A3D_ELEM(Weights, Z1, Y1, X0) += v110;
			A3D_ELEM(Weights, Z1, Y1, X1) += v111;
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(vol) {
			A3D_ELEM(vol, k, i, j) /= std::max((DOUBLE)(1e-10), A3D_ELEM(Weights, k, i, j));
		}

		Weights.coreDeallocate();

		resizeMap(vol, Dims.x);
		return;
	}
	else if (Mode == ATOM_GAUSSIAN) {
		vol.resizeNoCopy(Dims.z*super, Dims.y*super, Dims.x*super);
		auto itPos = AtomPositions.begin();
		auto itWeight = AtomWeights.begin();
#pragma omp parallel for
		for (int n = 0; n < AtomPositions.size(); n++) {
			drawOneGaussian(GaussianTable, 4 * Sigma*super, ZZ(AtomPositions[n])*super, YY(AtomPositions[n])*super, XX(AtomPositions[n])*super, vol, AtomWeights[n], GaussFactor / super);
		}

		if (super > 1.0) {
			resizeMap(vol, Dims.x);
		}
	}
}

void Pseudoatoms::IntensityFromVolume(MultidimArray<DOUBLE>& vol, DOUBLE super)
{
	AtomWeights.clear();
	AtomWeights.reserve(AtomPositions.size());
	MultidimArray<DOUBLE> volSuper = vol;
	resizeMap(volSuper, (int)(volSuper.xdim*super + 0.5));


	for (int p = 0; p < AtomPositions.size(); p++)
	{
		Matrix1D<DOUBLE> superPos = AtomPositions[p] * super;

		int X0 = (int)XX(superPos);
		DOUBLE ix = XX(superPos) - X0;
		int X1 = X0 + 1;

		int Y0 = (int)YY(superPos);
		DOUBLE iy = YY(superPos) - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)ZZ(superPos);
		DOUBLE iz = ZZ(superPos) - Z0;
		int Z1 = Z0 + 1;

		DOUBLE v000 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v001 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v010 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v011 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v100 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v101 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v110 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v111 = A3D_ELEM(volSuper, Z0, Y0, X0);

		DOUBLE v00 = Lerp(v000, v001, ix);
		DOUBLE v01 = Lerp(v010, v011, ix);
		DOUBLE v10 = Lerp(v100, v101, ix);
		DOUBLE v11 = Lerp(v110, v111, ix);

		DOUBLE v0 = Lerp(v00, v01, iy);
		DOUBLE v1 = Lerp(v10, v11, iy);

		DOUBLE v = Lerp(v0, v1, iz);

		AtomWeights.push_back(v);
	}

	volSuper.coreDeallocate();
}
