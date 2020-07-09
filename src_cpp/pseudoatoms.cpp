#include "pseudoatoms.h"
#include "readMRC.h"
using namespace relion;

void Pseudoatoms::RasterizeToVolume(MultidimArray<DOUBLE>& vol, int3 Dims, DOUBLE super, bool resize)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		vol.resizeNoCopy((long)(Dims.z * super + 0.5), (long)(Dims.y * super + 0.5), (long)(Dims.x * super + 0.5));
		vol.initZeros();
		//resizeMap(vol, (long)(Dims.z * super + 0.5));
		//ResizeMapGPU(vol, make_int3((int)(volSuper.xdim*super + 0.5), (int)(volSuper.ydim*super + 0.5), (int)(volSuper.zdim*super + 0.5)));


		MultidimArray<DOUBLE> Weights((long)(Dims.z * super + 0.5), (long)(Dims.y * super + 0.5), (long)(Dims.x * super + 0.5));
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 superPos = AtomPositions[p] * super;

			int X0 = (int)(superPos.x);
			DOUBLE ix = (superPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(superPos.y);
			DOUBLE iy = (superPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(superPos.z);
			DOUBLE iz = (superPos.z) - Z0;
			int Z1 = Z0 + 1;

			DOUBLE v0 = 1.0f - ix;
			DOUBLE v1 = ix;

			DOUBLE v00 = (1.0f - iy) * v0;
			DOUBLE v10 = iy * v0;
			DOUBLE v01 = (1.0f - iy) * v1;
			DOUBLE v11 = iy * v1;

			DOUBLE v000 = (1.0f - iz) * v00;
			DOUBLE v100 = iz * v00;
			DOUBLE v010 = (1.0f - iz) * v10;
			DOUBLE v110 = iz * v10;
			DOUBLE v001 = (1.0f - iz) * v01;
			DOUBLE v101 = iz * v01;
			DOUBLE v011 = (1.0f - iz) * v11;
			DOUBLE v111 = iz * v11;

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
		if(resize)
			ResizeMapGPU(vol, Dims);
		return;
	}
	else if (Mode == ATOM_GAUSSIAN) {
		resizeMap(vol, (long)(Dims.z * super + 0.5));
		auto itPos = AtomPositions.begin();
		auto itWeight = AtomWeights.begin();

		for (int n = 0; n < AtomPositions.size(); n++) {
			drawOneGaussian(GaussianTable, 4 * Sigma*super, (AtomPositions[n].z)*super, (AtomPositions[n].y)*super, (AtomPositions[n].x)*super, vol, AtomWeights[n], GaussFactor / super);
		}

		if (super > 1.0) {
			resizeMap(vol, Dims.x);
		}
	}
}


void Pseudoatoms::MoveAtoms(MultidimArray<DOUBLE>& superRefVol, int3 Dims, DOUBLE super, bool resize)
{
	if (Mode == ATOM_INTERPOLATE)
	{
		MultidimArray<float> SuperRastered;
		RasterizeToVolume(SuperRastered,Dims,super, false);
		MultidimArray<float> SuperDiffVol = SuperRastered;

		Substract_GPU(SuperDiffVol, superRefVol);
		{
			MRCImage<float> DiffIm(SuperDiffVol);
			DiffIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_SuperDiffIm.mrc", true);
		}

		{
			MRCImage<float> SuperRefIm(superRefVol);
			SuperRefIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_SuperRefVol.mrc", true);
			MRCImage<float> SuperRasteredIm(SuperRastered);
			SuperRasteredIm.writeAs<float>("D:\\EMD\\9233\\AtomFitting\\moving_rasteredVol.mrc", true);

		}
#pragma omp parallel for
		for (int p = 0; p < AtomPositions.size(); p++)
		{
			float3 superPos = AtomPositions[p] * super;

			int X0 = (int)(superPos.x);
			DOUBLE ix = (superPos.x) - X0;
			int X1 = X0 + 1;

			int Y0 = (int)(superPos.y);
			DOUBLE iy = (superPos.y) - Y0;
			int Y1 = Y0 + 1;

			int Z0 = (int)(superPos.z);
			DOUBLE iz = (superPos.z) - Z0;
			int Z1 = Z0 + 1;

			DOUBLE v0 = 1.0f - ix;
			DOUBLE vd0 = -1;
			DOUBLE v1 = ix;
			DOUBLE vd1 = 1;

			DOUBLE v00 = (1.0f - iy) * v0;
			DOUBLE v0d0 = (1.0f - iy) * vd0;
			DOUBLE vd00 = - 1 * v0;

			DOUBLE v10 = iy * v0;
			DOUBLE v1d0 = iy * vd0;
			DOUBLE vd10 = 1 * v0;


			DOUBLE v01 = (1.0f - iy) * v1;
			DOUBLE v0d1 = (1.0f - iy) * vd1;
			DOUBLE vd01 = ( - 1) * v1;

			DOUBLE v11 = iy * v1;
			DOUBLE v1d1 = iy * vd1;
			DOUBLE vd11 = 1 * v1;

			DOUBLE v000 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y0, X0);
			DOUBLE v00d0 = (1.0f - iz) * v0d0;
			DOUBLE v0d00 = (1.0f - iz) * vd00;
			DOUBLE vd000 = (- 1) * v00;

			DOUBLE v100 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y0, X0);
			DOUBLE v10d0 = iz * v0d0;
			DOUBLE v1d00 = iz * vd00;
			DOUBLE vd100 = 1 * v00;

			DOUBLE v010 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y1, X0);
			DOUBLE v01d0 = (1.0f - iz) * v1d0;
			DOUBLE v0d10 = (1.0f - iz) * vd10;
			DOUBLE vd010 = (-1) * v10;

			DOUBLE v110 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y1, X0);
			DOUBLE v11d0 = iz * v1d0;
			DOUBLE v1d10 = iz * vd10;
			DOUBLE vd110 = 1 * v10;

			DOUBLE v001 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y0, X1);
			DOUBLE v00d1 = (1.0f - iz) * v0d1;
			DOUBLE v0d01 = (1.0f - iz) * vd01;
			DOUBLE vd001 = (-1.0f) * v01;

			DOUBLE v101 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y0, X1);
			DOUBLE v10d1 = iz * v0d1;
			DOUBLE v1d01 = iz * vd01;
			DOUBLE vd101 = 1 * v01;

			DOUBLE v011 = DIRECT_A3D_ELEM(SuperDiffVol, Z0, Y1, X1);
			DOUBLE v01d1 = (1.0f - iz) * v1d1;
			DOUBLE v0d11 = (1.0f - iz) * vd11;
			DOUBLE vd011 = (-1.0f) * v11;

			DOUBLE v111 = DIRECT_A3D_ELEM(SuperDiffVol, Z1, Y1, X1);
			DOUBLE v11d1 = iz * v1d1;
			DOUBLE v1d11 = iz * vd11;
			DOUBLE vd111 = 1 * v11;

			float3 movement = {
				 (v00d1 * v001 + v01d1 * v011 + v10d1 * v101 + v11d1 * v111) - (v00d0 * v000 + v01d0 * v010 + v10d0 * v100 + v11d0 * v110),
				 (v0d10 * v010 + v1d10 * v110 + v0d11 * v011 + v1d11 * v111) - (v1d00 * v100 + v0d00 * v000 + v0d01 * v001 + v1d01 * v101),
				 (vd100 * v100 + vd110 * v110 + vd101 * v101 + vd111 * v111) - (vd000 * v000 + vd010 * v010 + vd001 * v001 + vd011 * v011) };
			if (true)
				;

		}



		return;
	}

}

void Pseudoatoms::IntensityFromVolume(MultidimArray<DOUBLE>& vol, DOUBLE super)
{
	AtomWeights.clear();
	AtomWeights.resize(AtomPositions.size());
	MultidimArray<DOUBLE> volSuper = vol;
	ResizeMapGPU(volSuper, make_int3((int)(volSuper.xdim*super + 0.5), (int)(volSuper.ydim*super + 0.5), (int)(volSuper.zdim*super + 0.5)));

#pragma omp parallel for
	for (int p = 0; p < AtomPositions.size(); p++)
	{
		float3 superPos = AtomPositions[p] * super;

		int X0 = (int)(superPos.x);
		DOUBLE ix = (superPos.x) - X0;
		int X1 = X0 + 1;

		int Y0 = (int)(superPos.y);
		DOUBLE iy = (superPos.y) - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)(superPos.z);
		DOUBLE iz = (superPos.z) - Z0;
		int Z1 = Z0 + 1;

		DOUBLE v000 = A3D_ELEM(volSuper, Z0, Y0, X0);
		DOUBLE v001 = A3D_ELEM(volSuper, Z0, Y0, X1);
		DOUBLE v010 = A3D_ELEM(volSuper, Z0, Y1, X0);
		DOUBLE v011 = A3D_ELEM(volSuper, Z0, Y1, X1);
		DOUBLE v100 = A3D_ELEM(volSuper, Z1, Y0, X0);
		DOUBLE v101 = A3D_ELEM(volSuper, Z1, Y0, X1);
		DOUBLE v110 = A3D_ELEM(volSuper, Z1, Y1, X0);
		DOUBLE v111 = A3D_ELEM(volSuper, Z1, Y1, X1);

		DOUBLE v00 = Lerp(v000, v001, ix);
		DOUBLE v01 = Lerp(v010, v011, ix);
		DOUBLE v10 = Lerp(v100, v101, ix);
		DOUBLE v11 = Lerp(v110, v111, ix);

		DOUBLE v0 = Lerp(v00, v01, iy);
		DOUBLE v1 = Lerp(v10, v11, iy);

		DOUBLE v = Lerp(v0, v1, iz);

		AtomWeights[p] = v;
	}

	volSuper.coreDeallocate();
}
