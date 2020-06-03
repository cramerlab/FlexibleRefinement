#include "funcs.h"


DOUBLE Lerp(DOUBLE a, DOUBLE b, DOUBLE x)
{
	return a + (b - a) * x;
}


/*
DOUBLE gaussian1D(DOUBLE x, DOUBLE sigma, DOUBLE mu = 0)
{
	x -= mu;
	return exp(-0.5*((x / sigma)*(x / sigma)));
}*/


void Uproject_to_plane(const Matrix1D<DOUBLE> &r,
	const Matrix2D<DOUBLE> &euler, Matrix1D<DOUBLE> &result)
{
	SPEED_UP_temps012;
	if (VEC_XSIZE(result) != 3)
		result.resize(3);
	M3x3_BY_V3x1(result, euler, r);
}

void Uproject_to_plane(const Matrix1D<DOUBLE> &point,
	const Matrix1D<DOUBLE> &direction, DOUBLE distance,
	Matrix1D<DOUBLE> &result)
{

	if (result.size() != 3)
		result.resize(3);
	DOUBLE xx = distance - (XX(point) * XX(direction) + YY(point) * YY(direction) +
		ZZ(point) * ZZ(direction));
	XX(result) = XX(point) + xx * XX(direction);
	YY(result) = YY(point) + xx * YY(direction);
	ZZ(result) = ZZ(point) + xx * ZZ(direction);
}

void drawOneGaussian(MultidimArray<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, DOUBLE gaussFactor)
{

	int k0 = CEIL(XMIPP_MAX(STARTINGZ(V), k - boundary));
	int i0 = CEIL(XMIPP_MAX(STARTINGY(V), i - boundary));
	int j0 = CEIL(XMIPP_MAX(STARTINGX(V), j - boundary));
	int kF = FLOOR(XMIPP_MIN(FINISHINGZ(V), k + boundary));
	int iF = FLOOR(XMIPP_MIN(FINISHINGY(V), i + boundary));
	int jF = FLOOR(XMIPP_MIN(FINISHINGX(V), j + boundary));
	for (int kk = k0; kk <= kF; kk++)
	{
		DOUBLE aux = kk - k;
		DOUBLE diffkk2 = aux * aux;
		for (int ii = i0; ii <= iF; ii++)
		{
			aux = ii - i;
			DOUBLE diffiikk2 = aux * aux + diffkk2;
			for (int jj = j0; jj <= jF; jj++)
			{
				aux = jj - j;
				DOUBLE r = sqrt(diffiikk2 + aux * aux);
				aux = r * gaussFactor;
				long iaux = lround(aux);
				A3D_ELEM(V, kk, ii, jj) += intensity * DIRECT_A1D_ELEM(gaussianTable, iaux);
			}
		}
	}

}


void drawOneGaussian(Matrix1D<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, DOUBLE gaussFactor)
{

	int k0 = CEIL(XMIPP_MAX(STARTINGZ(V), k - boundary));
	int i0 = CEIL(XMIPP_MAX(STARTINGY(V), i - boundary));
	int j0 = CEIL(XMIPP_MAX(STARTINGX(V), j - boundary));
	int kF = FLOOR(XMIPP_MIN(FINISHINGZ(V), k + boundary));
	int iF = FLOOR(XMIPP_MIN(FINISHINGY(V), i + boundary));
	int jF = FLOOR(XMIPP_MIN(FINISHINGX(V), j + boundary));
	for (int kk = k0; kk <= kF; kk++)
	{
		DOUBLE aux = kk - k;
		DOUBLE diffkk2 = aux * aux;
		for (int ii = i0; ii <= iF; ii++)
		{
			aux = ii - i;
			DOUBLE diffiikk2 = aux * aux + diffkk2;
			for (int jj = j0; jj <= jF; jj++)
			{
				aux = jj - j;
				DOUBLE r = sqrt(diffiikk2 + aux * aux);
				aux = r * gaussFactor;
				long iaux = lround(aux);
				A3D_ELEM(V, kk, ii, jj) += intensity * VEC_ELEM(gaussianTable, iaux);
			}
		}
	}

}

void writeFSC(MultidimArray<DOUBLE> &V1, MultidimArray<DOUBLE> &V2, FileName outpath) {
	MultidimArray<DOUBLE> fsc;
	getFSC(V1, V2, fsc);
	{
		std::ofstream ofs(outpath);

		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fsc) {
			ofs << i << "\t" << DIRECT_A1D_ELEM(fsc, i) << std::endl;
		}
	}
}