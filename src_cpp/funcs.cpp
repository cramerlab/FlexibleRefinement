#include "funcs.h"


void drawOneGaussian(MultidimArray<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, idxtype gaussFactor)
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


void drawOneGaussian(Matrix1D<DOUBLE> &gaussianTable, DOUBLE boundary, DOUBLE k, DOUBLE i, DOUBLE j, MultidimArray<DOUBLE> &V, DOUBLE intensity, idxtype gaussFactor)
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