
#include "PseudoProjector.h"
#include "liblion.h"
using namespace relion;



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



void PseudoProjector::project_Pseudo(DOUBLE * out, DOUBLE * out_nrm,
	float3 Euler, DOUBLE shiftX, DOUBLE shiftY,
	int direction)
{
	MultidimArray<DOUBLE> proj = MultidimArray<DOUBLE>(this->Dims.z, this->Dims.y, this->Dims.x);
	proj.data = out;
	proj.xinit = shiftX;
	proj.yinit = shiftY;
	proj.destroyData = false;
	
	MultidimArray<DOUBLE> proj_nrm = MultidimArray<DOUBLE>(this->Dims.z, this->Dims.y, this->Dims.x);
	if (out_nrm != NULL)
	{
		proj_nrm.data = out_nrm;
		proj_nrm.destroyData = NULL;
	}
	proj_nrm.xinit = shiftX;
	proj_nrm.yinit = shiftY;
	
	Matrix2D<DOUBLE> EulerMat = Matrix2D<DOUBLE>();
	Euler_angles2matrix(Euler.x, Euler.y, Euler.z, EulerMat);
	
	
	this->project_Pseudo(proj, proj_nrm,	EulerMat, shiftX, shiftY,direction);
}

void PseudoProjector::project_PseudoCTF(DOUBLE * out, DOUBLE * out_nrm, DOUBLE *gaussTable, DOUBLE * gaussTable2, DOUBLE border,
	float3 Euler, DOUBLE shiftX, DOUBLE shiftY,
	int direction)
{
	DOUBLE * tableTmp = gaussianProjectionTable.vdata;
	DOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;
	double oldBorder = this->tableLength;
	this->tableLength = border;
	gaussianProjectionTable.vdata = gaussTable;
	gaussianProjectionTable2.vdata = gaussTable2;

	MultidimArray<DOUBLE> proj = MultidimArray<DOUBLE>(this->Dims.z, this->Dims.y, this->Dims.x);
	proj.data = out;
	proj.xinit = shiftX;
	proj.yinit = shiftY;
	proj.destroyData = false;
	
	MultidimArray<DOUBLE> proj_nrm = MultidimArray<DOUBLE>(this->Dims.z, this->Dims.y, this->Dims.x);
	if (out_nrm != NULL)
	{
		proj_nrm.data = out_nrm;
		proj_nrm.destroyData = NULL;
	}
	proj_nrm.xinit = shiftX;
	proj_nrm.yinit = shiftY;
	
	Matrix2D<DOUBLE> EulerMat = Matrix2D<DOUBLE>();
	Euler_angles2matrix(Euler.x, Euler.y, Euler.z, EulerMat);
	
	
	this->project_Pseudo(proj, proj_nrm,	EulerMat, shiftX, shiftY,direction);
	gaussianProjectionTable.vdata = tableTmp;
	gaussianProjectionTable2.vdata = tableTmp2;
	this->tableLength = oldBorder;
}

 /** Projection of a pseudoatom volume */
void PseudoProjector::project_Pseudo(
	MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj,
	Matrix2D<DOUBLE> &Euler, DOUBLE shiftX, DOUBLE shiftY,
	int direction)
{
	// Project all pseudo atoms ............................................
	int numAtoms = atomPosition.size();
	Matrix1D<DOUBLE> actprj(3);
	double sigma4 = this->tableLength;
	Matrix1D<DOUBLE> actualAtomPosition;

	for (int n = 0; n < numAtoms; n++)
	{
		actualAtomPosition = atomPosition[n];
		XX(actualAtomPosition) -= Dims.x / 2;
		YY(actualAtomPosition) -= Dims.y / 2;
		ZZ(actualAtomPosition) -= Dims.z / 2;
		DOUBLE weight = atomWeight[n];
		/*
		for (int mode = 0; mode < lambdaSize; mode++)
		{
			const Matrix2D<double> &NMAmode = NMA[mode];
			double lambdam = lambda[mode];
			FOR_ALL_ELEMENTS_IN_MATRIX1D(actualAtomPosition)
				VEC_ELEM(actualAtomPosition, i) +=
				lambdam * MAT_ELEM(NMAmode, n, i);
		}
		*/
		Uproject_to_plane(actualAtomPosition, Euler, actprj);
		XX(actprj) += Dims.x / 2;
		YY(actprj) += Dims.y / 2;
		ZZ(actprj) += Dims.z / 2;
		XX(actprj) += shiftX;
		YY(actprj) += shiftY;
#ifdef DEBUG

		bool condition = true;
		if (condition)
		{
			std::cout << "Projecting point " << atomPositions[n].transpose() << std::endl;
			std::cout << "Vol there = " << atomWeight[n] << std::endl;
			std::cout << " Center of the basis proj (2D) " << XX(actprj) << "," << YY(actprj) << std::endl;
		}
#endif

		// Search for integer corners for this basis
		int XX_corner1 = CEIL(XMIPP_MAX(STARTINGX(proj), XX(actprj) - sigma4));
		int YY_corner1 = CEIL(XMIPP_MAX(STARTINGY(proj), YY(actprj) - sigma4));
		int XX_corner2 = FLOOR(XMIPP_MIN(FINISHINGX(proj), XX(actprj) + sigma4));
		int YY_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(proj), YY(actprj) + sigma4));

#ifdef DEBUG

		if (condition)
		{
			std::cout << "Clipped and rounded Corner 1 " << XX_corner1
				<< " " << YY_corner1 << " " << std::endl;
			std::cout << "Clipped and rounded Corner 2 " << XX_corner2
				<< " " << YY_corner2 << " " << std::endl;
		}
#endif

		// Check if the basis falls outside the projection plane
		if (XX_corner1 <= XX_corner2 && YY_corner1 <= YY_corner2)
		{
			DOUBLE vol_corr = 0;

			// Effectively project this basis
			for (int y = YY_corner1; y <= YY_corner2; y++)
			{
				DOUBLE y_diff2 = y - YY(actprj);
				y_diff2 = y_diff2 * y_diff2;
				for (int x = XX_corner1; x <= XX_corner2; x++)
				{
					DOUBLE x_diff2 = x - XX(actprj);
					x_diff2 = x_diff2 * x_diff2;
					DOUBLE r = sqrt(x_diff2 + y_diff2);
					DOUBLE didx = r * GAUSS_FACTOR;
					int idx = ROUND(didx);
					DOUBLE a = VEC_ELEM(gaussianProjectionTable, idx);
					DOUBLE a2 = VEC_ELEM(gaussianProjectionTable2, idx);
#ifdef DEBUG

					if (condition)
						std::cout << "=" << a << " , " << a2;
#endif
					if (a < 0 || a>2) {
						bool is = true;
					}
					if (direction == PSEUDO_FORWARD)
					{
						A2D_ELEM(proj, y, x) += weight * a;
						A2D_ELEM(norm_proj, y, x) += a2;
#ifdef DEBUG

						if (condition)
						{
							std::cout << " proj= " << A2D_ELEM(proj, y, x)
								<< " norm_proj=" << A2D_ELEM(norm_proj, y, x) << std::endl;
							std::cout.flush();
						}
#endif

					}
					else
					{
						vol_corr += A2D_ELEM(norm_proj, y, x) * a;
#ifdef DEBUG

						if (condition)
						{
							std::cout << " corr_img= " << A2D_ELEM(norm_proj, y, x)
								<< " correction=" << vol_corr << std::endl;
							std::cout.flush();
						}
#endif

					}
				}
			}

			if (direction == PSEUDO_BACKWARD)
			{
				atomWeight[n] += vol_corr;
#ifdef DEBUG

				if (condition)
				{
					std::cout << "\nFinal value at ( " << n << ") = "
						<< atomWeight[n] << std::endl;
				}
#endif

			}
		} // If not collapsed
	}
}

DOUBLE PseudoProjector::ART_single_image(const MultidimArray<DOUBLE> &Iexp, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY)
{
	MultidimArray<DOUBLE> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Matrix2D<DOUBLE> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_FORWARD);
	Idiff.resize(Iexp);

	DOUBLE mean_error = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
	{
		// Compute difference image and error
		DIRECT_A2D_ELEM(Idiff, i, j) = DIRECT_A2D_ELEM(Iexp, i, j) - DIRECT_A2D_ELEM(Itheo, i, j);
		mean_error += DIRECT_A2D_ELEM(Idiff, i, j) * DIRECT_A2D_ELEM(Idiff, i, j);

		// Compute the correction image
		DIRECT_A2D_ELEM(Icorr, i, j) = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
		DIRECT_A2D_ELEM(Icorr, i, j) =
			this->lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
	}
	mean_error /= YXSIZE(Iexp);

	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_BACKWARD);
	return mean_error;
}

DOUBLE PseudoProjector::ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE tableLength, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<DOUBLE> > Images;
	DOUBLE itError = 0.0;
	
	for (size_t i = 0; i < numImages; i++)
	{
		DOUBLE * tableTmp = gaussianProjectionTable.vdata;
		DOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;

		gaussianProjectionTable.vdata = gaussTables + i * (GAUSS_FACTOR * Dims.x / 2 );
		gaussianProjectionTable2.vdata = gaussTables2 + i * (GAUSS_FACTOR * Dims.x / 2 );
		double oldBorder = this->tableLength;
		this->tableLength = tableLength;
		MultidimArray<DOUBLE> tmp = MultidimArray<DOUBLE>(Dims.z, Dims.y, Dims.x);
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		tmp.destroyData = false;
		itError += ART_single_image(tmp, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);

		gaussianProjectionTable.vdata = tableTmp;
		gaussianProjectionTable2.vdata = tableTmp2;
		this->tableLength = oldBorder;
	}
	return itError;
}

DOUBLE PseudoProjector::ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<DOUBLE> > Images;
	DOUBLE itError = 0.0;
	for (size_t i = 0; i < numImages; i++)
	{
		MultidimArray<DOUBLE> tmp = MultidimArray<DOUBLE>(Dims.z, Dims.y, Dims.x);
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		tmp.destroyData = false;
		itError += ART_single_image(tmp, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);
	}
	return itError;
}

DOUBLE PseudoProjector::ART_multi_Image_step(std::vector< MultidimArray<DOUBLE> > Iexp, std::vector<float3> angles, DOUBLE shiftX, DOUBLE shiftY) {
	DOUBLE itError = 0.0;
	for(unsigned int i=0; i < Iexp.size(); i++)
	{
		itError += ART_single_image(Iexp[i], angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);
	}
	return itError / Iexp.size(); // Mean Error
}

/*
void ProgARTPseudo::show() const
{
	if (verbose > 0)
	{
		std::cout << " =======================================================" << std::endl;
		std::cout << " ART reconstruction method using pseudo atomic structure" << std::endl;
		std::cout << " =======================================================" << std::endl;
		std::cout << "Input images:    " << fnDoc << std::endl;
		std::cout << "Pseudoatoms:     " << fnPseudo << std::endl;
		std::cout << "Sigma:           " << sigma << std::endl;
		std::cout << "Sampling rate:   " << sampling << std::endl;
		if (!fnNMA.empty())
			std::cout << "NMA:             " << fnNMA << std::endl;
		std::cout << "Output rootname: " << fnRoot << std::endl;
		std::cout << "Lambda ART:      " << lambdaART << std::endl;
		std::cout << "N. Iterations:   " << Nit << std::endl;
		std::cout << "\n -----------------------------------------------------" << std::endl;
	}
}

void ProgARTPseudo::defineParams()
{
	addUsageLine("Generate 3D reconstructions from projections using ART on pseudoatoms.");
	addUsageLine("+This program reconstructs based on irregular grids given by pseudo atomic structures. ");
	addUsageLine("+In case of regular grids, please refer to other programs as reconstruct_art ");
	addUsageLine("+or reconstruct_fourier. Optionally, a deformation file generated by Nodal Mode ");
	addUsageLine("+Alignment (NMA) can be passed with --nma parameter.");

	addSeeAlsoLine("volume_to_pseudoatoms, nma_alignment");

	addParamsLine("   -i <md_file>          : Metadata file with input projections");
	addParamsLine("   --oroot <rootname>    : Output rootname");
	addParamsLine("   alias -o;");
	addParamsLine("   --pseudo <pseudofile> : Pseudo atomic structure (PDB format)");
	addParamsLine("     alias -p;");
	addParamsLine("  [--sigma <s=-1>]       : Pseudoatom sigma. By default, from pseudo file");
	addParamsLine("  [--sampling_rate <Ts=1>]  : Pixel size (Angstrom)");
	addParamsLine("  alias -s;");
	addParamsLine("  [-l <lambda=0.1>]      : Relaxation factor");
	addParamsLine("  [-n <N=1>]             : Number of iterations");
	addParamsLine("  [--nma <selfile=\"\">] : Selfile with NMA");

	addExampleLine("Reconstruct with NMA file and relaxation factor of 0.2:", false);
	addExampleLine("xmipp_reconstruct_art_pseudo -i projections.xmd -o art_rec --nma nmafile.xmd -l 0.2");
}

void ProgARTPseudo::readParams()
{
	fnDoc = getParam("-i");
	fnPseudo = getParam("--pseudo");
	fnRoot = getParam("--oroot");
	lambdaART = getDoubleParam("-l");
	Nit = getIntParam("-n");
	sigma = getDoubleParam("--sigma");
	fnNMA = getParam("--nma");
	sampling = getDoubleParam("--sampling_rate");
}

void ProgARTPseudo::produceSideInfo()
{
	DF.read(fnDoc);
	std::ifstream fhPseudo;
	fhPseudo.open(fnPseudo.c_str());
	if (!fhPseudo)
		REPORT_ERROR(ERR_IO_NOTEXIST, fnPseudo);
	while (!fhPseudo.eof())
	{
		std::string line;
		getline(fhPseudo, line);
		if (line.length() == 0)
			continue;
		if (line.substr(7, 13) == "fixedGaussian" && sigma < 0)
		{
			std::vector < std::string> results;
			splitString(line, " ", results);
			sigma = textToFloat(results[2]);
			sigma /= sampling;
		}
		else if (line.substr(0, 4) == "ATOM")
		{
			Matrix1D<double> v(3);
			v(0) = textToFloat(line.substr(30, 8));
			v(1) = textToFloat(line.substr(38, 8));
			v(2) = textToFloat(line.substr(46, 8));
			v /= sampling;
			atomPosition.push_back(v);
			atomWeight.push_back(0);
		}
	}
	fhPseudo.close();

	double sigma4 = 4 * sigma;
	gaussianProjectionTable.resize(CEIL(sigma4*sqrt(2) * 1000));
	FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
		gaussianProjectionTable(i) = gaussian1D(i / 1000.0, sigma);
	gaussianProjectionTable *= gaussian1D(0, sigma);
	gaussianProjectionTable2 = gaussianProjectionTable;
	gaussianProjectionTable2 *= gaussianProjectionTable;

	// NMA
	if (!fnNMA.empty())
	{
		MetaData DFNMA(fnNMA);
		DFNMA.removeDisabled();
		FOR_ALL_OBJECTS_IN_METADATA(DFNMA)
		{
			Matrix2D<double> mode;
			mode.initZeros(atomPosition.size(), 3);
			FileName fnMode;
			DFNMA.getValue(MDL_NMA_MODEFILE, fnMode, __iter.objId);
			mode.read(fnMode);
			NMA.push_back(mode);
		}
	}
}

void ProgARTPseudo::run()
{
	show();
	produceSideInfo();
	Image<double> Iexp;
	for (int it = 0; it < Nit; it++)
	{
		double itError = 0;
		FOR_ALL_OBJECTS_IN_METADATA(DF)
		{
			FileName fnExp;
			DF.getValue(MDL_IMAGE, fnExp, __iter.objId);
			double rot;
			DF.getValue(MDL_ANGLE_ROT, rot, __iter.objId);
			double tilt;
			DF.getValue(MDL_ANGLE_TILT, tilt, __iter.objId);
			double psi;
			DF.getValue(MDL_ANGLE_PSI, psi, __iter.objId);
			double shiftX;
			DF.getValue(MDL_SHIFT_X, shiftX, __iter.objId);
			double shiftY;
			DF.getValue(MDL_SHIFT_Y, shiftY, __iter.objId);
			std::vector<double> lambda;
			if (NMA.size() > 0)
				DF.getValue(MDL_NMA, lambda, __iter.objId);

			Iexp.read(fnExp);
			Iexp().setXmippOrigin();
			itError += ART_single_step(Iexp(), rot, tilt, psi, -shiftX, -shiftY, lambda);
		}
		if (DF.size() > 0)
			itError /= DF.size();
		std::cerr << "Error at iteration " << it << " = " << itError << std::endl;
	}
	writePseudo();
}

void ProgARTPseudo::writePseudo()
{
	// Convert from pseudoatoms to volume
	Image<double> V;
	size_t objId = DF.firstObject();
	FileName fnExp;
	DF.getValue(MDL_IMAGE, fnExp, objId);
	Image<double> I;
	I.read(fnExp, HEADER);
	V().resize(XSIZE(I()), XSIZE(I()), XSIZE(I()));
	V().setXmippOrigin();

	int nmax = atomPosition.size();
	double sigma4 = 4 * sigma;
	for (int n = 0; n < nmax; n++)
	{
		int XX_corner1 = CEIL(XMIPP_MAX(STARTINGX(V()), XX(atomPosition[n]) - sigma4));
		int YY_corner1 = CEIL(XMIPP_MAX(STARTINGY(V()), YY(atomPosition[n]) - sigma4));
		int ZZ_corner1 = CEIL(XMIPP_MAX(STARTINGY(V()), ZZ(atomPosition[n]) - sigma4));
		int XX_corner2 = FLOOR(XMIPP_MIN(FINISHINGX(V()), XX(atomPosition[n]) + sigma4));
		int YY_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(V()), YY(atomPosition[n]) + sigma4));
		int ZZ_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(V()), ZZ(atomPosition[n]) + sigma4));
		if (XX_corner1 <= XX_corner2 && YY_corner1 <= YY_corner2 &&
			ZZ_corner1 <= ZZ_corner2)
		{
			for (int z = ZZ_corner1; z <= ZZ_corner2; z++)
				for (int y = YY_corner1; y <= YY_corner2; y++)
					for (int x = XX_corner1; x <= XX_corner2; x++)
						V(z, y, x) += atomWeight[n] *
						gaussian1D(z - ZZ(atomPosition[n]), sigma)*
						gaussian1D(y - YY(atomPosition[n]), sigma)*
						gaussian1D(x - XX(atomPosition[n]), sigma);
		}
	}
	V.write(fnRoot + ".vol");

	// Histogram of the intensities
	MultidimArray<double> intensities(atomWeight);
	Histogram1D hist;
	compute_hist(intensities, hist, 100);
	hist.write(fnRoot + "_intensities.hist");
}

double ProgARTPseudo::ART_single_step(const MultidimArray<double> &Iexp,
	double rot, double tilt, double psi, double shiftX, double shiftY,
	const std::vector<double> &lambda)
{
	MultidimArray<double> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Matrix2D<double> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	project_Pseudo(atomPosition, atomWeight, sigma, Itheo, Icorr,
		Euler, shiftX, shiftY, lambda, NMA, FORWARD,
		gaussianProjectionTable, gaussianProjectionTable2);
	Idiff.initZeros(Iexp);

	double mean_error = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
	{
		// Compute difference image and error
		DIRECT_A2D_ELEM(Idiff, i, j) = DIRECT_A2D_ELEM(Iexp, i, j) - DIRECT_A2D_ELEM(Itheo, i, j);
		mean_error += DIRECT_A2D_ELEM(Idiff, i, j) * DIRECT_A2D_ELEM(Idiff, i, j);

		// Compute the correction image
		DIRECT_A2D_ELEM(Icorr, i, j) = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
		DIRECT_A2D_ELEM(Icorr, i, j) =
			lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
	}
	mean_error /= YXSIZE(Iexp);

	project_Pseudo(atomPosition, atomWeight, sigma, Itheo, Icorr,
		Euler, shiftX, shiftY, lambda, NMA, BACKWARD,
		gaussianProjectionTable, gaussianProjectionTable2);
	return mean_error;
}*/