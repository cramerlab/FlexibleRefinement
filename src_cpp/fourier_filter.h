/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef _FOURIER_FILTER_HH
#define _FOURIER_FILTER_HH

#include "liblionImports.h"
#include "Types.h"
#include "readMRC.h"

using namespace relion;

#define FFT_IDX2DIGFREQ(idx, size, freq) \
    freq = (size<=1)? 0:(( (((int)idx) <= (((int)(size)) >> 1)) ? ((int)(idx)) : -((int)(size)) + ((int)(idx))) / \
           (RDOUBLE)(size));

#define FFT_IDX2DIGFREQ_RDOUBLE(idx, size, freq) \
    freq = (size<=1)? 0:(( (((RDOUBLE)idx) <= (((RDOUBLE)(size)) / 2.0)) ? ((RDOUBLE)(idx)) : -((RDOUBLE)(size)) + ((RDOUBLE)(idx))) / \
           (RDOUBLE)(size));

#define FFT_IDX2DIGFREQ_FAST(idx, size, size_2, isize, freq) \
    freq = ( ((idx) <= (size_2)) ? (idx) : -(size) + (idx) ) * (isize);

//#include "filters.h"

/**@defgroup FourierMasks Masks in Fourier space
   @ingroup ReconsLibrary */
//@{
/** Filter class for Fourier space.

   Example of use for highpass filtering
   @code
      Image<RDOUBLE> I;
      I.read("image.xmp");
      FourierFilter Filter;
      Filter.FilterBand=HIGHPASS;
      Filter.w1=w_cutoff;
      Filter.raised_w=slope;
      I().setXmippOrigin();
      Filter.applyMaskSpace(I());
      I.write("filtered_image.xmp");
   @endcode
   
   Example of use for wedge filtering
   @code
        Image<RDOUBLE> V;
        V.read("1rux64.vol");
        V().setXmippOrigin();
        FourierFilter Filter;
        Filter.FilterBand=WEDGE;
        Filter.FilterShape=WEDGE;
        Filter.t1=-60;
        Filter.t2= 60;
        Filter.rot=Filter.tilt=Filter.psi=0;
        Filter.do_generate_3dmask=true;
        Filter.generateMask(V());
        Filter.applyMaskSpace(V());
   @endcode

   For volumes you the mask is computed on the fly and
   in this way memory is saved (unless do_generate_3dmask == true).
*/
class FourierFilter
{
public:
#define RAISED_COSINE 1
    /** Shape of the decay in the filter.
       Valid types are RAISED_COSINE. */
    int FilterShape;

#define LOWPASS       1
#define HIGHPASS      2
#define BANDPASS      3
#define STOPBAND      4
#define CTF           5
#define WEDGE         7
#define GAUSSIAN      8
#define CONE          9
#define CTFPOS       10
#define BFACTOR      11
#define REALGAUSSIAN 12
#define SPARSIFY     13
#define STOPLOWBANDX 14
#define STOPLOWBANDY 15
#define FSCPROFILE   16
#define BINARYFILE   17
#define ASTIGMATISMPROFILE 18
#define CTFINV       19
#define CTFPOSINV    20

    /** Pass band. LOWPASS, HIGHPASS, BANDPASS, STOPBAND, CTF, CTFPOS,
       WEDGE, CONE, GAUSSIAN, FROM_FILE, REALGAUSSIAN, BFACTOR, SPARSIFY,
       STOPLOWBANDX, STOPLOWBANDY, FSCPROFILE, BINARYFILE, ASTIGMATISMPROFILE,
       CTFINV, CTFPOSINV */
    int FilterBand;

    /** Cut frequency for Low and High pass filters, first freq for bandpass.
        Normalized to 1/2*/
    RDOUBLE w1;

    /** Second frequency for bandpass and stopband. Normalized to 1/2 */
    RDOUBLE w2;

    /** Input Image sampling rate */
    RDOUBLE sampling_rate;

    /** Wedge and cone filter parameters */
    RDOUBLE t1, t2,rot,tilt,psi;

    /** Percentage of coefficients to throw */
    RDOUBLE percentage;

    /** Filename in which store the mask (valid only for fourier masks) */
    FileName maskFn;

    /** Pixels around the central frequency for the raised cosine */
    RDOUBLE raised_w;

    ///** CTF parameters. */
    //CTFDescription ctf;
    
    /** Minimum CTF for inversion */
    RDOUBLE minCTF;

    /** FSC file */
    FileName fnFSC;

    /** Binary file with the filter */
    FileName fnFilter;

    /** Flag to generate 3D mask */
    bool do_generate_3dmask;

public:
    ///** Define parameters */
    //static void defineParams(XmippProgram * program);

    ///** Read parameters from command line.
    //    If a CTF description file is provided it is read. */
    //void readParams(XmippProgram * program);

    /** Process one image */
    void apply(MultidimArray<RDOUBLE> &img);

    /** Empty constructor */
    FourierFilter();

    /** Clear */
    void init();

    /** Show. */
    void show();

    /** Compute the mask value at a given frequency.
        The frequency must be normalized so that the maximum frequency
        in each direction is 0.5 */
    RDOUBLE maskValue(const Matrix1D<RDOUBLE> &w);

    /** Generate nD mask. */
    void generateMask(MultidimArray<RDOUBLE> &v);

    /** Apply mask in real space. */
    void applyMaskSpace(MultidimArray<RDOUBLE> &v);

    /** Apply mask in Fourier space.
     * The image remains in Fourier space.
     */
    void applyMaskFourierSpace(const MultidimArray<RDOUBLE> &v, MultidimArray<Complex> &V);

    /** Get the power of the nD mask. */
    RDOUBLE maskPower();
    
    /** Correct phase */
    void correctPhase();
public:
    // Auxiliary vector for representing frequency values
    Matrix1D<RDOUBLE> w;

    // Auxiliary mask for the filter in 3D
    MultidimArray<int> maskFourier;

    // Auxiliary mask for the filter in 3D
    MultidimArray<RDOUBLE> maskFourierd;

    // Transformer
    FourierTransformer transformer;

    // Auxiliary variables for sparsify
    MultidimArray<RDOUBLE> vMag, vMagSorted;

    // Auxiliary variables for FSC profile
    std::vector<RDOUBLE> freqContFSC, FSC;
};


///** Fast access to bandpass filter.
// * Frequencies are normalized to 0.5 */
void bandpassFilter(MultidimArray<RDOUBLE> &img, RDOUBLE w1, RDOUBLE w2, RDOUBLE raised_w);
//
///** Fast access to Gaussian filter.
// * Frequencies are normalized to 0.5 */
//void gaussianFilter(MultidimArray<RDOUBLE> &img, RDOUBLE w1);
//
///** Fast access to real gaussian filter.
// * Sigma is in pixel units.
// */
//void realGaussianFilter(MultidimArray<RDOUBLE> &img, RDOUBLE sigma);
//@}
#endif
