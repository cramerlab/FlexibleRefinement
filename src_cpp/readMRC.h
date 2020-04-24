#pragma once

#include "liblionImports.h"
#include "Types.h"

#include <string>
#ifndef IO_CUH
#define IO_CUH

namespace relion
{
	//////
	//IO//
	//////

	enum MRC_DATATYPE
	{
		MRC_BYTE = 0,
		MRC_SHORT = 1,
		MRC_FLOAT = 2,
		MRC_SHORTCOMPLEX = 3,
		MRC_FLOATCOMPLEX = 4,
		MRC_UNSIGNEDSHORT = 6,
		MRC_RGB = 16
	};

	const size_t MRC_DATATYPE_SIZE[17] = { 1, 2, 4, 4, 8, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3 };

	struct HeaderMRC
	{
		int3 dimensions;
		MRC_DATATYPE mode;
		int3 startsubimage;
		int3 griddimensions;
		float3 pixelsize;
		float3 angles;
		int3 maporder;

		float minvalue;
		float maxvalue;
		float meanvalue;
		int spacegroup;

		int extendedbytes;
		short creatid;

		uchar extradata1[30];

		short nint;
		short nreal;

		uchar extradata2[28];

		short idtype;
		short lens;
		short nd1;
		short nd2;
		short vd1;
		short vd2;

		float3 tiltoriginal;
		float3 tiltcurrent;
		float3 origin;

		uchar cmap[4];
		uchar stamp[4];

		float stddevvalue;

		int numlabels;
		uchar labels[10][80];

		HeaderMRC() :
			dimensions(toInt3(1, 1, 1)),
			mode(MRC_FLOAT),
			startsubimage(toInt3(0, 0, 0)),
			griddimensions(toInt3(1, 1, 1)),
			pixelsize(make_float3(1.0f, 1.0f, 1.0f)),
			angles(make_float3(0, 0, 0)),
			maporder(toInt3(1, 2, 3)),
			minvalue(-1.0f),
			maxvalue(1.0f),
			meanvalue(0),
			extendedbytes(0),
			creatid(0),
			nint(0),
			nreal(0),
			idtype(0),
			lens(0),
			nd1(0),
			nd2(0),
			vd1(0),
			vd2(0),
			tiltoriginal(make_float3(0, 0, 0)),
			tiltcurrent(make_float3(0, 0, 0)),
			origin(make_float3(0, 0, 0)),
			stddevvalue(0),
			numlabels(0) {}
	};

	enum EM_DATATYPE : unsigned char
	{
		EM_BYTE = 1,
		EM_SHORT = 2,
		EM_SHORTCOMPLEX = 3,
		EM_LONG = 4,
		EM_SINGLE = 5,
		EM_SINGLECOMPLEX = 8,
		EM_DOUBLE = 9,
		EM_DOUBLECOMPLEX = 10
	};

	const size_t EM_DATATYPE_SIZE[11] = { 1, 1, 2, 4, 4, 4, 8, 8, 16 };

	struct HeaderEM
	{
		uchar machinecoding;
		uchar os9;
		uchar invalid;
		EM_DATATYPE mode;

		int3 dimensions;

		uchar comment[80];

		int voltage;
		int Cs;
		int aperture;
		int magnification;
		int ccdmagnification;
		int exposuretime;
		int pixelsize;
		int emcode;
		int ccdpixelsize;
		int ccdwidth;
		int defocus;
		int astigmatism;
		int astigmatismangle;
		int focusincrement;
		int qed;
		int c2intensity;
		int slitwidth;
		int energyoffset;
		int tiltangle;
		int tiltaxis;
		int noname1;
		int noname2;
		int noname3;
		int2 markerposition;
		int resolution;
		int density;
		int contrast;
		int noname4;
		int3 centerofmass;
		int height;
		int noname5;
		int dreistrahlbereich;
		int achromaticring;
		int lambda;
		int deltatheta;
		int noname6;
		int noname7;

		uchar userdata[256];
	};

	template <typename T>
	class MRCImage
	{
	
	private:
			
			relion::MultidimArray<T> data;
	public:
		HeaderMRC header;

			static HeaderMRC ReadMRCHeader(FILE* inputfile);
			static HeaderMRC ReadMRCHeader(std::string path);

			static void ReadMRC(std::string path, relion::MultidimArray<T> &data, int nframe = -1);
			static void ReadMRC(std::string path, relion::MultidimArray<T> &data, HeaderMRC &header, int nframe = -1);
			static void ReadMRC(std::string path, void** data, int nframe = -1);
			static void WriteMRC(void* data, HeaderMRC &header, std::string path);


			static MRCImage<T> readAs(std::string path);

			template <typename I>
			void writeAs(std::string path, bool doStatistics=false);

			MRCImage() {
				header = HeaderMRC();
				setType();

			}

			MRCImage(std::string path) {
			
				ReadMRC(path, data, header, -1);
			}

			MRCImage(FileName path) {

				ReadMRC(path, data, header, -1);
			}

			MRCImage(HeaderMRC p_header) {

				header = p_header;
				setType(); //overwrite type in header to type of image
			}


			MRCImage(relion::MultidimArray<T> p_data) {

				header = HeaderMRC();
				header.dimensions.x = p_data.xdim;
				header.dimensions.y = p_data.ydim;
				header.dimensions.z = p_data.zdim;

				data = p_data;
			}

			float3 getCenterOfMass();

			void setData(relion::MultidimArray<T> p_data, bool copy = false) {
				header.dimensions.x = p_data.xdim;
				header.dimensions.z = p_data.zdim;
				header.dimensions.y = p_data.ydim;

				if (!copy)
					data = p_data;
				else {
					data.resize(p_data);
					memcpy(data.data, p_data.data, sizeof(T)*data.nzyxdim);
				}
			}

			bool checkType() {
				return false;
			}

			void setType() {
				
			}

			HeaderMRC getHeader() {
				return header;
			}

			void WriteMRC(std::string path);

			void write(std::string path);

			void setXmippOrigin() {
				data.setXmippOrigin();
			}
			
			void setZeroOrigin() {
				data.xinit = 0;
				data.yinit = 0;
				data.zinit = 0;
			}

			relion::MultidimArray<T>& operator()(){
				return data;
			}


			const relion::MultidimArray<T>& operator()() const{
				return data;
			}

			T & operator()(int k, int i, int j) {
				return A3D_ELEM(data, k,i,j);
			}


			

	};


	//mrc.cu:
	

	/*
	//em.cu:
	void ReadEM(std::string path, void** data, int nframe = -1);
	HeaderEM ReadEMHeader(std::string path);
	HeaderEM ReadEMHeader(FILE* inputfile);

	//raw.cu:
	void ReadRAW(std::string path, void** data, EM_DATATYPE datatype, int3 dims, size_t headerbytes, int nframe = -1);*/
}
#endif