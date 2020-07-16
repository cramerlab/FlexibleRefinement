#include "readMRC.h"
#include <limits>
namespace relion
{
	template<>
	bool MRCImage<float>::checkType() {
		return header.mode==MRC_FLOAT;
	}

	template<>
	bool MRCImage<__int16>::checkType() {
		return header.mode == MRC_SHORT;
	}

	template<>
	bool MRCImage<__int8>::checkType() {
		return header.mode == MRC_BYTE;
	}

	template<>
	bool MRCImage<unsigned __int16>::checkType() {
		return header.mode == MRC_UNSIGNEDSHORT;
	}

	template<>
	void MRCImage<float>::setType() {
		header.mode = MRC_FLOAT;
	}

	template<>
	void MRCImage<__int16>::setType() {
		header.mode = MRC_SHORT;
	}

	template<>
	void MRCImage<__int8>::setType() {
		header.mode = MRC_BYTE;
	}

	template<>
	void MRCImage<unsigned __int16>::setType() {
		header.mode = MRC_UNSIGNEDSHORT;
	}


	template <typename T>
	HeaderMRC MRCImage<T>::ReadMRCHeader(FILE* inputfile) {
		HeaderMRC header;
		char* headerp = (char*)&header;

		fread(headerp, sizeof(char), sizeof(HeaderMRC), inputfile);
#if _MSC_VER > 1
		_fseeki64(inputfile, (size_t)header.extendedbytes, SEEK_CUR);
#elif __GNUC__ > 3
		fseeko64(inputfile, (size_t)header.extendedbytes, SEEK_CUR);
#endif
		return header;
	}


	template <typename T>
	HeaderMRC MRCImage<T>::ReadMRCHeader(std::string path) {
		FILE* inputfile = fopen(path.c_str(), "rb");

		if (inputfile == NULL) {
			REPORT_ERROR("Failed to open File" + path);
		}

#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderMRC header = ReadMRCHeader(inputfile);
		fclose(inputfile);

		return header;
	}
	
	template <typename T>
	void MRCImage<T>::ReadMRC(std::string path, relion::MultidimArray<T> &data, int nframe) {
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderMRC header = ReadMRCHeader(inputfile);

		size_t datasize;
		if (nframe >= 0)
			datasize = Elements2(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];
		else
			datasize = Elements(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];

		data.resizeNoCopy(header.dimensions.z, header.dimensions.y, header.dimensions.x);

		if (nframe >= 0)
#if _MSC_VER > 1
			_fseeki64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#elif __GNUC__ > 3
			fseeko64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#endif

		fread(data.data, sizeof(char), datasize, inputfile);

		fclose(inputfile);
	}

	template <typename T>
	void MRCImage<T>::ReadMRC(std::string path, relion::MultidimArray<T> &data, HeaderMRC &header, int nframe) {
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		header = ReadMRCHeader(inputfile);

		size_t datasize;
		if (nframe >= 0)
			datasize = Elements2(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];
		else
			datasize = Elements(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];

		data.resizeNoCopy(header.dimensions.z, header.dimensions.y, header.dimensions.x);

		if (nframe >= 0)
#if _MSC_VER > 1
			_fseeki64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#elif __GNUC__ > 3
			fseeko64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#endif

		fread(data.data, sizeof(char), datasize, inputfile);

		fclose(inputfile);
	}

	template <typename T>
	void MRCImage<T>::ReadMRC(std::string path, void** data, int nframe) {
		FILE* inputfile = fopen(path.c_str(), "rb");
#if _MSC_VER > 1
		_fseeki64(inputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(inputfile, 0L, SEEK_SET);
#endif

		HeaderMRC header = ReadMRCHeader(inputfile);

		size_t datasize;
		if (nframe >= 0)
			datasize = Elements2(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];
		else
			datasize = Elements(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];

		*data = malloc(datasize);

		if (nframe >= 0)
#if _MSC_VER > 1
			_fseeki64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#elif __GNUC__ > 3
			fseeko64(inputfile, datasize * (size_t)nframe, SEEK_CUR);
#endif

		fread(*data, sizeof(char), datasize, inputfile);

		fclose(inputfile);
	}

	template <typename T>
	void MRCImage<T>::WriteMRC(void* data, HeaderMRC &header, std::string path){
		FILE* outputfile = fopen(path.c_str(), "wb");
		if (outputfile == NULL) {
			REPORT_ERROR(path + " Failed to open ");
		}
#if _MSC_VER > 1
		_fseeki64(outputfile, 0L, SEEK_SET);
#elif __GNUC__ > 3
		fseeko64(outputfile, 0L, SEEK_SET);
#endif

		fwrite(&header, sizeof(HeaderMRC), 1, outputfile);

		size_t elementsize = MRC_DATATYPE_SIZE[(int)header.mode];
		fwrite(data, elementsize, Elements(header.dimensions), outputfile);

		fclose(outputfile);
	}

	template <typename T>
	void MRCImage<T>::WriteMRC(std::string path) {
		WriteMRC(data.data, header, path);
	}

	template <typename T>
	void MRCImage<T>::write(std::string path)
	{
		WriteMRC(path);
	}

	template<typename T>
	RDOUBLE MRCImage<T>::GetInterpolatedValue(float3 pos)
	{
		float3 Weights = { pos.x - (RDOUBLE)std::floor(pos.x),
			pos.y - (RDOUBLE)std::floor(pos.y),
			pos.z - (RDOUBLE)std::floor(pos.z) };



		int3 Pos0 = { std::max((size_t)0, std::min(data.xdim - 1, (size_t)pos.x)),
			std::max((size_t)0, std::min(data.ydim - 1, (size_t)pos.y)),
			std::max((size_t)0, std::min(data.zdim - 1, (size_t)pos.z)) };
		int3 Pos1 = {std::min(data.xdim - 1, (size_t)(Pos0.x + 1)),
			std::min(data.ydim - 1, (size_t)Pos0.y + 1),
			std::min(data.zdim - 1, (size_t)Pos0.z + 1)};

		if (data.zdim == 1)
		{
			RDOUBLE v00 = DIRECT_A3D_ELEM(data, 0, Pos0.y, Pos0.x);
			RDOUBLE v01 = DIRECT_A3D_ELEM(data, 0, Pos0.y, Pos1.x);
			RDOUBLE v10 = DIRECT_A3D_ELEM(data, 0, Pos1.y, Pos0.x);
			RDOUBLE v11 = DIRECT_A3D_ELEM(data, 0, Pos1.y, Pos1.x);

			RDOUBLE v0 = Lerp(v00, v01, Weights.x);
			RDOUBLE v1 = Lerp(v10, v11, Weights.x);

			return Lerp(v0, v1, Weights.y);
		}
		else
		{
			RDOUBLE v000 = DIRECT_A3D_ELEM(data, Pos0.z, Pos0.y, Pos0.x);
			RDOUBLE v001 = DIRECT_A3D_ELEM(data, Pos0.z, Pos0.y, Pos1.x);
			RDOUBLE v010 = DIRECT_A3D_ELEM(data, Pos0.z, Pos1.y, Pos0.x);
			RDOUBLE v011 = DIRECT_A3D_ELEM(data, Pos1.z, Pos0.y, Pos0.x);

			RDOUBLE v100 = DIRECT_A3D_ELEM(data, Pos1.z, Pos0.y, Pos0.x);
			RDOUBLE v101 = DIRECT_A3D_ELEM(data, Pos1.z, Pos0.y, Pos1.x);
			RDOUBLE v110 = DIRECT_A3D_ELEM(data, Pos1.z, Pos1.y, Pos0.x);
			RDOUBLE v111 = DIRECT_A3D_ELEM(data, Pos1.z, Pos1.y, Pos1.x);

			RDOUBLE v00 = Lerp(v000, v001, Weights.x);
			RDOUBLE v01 = Lerp(v010, v011, Weights.x);
			RDOUBLE v10 = Lerp(v100, v101, Weights.x);
			RDOUBLE v11 = Lerp(v110, v111, Weights.x);

			RDOUBLE v0 = Lerp(v00, v01, Weights.y);
			RDOUBLE v1 = Lerp(v10, v11, Weights.y);

			RDOUBLE tmp = Lerp(v0, v1, Weights.z);
			return tmp;
		}
	}

	template<typename T>
	MRCImage<T> MRCImage<T>::readAs(std::string path){
		HeaderMRC header = MRCImage<T>::ReadMRCHeader(path);
		MRCImage<T> returnImage(header);
		MultidimArray<T> castedData;
		castedData.resizeNoCopy(header.dimensions.z, header.dimensions.y, header.dimensions.x);
		if (header.mode == MRC_FLOAT) {
			MRCImage<float> readIm(path);
			
			castedData.resize(readIm());
			FOR_ALL_ELEMENTS_IN_ARRAY3D(readIm()) {
				DIRECT_A3D_ELEM(castedData, k, i, j) = static_cast<T>(readIm(k, i, j));
			}
		}
		else if (header.mode == MRC_BYTE) {
			MRCImage<__int8> readIm(path);

			castedData.resize(readIm());
			FOR_ALL_ELEMENTS_IN_ARRAY3D(readIm()) {
				DIRECT_A3D_ELEM(castedData, k, i, j) = static_cast<T>(readIm(k, i, j));
			}
		}
		else if (header.mode == MRC_SHORT) {
			MRCImage<__int16> readIm(path);

			castedData.resize(readIm());
			FOR_ALL_ELEMENTS_IN_ARRAY3D(readIm()) {
				DIRECT_A3D_ELEM(castedData, k, i, j) = static_cast<T>(readIm(k, i, j));
			}
		}
		else if (header.mode == MRC_UNSIGNEDSHORT) {
			MRCImage<unsigned __int16> readIm(path);

			castedData.resize(readIm());
			FOR_ALL_ELEMENTS_IN_ARRAY3D(readIm()) {
				DIRECT_A3D_ELEM(castedData, k, i, j) = static_cast<T>(readIm(k, i, j));
			}
		}
		else {
			REPORT_ERROR("Unkown Datatype ");
		}
		returnImage.setData(castedData, true);
		return returnImage;
	}


	template <typename T>
	template <typename I>
	void MRCImage<T>::writeAs(std::string path, bool doStatistics)
	{
		MRCImage<I> writeIm;
		MultidimArray<I> writeData;
		writeData.resize(data);
		writeData.setXmippOrigin();
		I min = std::numeric_limits<I>::max();
		I max = std::numeric_limits<I>::lowest();
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(data) {
			I value = static_cast<I>(DIRECT_A3D_ELEM(data, k, i, j));
			DIRECT_A3D_ELEM(writeData, k, i, j) = value;
			min = std::min(value, min);
			max = std::max(value, max);
		}
		writeIm.setData(writeData);
		writeIm.header.maxvalue = max;
		writeIm.header.minvalue = min;
		writeIm.WriteMRC(path);

	}

	template <typename T>
	float3 MRCImage<T>::getCenterOfMass() {
		float3 center = { 0.0f, 0.0f, 0.0f };
		RDOUBLE M = 0;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(data)
		{
			RDOUBLE weight = A3D_ELEM(data, k, i, j);
			center.x += weight * j;
			center.y += weight * i;
			center.z += weight * k;
			M += weight;
		}
		center.x /= M;
		center.y /= M;
		center.z /= M;
		return center;
	}

	template<>
	template<>
	void MRCImage<float>::writeAs<float>(std::string path, bool doStatistics)
	{

		float min = std::numeric_limits<float>::max();
		float max = std::numeric_limits<float>::lowest();
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(data) {
			float value = DIRECT_A3D_ELEM(data, k, i, j);
			min = std::min(value, min);
			max = std::max(value, max);
		}
		header.minvalue = min;
		header.maxvalue = max;
		WriteMRC(path);
	}

	/* Instantate possible usages*/

	template void MRCImage<double>::writeAs<float>(std::string path, bool doStatistics);

	template void MRCImage<int>::writeAs<float>(std::string path, bool doStatistics);
	template void MRCImage<__int16>::writeAs<float>(std::string path, bool doStatistics);
	template void MRCImage<unsigned __int16>::writeAs<float>(std::string path, bool doStatistics);
	template void MRCImage<__int8>::writeAs<float>(std::string path, bool doStatistics);

	template void MRCImage<double>::writeAs<__int16>(std::string path, bool doStatistics);
	template void MRCImage<float>::writeAs<__int16>(std::string path, bool doStatistics);
	template void MRCImage<int>::writeAs<__int16>(std::string path, bool doStatistics);
	template void MRCImage<__int16>::writeAs<float>(std::string path, bool doStatistics);
	template void MRCImage<__int16>::writeAs<__int16>(std::string path, bool doStatistics);
	template void MRCImage<unsigned __int16>::writeAs<__int16>(std::string path, bool doStatistics);
	template void MRCImage<__int8>::writeAs<__int16>(std::string path, bool doStatistics);

	template void MRCImage<double>::writeAs<unsigned __int16>(std::string path, bool doStatistics);
	template void MRCImage<float>::writeAs<unsigned __int16>(std::string path, bool doStatistics);
	template void MRCImage<int>::writeAs<unsigned __int16>(std::string path, bool doStatistics);
	template void MRCImage<__int16>::writeAs<unsigned __int16>(std::string path, bool doStatistics);
	template void MRCImage<unsigned __int16>::writeAs<unsigned __int16>(std::string path, bool doStatistics);
	template void MRCImage<__int8>::writeAs<unsigned __int16>(std::string path, bool doStatistics);

	template void MRCImage<double>::writeAs<__int8>(std::string path, bool doStatistics);
	template void MRCImage<float>::writeAs<__int8>(std::string path, bool doStatistics);
	template void MRCImage<int>::writeAs<__int8>(std::string path, bool doStatistics);
	template void MRCImage<__int16>::writeAs<__int8>(std::string path, bool doStatistics);
	template void MRCImage<unsigned __int16>::writeAs<__int8>(std::string path, bool doStatistics);
	template void MRCImage<__int8>::writeAs<__int8>(std::string path, bool doStatistics);


	template class MRCImage<double>;
	template class MRCImage<float>;
	template class MRCImage<int>;
	template class MRCImage<__int16>;
	template class MRCImage<unsigned __int16>;
	template class MRCImage<__int8>;


}