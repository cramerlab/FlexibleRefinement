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
		_fseeki64(inputfile, (long)header.extendedbytes, SEEK_CUR);
#elif __GNUC__ > 3
		fseeko64(inputfile, (long)header.extendedbytes, SEEK_CUR);
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
		DOUBLE M = 0;
		FOR_ALL_ELEMENTS_IN_ARRAY3D(data)
		{
			DOUBLE weight = A3D_ELEM(data, k, i, j);
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


	/* Instantate possible usages*/

	template void MRCImage<double>::writeAs<float>(std::string path, bool doStatistics);
	template void MRCImage<float>::writeAs<float>(std::string path, bool doStatistics);
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