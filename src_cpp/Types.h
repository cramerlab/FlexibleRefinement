#pragma once
#ifndef CORE_TYPES_H
#define CORE_TYPES_H
//#include "liblionImports.h"
#include <vector>
typedef unsigned char uchar;
typedef unsigned int uint;
typedef float tfloat;
typedef size_t idxtype;
#define tmin(a, b) (((a) < (b)) ? (a) : (b))
#define tmax(a, b) (((a) > (b)) ? (a) : (b))

#define PI 3.1415926535897932384626433832795f
#define PI2 6.283185307179586476925286766559f
#define PIHALF 1.5707963267948966192313216916398f

#define ToRad(x) ((tfloat)(x) / (tfloat)180 * PI)
#define ToDeg(x) ((tfloat)(x) / PI * (tfloat)180)

#define getOffset(x, y, stride) ((y) * (stride) + (x))
#define getOffset3(x, y, z, stridex, stridey) (((z) * (stridey) + (y)) * (stridex) + (x))
#define DimensionCount(dims) (3 - tmax(2 - tmax((dims).z, 1), 0) - tmax(2 - tmax((dims).y, 1), 0) - tmax(2 - tmax((dims).x, 1), 0))
#define NextMultipleOf(value, base) (((value) + (base) - 1) / (base) * (base))
#define ElementsFFT1(dims) ((dims) / 2 + 1)
#define Elements2(dims) ((dims).x * (dims).y)
#define ElementsFFT2(dims) (ElementsFFT1((dims).x) * (dims).y)
#define Elements(dims) (Elements2(dims) * (dims).z)
#define ElementsFFT(dims) (ElementsFFT1((dims).x) * (dims).y * (dims).z)
#define FFTShift(x, dim) (((x) + (dim) / 2) % (dim))
#define IFFTShift(x, dim) (((x) + ((dim) + 1) / 2) % (dim))

#define crossp(a, b) tfloat3((a).y * (b).z - (a).z * (b).y, (a).z * (b).x - (a).x * (b).z, (a).x * (b).y - (a).y - (b).x)
#define dotp(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define dotp2(a, b) ((a).x * (b).x + (a).y * (b).y)



struct int2
{
	int x, y;
};

struct uint2
{
	unsigned int x, y;
};

struct int3
{
	int x, y, z;


	int3() {
		x = 0;
		y = 0;
		z = 0;
	}

	int3(int X, int Y, int Z) {
		x = X;
		y = Y;
		z = Z;
	}

	int3 operator+(int3 right) {
		return int3(x + right.x, y + right.y, z + right.z);
	}
};

struct uint3
{
	unsigned int x, y, z;
};

struct float3
{
	float x, y, z;

	float3() {
		x = 0;
		y = 0;
		z = 0;
	}

	float3(float X, float Y, float Z){
		x = X;
		y = Y;
		z = Z;
	}
	float3(float f) {
		x = f;
		y = f;
		z = f;
	}
	float3 operator+(float3 right) {
		return float3(x + right.x, y + right.y, z + right.z);
	}


	float3 operator-(float3 right) {
		return float3(x - right.x, y - right.y, z - right.z);
	}

	float3 operator+(float right) {
		return float3(x + right, y + right, z + right);
	}

	float3 operator/(float right) {
		return float3(x / right, y / right, z / right);
	}

	float3 operator*(float right) {
		return float3(x * right, y * right, z * right);
	}


};

struct float2
{
	float x, y;
};



inline int2 toInt2(int x, int y);

inline int2 toInt2(int3 dims);

inline int2 toInt2FFT(int2 val);

inline int2 toInt2FFT(int3 val);

inline uint2 toUint2(uint x, uint y);

inline uint2 toUint2(int2 o);

inline int3 toInt3(int x, int y, int z);

inline int3 toInt3(float3 val);

inline int3 toInt3FFT(int3 val);

inline int3 toInt3FFT(int2 val);

inline uint3 toUint3(uint x, uint y, uint z);

inline uint3 toUint3(int x, int y, int z);

inline uint3 toUint3(int3 o);


inline int3 toInt3(int2 val);

inline float3 make_float3(float x, float y, float z);

inline int2 toInt2(int x, int y)
{
	int2 value = { x, y };
	return value;
}

inline int2 toInt2(int3 dims)
{
	int2 value = { dims.x, dims.y };
	return value;
}

inline int3 toInt3(float3 val) {
	int3 value = { (int)val.x, (int)val.y, (int)val.z };
	return value;
}

inline int2 toInt2FFT(int2 val)
{
	int2 value = { val.x / 2 + 1, val.y };
	return value;
}

inline int2 toInt2FFT(int3 val)
{
	int2 value = { val.x / 2 + 1, val.y };
	return value;
}

inline uint2 toUint2(uint x, uint y)
{
	uint2 value = { x, y };
	return value;
}

inline uint2 toUint2(int2 o)
{
	uint2 value = { (uint)o.x, (uint)o.y };
	return value;
}

inline int3 toInt3(int x, int y, int z)
{
	int3 value = { x, y, z };
	return value;
}

inline int3 toInt3FFT(int3 val)
{
	int3 value = { val.x / 2 + 1, val.y, val.z };
	return value;
}

inline int3 toInt3FFT(int2 val)
{
	int3 value = { val.x / 2 + 1, val.y, 1 };
	return value;
}

inline uint3 toUint3(uint x, uint y, uint z)
{
	uint3 value = { x, y, z };
	return value;
}

inline uint3 toUint3(int x, int y, int z)
{
	uint3 value = { (uint)x, (uint)y, (uint)z };
	return value;
}

inline uint3 toUint3(int3 o)
{
	uint3 value = { (uint)o.x, (uint)o.y, (uint)o.z };
	return value;
}


inline int3 toInt3(int2 val)
{
	int3 value = { val.x, val.y, 1 };
	return value;
}

inline float3 make_float3(float x, float y, float z)
{
	float3 t; t.x = x; t.y = y; t.z = z; return t;
}



inline int3 make_int3(int x, int y, int z)
{
	int3 t; t.x = x; t.y = y; t.z = z; return t;
}


float3 mean(std::vector<float3> vec);
/*
float3 mean(MultidimArray<float3> &vec) {
	float3 sum = { 0,0,0 };
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vec)
		sum = sum + vec.data[n];

	return sum / NZYXSIZE(vec);
}
*/
#endif
