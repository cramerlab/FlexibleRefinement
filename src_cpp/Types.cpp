#pragma once
#include "Types.h"
float3 mean(std::vector<float3> vec) {
	float3 Sum = { 0, 0, 0 };
	for (auto p : vec)
		Sum = Sum + p;

	return Sum / vec.size();
}

FR_float3 mean(std::vector<FR_float3> vec) {
	FR_float3 Sum = { 0, 0, 0 };
	for (auto p : vec)
		Sum = Sum + p;

	return Sum / vec.size();
}
/*
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
}*/