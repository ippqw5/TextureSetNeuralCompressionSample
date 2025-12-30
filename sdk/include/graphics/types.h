/*
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#pragma once

// Internal includes
#include "math/types.h"

#include <iostream>

// General graphics objects
typedef uint64_t GraphicsDevice;
typedef uint64_t RenderWindow;
typedef uint64_t CommandQueue;
typedef uint64_t SwapChain;
typedef uint64_t CommandBuffer;

// Shaders objects
typedef uint64_t ComputeShader;
typedef uint64_t GraphicsPipeline;
typedef uint64_t RayTracingShader;

// Syncrhonization
typedef uint64_t Fence;

// Resources objects
typedef uint64_t Texture;
typedef uint64_t RenderTexture;
typedef uint64_t GraphicsBuffer;
typedef uint64_t ConstantBuffer;
typedef uint64_t Sampler;
typedef uint64_t TopLevelAS;
typedef uint64_t BottomLevelAS;

// Profiling
typedef uint64_t ProfilingScope;

enum class GraphicsAPI
{
	DX12 = 0,
	Count
};

// Device pick strategy
enum class DevicePickStrategy
{
    VRAMSize = 0,
    VendorID,
    AdapterID,
	Count
};

// Vendors for the GPUs
enum class GPUVendor
{
    Intel = 0,
    AMD,
    Nvidia,
    Other,
	Count
};

enum class GPUFeature
{
	RayTracing = 0,
	DoubleOps,
	HalfOps,
	MeshShader,
	CoopMatrix,
	CoopVector,
	Count
};

// Types of constant buffers
enum class ConstantBufferType
{
    Static = 0x01,
    Runtime = 0x02,
    Mixed = 0x03
};

// Types of graphics buffers
enum class GraphicsBufferType
{
    Default = 0,
    Upload,
    Readback,
    RTAS,
    Count
};

enum class GraphicsBufferFlags
{
	Default = 0,
	VertexBuffer = 0x01,
	IndexBuffer = 0x02,
	Indirect = 0x04,
	Count
};

// Types of the command buffer
enum class CommandBufferType
{
    Default = 0,
    Compute = 2,
    Copy = 3
};

// Command queue priority
enum class CommandQueuePriority
{
    Normal = 0,
    High = 1,
    Realtime = 2,
	Count
};

// Types of textures
enum class TextureType
{
	Tex1D = 0,
	Tex1DArray,
	Tex2D,
	Tex2DArray,
	Tex3D,
	TexCube,
	TexCubeArray,
	Count
};
inline std::ostream& operator<<(std::ostream& os, TextureType type) {
	switch (type) {
		case TextureType::Tex1D: os << "Tex1D"; break;
		case TextureType::Tex1DArray: os << "Tex1DArray"; break;
		case TextureType::Tex2D: os << "Tex2D"; break;
		case TextureType::Tex2DArray: os << "Tex2DArray"; break;
		case TextureType::Tex3D: os << "Tex3D"; break;
		case TextureType::TexCube: os << "TexCube"; break;
		case TextureType::TexCubeArray: os << "TexCubeArray"; break;
		case TextureType::Count: os << "Count"; break;
		default: os << "Unknown"; break;
	}
	return os;
}

// Texture formats supported
enum class TextureFormat
{
	// 8 Formats
	R8_SNorm,
	R8_UNorm,
	R8_SInt,
	R8_UInt,
	R8G8_SNorm,
	R8G8_UNorm,
	R8G8_SInt,
	R8G8_UInt,
	R8G8B8A8_SNorm,
	R8G8B8A8_UNorm,
	R8G8B8A8_UNorm_SRGB,
	R8G8B8A8_UInt,
	R8G8B8A8_SInt,

	// 16 Formats
	R16_Float,
	R16_SInt,
	R16_UInt,
	R16G16_Float,
	R16G16_SInt,
	R16G16_UInt,
	R16G16B16A16_Float,
	R16G16B16A16_UInt,
	R16G16B16A16_SInt,

	// 32 Formats
	R32_Float,
	R32_SInt,
	R32_UInt,
	R32G32_Float,
	R32G32_SInt,
	R32G32_UInt,
	R32G32B32_UInt,
	R32G32B32_Float,
	R32G32B32A32_Float,
	R32G32B32A32_UInt,
	R32G32B32A32_SInt,

	// Depth buffer formats
	Depth32,
	Depth32Stencil8,
	Depth24Stencil8,

	// Other Formats
	R10G10B10A2_UNorm,
	R10G10B10A2_UInt,
	R11G11B10_Float,
	BC1_RGB,
	BC6_RGB,

	// Count
	Count
};
inline std::ostream& operator<<(std::ostream& os, TextureFormat format) {
    switch (format) {
        case TextureFormat::R8_SNorm: os << "R8_SNorm"; break;
        case TextureFormat::R8_UNorm: os << "R8_UNorm"; break;
        case TextureFormat::R8_SInt: os << "R8_SInt"; break;
        case TextureFormat::R8_UInt: os << "R8_UInt"; break;
        case TextureFormat::R8G8_SNorm: os << "R8G8_SNorm"; break;
        case TextureFormat::R8G8_UNorm: os << "R8G8_UNorm"; break;
        case TextureFormat::R8G8_SInt: os << "R8G8_SInt"; break;
        case TextureFormat::R8G8_UInt: os << "R8G8_UInt"; break;
        case TextureFormat::R8G8B8A8_SNorm: os << "R8G8B8A8_SNorm"; break;
        case TextureFormat::R8G8B8A8_UNorm: os << "R8G8B8A8_UNorm"; break;
        case TextureFormat::R8G8B8A8_UNorm_SRGB: os << "R8G8B8A8_UNorm_SRGB"; break;
        case TextureFormat::R8G8B8A8_UInt: os << "R8G8B8A8_UInt"; break;
        case TextureFormat::R8G8B8A8_SInt: os << "R8G8B8A8_SInt"; break;
        case TextureFormat::R16_Float: os << "R16_Float"; break;
        case TextureFormat::R16_SInt: os << "R16_SInt"; break;
        case TextureFormat::R16_UInt: os << "R16_UInt"; break;
        case TextureFormat::R16G16_Float: os << "R16G16_Float"; break;
        case TextureFormat::R16G16_SInt: os << "R16G16_SInt"; break;
        case TextureFormat::R16G16_UInt: os << "R16G16_UInt"; break;
        case TextureFormat::R16G16B16A16_Float: os << "R16G16B16A16_Float"; break;
        case TextureFormat::R16G16B16A16_UInt: os << "R16G16B16A16_UInt"; break;
        case TextureFormat::R16G16B16A16_SInt: os << "R16G16B16A16_SInt"; break;
        case TextureFormat::R32_Float: os << "R32_Float"; break;
        case TextureFormat::R32_SInt: os << "R32_SInt"; break;
        case TextureFormat::R32_UInt: os << "R32_UInt"; break;
        case TextureFormat::R32G32_Float: os << "R32G32_Float"; break;
        case TextureFormat::R32G32_SInt: os << "R32G32_SInt"; break;
        case TextureFormat::R32G32_UInt: os << "R32G32_UInt"; break;
        case TextureFormat::R32G32B32_UInt: os << "R32G32B32_UInt"; break;
        case TextureFormat::R32G32B32_Float: os << "R32G32B32_Float"; break;
        case TextureFormat::R32G32B32A32_Float: os << "R32G32B32A32_Float"; break;
        case TextureFormat::R32G32B32A32_UInt: os << "R32G32B32A32_UInt"; break;
        case TextureFormat::R32G32B32A32_SInt: os << "R32G32B32A32_SInt"; break;
        case TextureFormat::Depth32: os << "Depth32"; break;
        case TextureFormat::Depth32Stencil8: os << "Depth32Stencil8"; break;
        case TextureFormat::Depth24Stencil8: os << "Depth24Stencil8"; break;
        case TextureFormat::R10G10B10A2_UNorm: os << "R10G10B10A2_UNorm"; break;
        case TextureFormat::R10G10B10A2_UInt: os << "R10G10B10A2_UInt"; break;
        case TextureFormat::R11G11B10_Float: os << "R11G11B10_Float"; break;
        case TextureFormat::BC1_RGB: os << "BC1_RGB"; break;
        case TextureFormat::BC6_RGB: os << "BC6_RGB"; break;
        case TextureFormat::Count: os << "Count"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

enum class FilterMode
{
	Point = 0,
	Linear,
	Anisotropic,
	Count
};

enum class SamplerMode
{
	Wrap = 1,
	Mirror,
	Clamp,
	Border,
	MirrorOnce,
	Count
};

enum class DepthTest
{
	Never = 1,
	Less,
	Equal,
	LEqual,
	Greater,
	NotEqual,
	GEqual,
	Always,
};

enum class StencilTest
{
	Never = 1,
	Less,
	Equal,
	LEqual,
	Greater,
	NotEqual,
	GEqual,
	Always,
};

enum class StencilOp
{
	Keep = 1,
	Zero = 2,
	Replace = 3,
	IncrementClamp = 4,
	DecrementClamp = 5,
	Invert = 6,
	IncrementWrap = 7,
	DecrementWrap = 8
};

enum class BlendOperator
{
	Add = 1,
	Subtract,
	RevSubstract,
	Min,
	Max
};

enum class BlendFactor
{
	Zero = 1,
	One = 2,
	SrcColor = 3,
	InvSrcColor = 4,
	SrcAlpha = 5,
	InvSrcAlpha = 6,
	DestAlpha = 7,
	InvDestAlpha = 8,
	DestColor = 9,
	InvDestColor = 10
};

enum class CullMode
{
	None = 1,
	Front = 2,
	Back = 3,
};

enum class DrawPrimitive
{
	Line,
	Triangle
};

enum class CoopMatTier
{
	Other = 0,
	F16_to_F32_8_8_16,
	F16_to_F32_8_16_16,
	F16_to_F32_16_16_16,
	F16_to_F16_8_16_16,
	F16_to_F16_16_16_16,
	Count
};

struct VertexData
{
	float3 position;
	float3 normal;
	float3 tangent;
	float2 texCoord;
	uint32_t matID;
};