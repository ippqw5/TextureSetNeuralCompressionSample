# Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors


This project show a demo about the neural block texture compression method described in the paper "Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors", presented at High-Performance Graphics (HPG) 2025.

This project does not include the training code. It only includes:
1. A model's: uncompressed textures , traditional BC6 textures , trained (4 BC1 latent textures + 1 MLP) 
2. Inference code that runs on GPU with DirectX 12 Cooperative Vectors.
3. PBR rendering code that uses the decoded features from the MLP or the uncompressed/BC6 textures.

I want to write training code to compress the provided uncompressed textures, then export the resulting latent textures and MLP as the same format as provided in this project. Then see the rendered image quality.

To achieve this, I take the following steps:
1. I copy the relevant sections from the paper here for easy reference.
2. I analyze the data layout of provided latent textures and MLP weights.
3. I implement the training code in PyTorch to reproduce the trained model.

## 1.Algorithm in Paper

From Paper:
```

Network Architecture:
a set of block compressed textures with different resolutions stores a latent representation. To evaluate a specific pixel at a given UV coordinate p(uv), we evaluate each latent texture ck = Tk(uv), concatenate all latent colors x = [c0 ... cK] and feed the resulting vector to a Multi- Layers Perceptron (MLP) to obtain a vector of features (albedo, normal, ...): f = f(x). Latent BC1 compression. Each latent texture Tk has its own res- olution (Wk ×Hk ) and is stored in BC1 format (see Khronos docu- mentation [Khr25] for details). Hence, each pixel is defined as the interpolation of two endpoints e0, e1 with a blending factor α: T(uv) = (1−α(uv)) e0(uv) +α(uv)e1(uv). With BC1 format, the endpoints are shared by groups of 4×4 texels while every texel has its own α. In practice, α is quantized using 2 bits and e0, e1 are quantized with a [5,6,5] bits pattern.

Training:
We apply a L1 loss between the decoded features and reference features sampled using a trilinear filtering. Gradients are passed through the MLP and each element of the BC1 textures are updated using the Adam optimizer (with lr = 10−3 and lr = 10−2 respec- tively). For each BC1 texture, we store its endpoints and alpha as floating point tensor. To decompress the unsigned quantized values (α¯, e¯0, and e¯1), we apply a sigmoid activation function to the float- ing point values before performing quantization aware training: α¯ = quant(sigmoid(α),[2]) (1) e¯0 = quant(sigmoid(α),[5,6,5]) (2) e¯1 = quant(sigmoid(α),[5,6,5]) (3) where quant(x,b) evaluate the quantized form of x using b bits but let the gradients of its floating point value pass through. We did not find evidence that optimizing first unquantized la- tent maps before quantizing during training was brining any gain in quality. Hence, we optimize our model with quantized representation from the start.
```

## 2.Data Layout of Provided Binaries
The project provides three binary files under `models/michel/`:
1. `uncompressed/` - folder containing uncompressed textures in custom binary format
2. `bc1_mip/` - folder containing the neural compressed textures and MLP weights
3. `bc6/` - folder containing traditional BC6 compressed textures (We won't analyze this here)

### uncompressed/

The textures used are located in `models/michel`. The folder contains three subfolders:
```
models/michel$ tree 
└── uncompressed 
    ├── [ 85M]  tex0.tex_bin
    ├── [ 85M]  tex1.tex_bin
    ├── [ 85M]  tex2.tex_bin
    ├── [ 85M]  tex3.tex_bin
    └── [ 85M]  tex4.tex_bin
```

From `sdk\src\render_pipeline\texture_manager.cpp:14`:
```cpp
Texture read_binary_texture_and_upload(GraphicsDevice device, CommandQueue cmdQ, CommandBuffer cmdB, const std::string& texFile)
{
	// Read from disk
	BinaryTexture binTex;
	binary_texture::import_binary_texture(texFile.c_str(), binTex);

	// Allocate the texture
	TextureDescriptor desc;
	desc.type = binTex.type;
	desc.width = binTex.width;
	desc.height = binTex.height;
	desc.depth = binTex.depth;
	desc.mipCount = binTex.mipCount;
	desc.format = binTex.format;
	Texture tex = graphics::resources::create_texture(device, desc);

	// Create the upload buffer
	GraphicsBuffer imageBuffer = graphics::resources::create_graphics_buffer(device, binTex.data.size(), sizeof(uint32_t), GraphicsBufferType::Upload);
	graphics::resources::set_buffer_data(imageBuffer, (const char*)binTex.data.data(), binTex.data.size());

	// Copy the buffer to a texture
	graphics::command_buffer::reset(cmdB);
	graphics::command_buffer::copy_buffer_into_texture_mips(cmdB, imageBuffer, 0, sizeof(uint32_t) * binTex.width * binTex.height, tex, 0);
	graphics::command_buffer::close(cmdB);
	graphics::command_queue::execute_command_buffer(cmdQ, cmdB);
	graphics::command_queue::flush(cmdQ);

	// Destroy the graphics buffer
	graphics::resources::destroy_graphics_buffer(imageBuffer);

	// return the texture
	return tex;
}
```

and `sdk\src\render_pipeline\texture_manager.cpp:112`:
```cpp
void TextureManager::upload_textures(CommandQueue cmdQ, CommandBuffer cmdB, const std::string& modelDir, const std::string& modelName)
{
	// Uncompressed textures
	{
		const std::string tex0Path = modelDir + "\\" + modelName + "\\uncompressed\\tex0.tex_bin";
		m_UncompressedSet.tex0 = read_binary_texture_and_upload(m_Device, cmdQ, cmdB, tex0Path);

		const std::string tex1Path = modelDir + "\\" + modelName + "\\uncompressed\\tex1.tex_bin";
		m_UncompressedSet.tex1 = read_binary_texture_and_upload(m_Device, cmdQ, cmdB, tex1Path);

		const std::string tex2Path = modelDir + "\\" + modelName + "\\uncompressed\\tex2.tex_bin";
		m_UncompressedSet.tex2 = read_binary_texture_and_upload(m_Device, cmdQ, cmdB, tex2Path);

		const std::string tex3Path = modelDir + "\\" + modelName + "\\uncompressed\\tex3.tex_bin";
		m_UncompressedSet.tex3 = read_binary_texture_and_upload(m_Device, cmdQ, cmdB, tex3Path);

		const std::string tex4Path = modelDir + "\\" + modelName + "\\uncompressed\\tex4.tex_bin";
		m_UncompressedSet.tex4 = read_binary_texture_and_upload(m_Device, cmdQ, cmdB, tex4Path);
	}
    ...
```

The program uses 5 tex_bin and the data layout seems to be:
```
OFFSET(in bytes)  | Description
------------------|-----------------------------
0                 | width (uint32)
4                 | height (uint32)
8                 | depth (uint32)
12                | mipCount (uint32)
16                | format (uint32, TextureFormat enum)
20                | type (uint32, TextureType enum)
24                | data.size() (uint32, number of bytes)
28                | start of texture data (uint8_t array)

Texture       | width | height | depth | mipCount | format | type |
--------------|-------|--------|-------|----------|--------|------|
tex0.tex_bin  | 4096  | 4096   | 1     | 13       | 9      | 2    |
tex1.tex_bin  | 4096  | 4096   | 1     | 13       | 9      | 2    |
tex2.tex_bin  | 4096  | 4096   | 1     | 13       | 9      | 2    |
tex3.tex_bin  | 4096  | 4096   | 1     | 13       | 9      | 2    |
tex4.tex_bin  | 4096  | 4096   | 1     | 13       | 9      | 2    |
```

Note: Format 9 = R8G8B8A8_UNorm, Type 2 = Tex2D.

5 uncompressed textures' Channel mapping(shaders/Material/Textures/MaterialPass.compute:87):
```hlsl
    // Sampler the textures
    float4 data0 = _Texture0.SampleGrad(s_texture_sampler, uv, uvDX, uvDY);
    float4 data1 = _Texture1.SampleGrad(s_texture_sampler, uv, uvDX, uvDY);
    float4 data2 = _Texture2.SampleGrad(s_texture_sampler, uv, uvDX, uvDY);
    float4 data3 = _Texture3.SampleGrad(s_texture_sampler, uv, uvDX, uvDY);
    float4 data4 = _Texture4.SampleGrad(s_texture_sampler, uv, uvDX, uvDY);

    // Fill the surface data
    SurfaceData surfaceData;
    surfaceData.baseColor = float3(data3.yz, data4.x);
    surfaceData.normalTS = float3(data2.yz, data3.x);
    surfaceData.ambientOcclusion = data2.x;
    surfaceData.perceptualRoughness = data1.z;
    surfaceData.metalness = data1.y;
    surfaceData.thickness = data0.x;
    surfaceData.mask = data0.yz;
```

### bc1_mip/
```
models/michel$ tree 
├── bc1_mip
│   ├── [ 23K]  mlp_0.bin   # MLP weights 
│   ├── [ 11M]  tex0_0.bc1  # Latent texture 0
│   ├── [ 11M]  tex1_0.bc1  # Latent texture 1
│   ├── [2.7M]  tex2_0.bc1  # Latent texture 2
│   └── [2.7M]  tex3_0.bc1  # Latent texture 3
```

In `sdk\src\network\tsnc.cpp:51`:
```cpp
void TSNC::reload_network(const std::string& modelDir, uint32_t numSets){
    ...

    // Copy all the sets
    for (uint32_t setIdx = 0; setIdx < numSets; ++setIdx)
    {
        // Read the file to a buffer
        std::vector<char> mlpBuffer;
        load_file_to_array((modelDir + "\\mlp_" + std::to_string(setIdx) +".bin").c_str(), mlpBuffer);

        // Unpack the MLP and adjust
        const char* rawData = (const char*)mlpBuffer.data();
        unpack_type(rawData, m_MLPArray[setIdx]);
        mlp::align_dimensions(m_MLPArray[setIdx]);

        // Load the textures
        m_TexData[4 * setIdx + 0].texBuffer = load_bc1_to_graphics_buffer(m_Device, (modelDir + "\\tex0_" + std::to_string(setIdx) + ".bc1").c_str(), m_TexData[4 * setIdx + 0].texSize, m_UVOffset[4 * setIdx + 0]);
        m_TexData[4 * setIdx + 1].texBuffer = load_bc1_to_graphics_buffer(m_Device, (modelDir + "\\tex1_" + std::to_string(setIdx) + ".bc1").c_str(), m_TexData[4 * setIdx + 1].texSize, m_UVOffset[4 * setIdx + 1]);
        m_TexData[4 * setIdx + 2].texBuffer = load_bc1_to_graphics_buffer(m_Device, (modelDir + "\\tex2_" + std::to_string(setIdx) + ".bc1").c_str(), m_TexData[4 * setIdx + 2].texSize, m_UVOffset[4 * setIdx + 2]);
        m_TexData[4 * setIdx + 3].texBuffer = load_bc1_to_graphics_buffer(m_Device, (modelDir + "\\tex3_" + std::to_string(setIdx) + ".bc1").c_str(), m_TexData[4 * setIdx + 3].texSize, m_UVOffset[4 * setIdx + 3]);
    }
    ...
}
```

#### Latent Texture Layout 
From `sdk\src\tools\texture_utils.cpp:19`:
```cpp
GraphicsBuffer load_bc1_to_graphics_buffer(GraphicsDevice device, const char* texturePath, uint3& dimensions, float2& uvOffset){
    ...
	// Read the sizes
	uint32_t* intArray = (uint32_t*)binaryFile.data();
	float* floatArray = (float*)binaryFile.data();
	dimensions.x = intArray[0] * 4;
	dimensions.y = intArray[1] * 4;
	dimensions.z = intArray[2];
	uvOffset.x = floatArray[3];
	uvOffset.y = floatArray[4];

	uint32_t sizesOffset = sizeof(uint32_t) * 5;
	dimensions.z = std::max(1, (int32_t)dimensions.z - 2);
	const uint32_t bufferSize = (uint32_t)binaryFile.size() - sizesOffset;

	// Create the buffer, upload to it and return it
	GraphicsBuffer textureBuffer = graphics::resources::create_graphics_buffer(device, bufferSize, 4, GraphicsBufferType::Upload);
	graphics::resources::set_buffer_data(textureBuffer, binaryFile.data() + sizesOffset, bufferSize);
	return textureBuffer;
}
```

The layout seems to be:
```
OFFSET(in bytes)  | Description
------------------|-----------------------------
0                 | width (uint32)
4                 | height (uint32)
8                 | z (uint32)
12                | uv.x offset (float)
16                | uv.y offset (float)
20                | start of BC1 format compressed data

Texture       | width   | height   | z | 
--------------|---------|----------|---|
tex0_0.bc1    |  1024   |  1024    | 13|
tex1_0.bc1    |  1024   |  1024    | 13|
tex2_0.bc1    |  512    |   512    | 12|
tex3_0.bc1    |  512    |   512    | 12|
```

I guess the `z` here represents the number of mip levels of original texture,
Because in BC1 format, each 4x4 block represents a 4x4 texel area, and in the code `dimensions.z = std::max(1, (int32_t)dimensions.z - 2);`. For example, `tex0_0.bc1`'s size is 1024x1024 which should only have 11 mip levels, so `z=13` here means original mip levels = 13 - 2 = 11.

#### MLP Layout

In `sdk\src\network\mlp.cpp:167`:
```cpp
void unpack_type(const char*& stream, CPUMLP& mlp)
{
    // MLP data
    unpack_bytes<uint32_t>(stream, mlp.nbMlp);
    unpack_bytes<uint32_t>(stream, mlp.finalChannelCount);
    unpack_bytes<uint32_t>(stream, mlp.finalBlockWidth);

    // MLP layer 0
    unpack_bytes<uint32_t>(stream, mlp.mlp0Width);
    unpack_bytes<uint32_t>(stream, mlp.mlp0Height);
    const uint32_t mlp0Size = mlp.mlp0Width * mlp.mlp0Height + mlp.mlp0Width;
    mlp.mlp0Buffer.resize(mlp0Size);
    unpack_buffer(stream, mlp0Size * sizeof(float), (char*)mlp.mlp0Buffer.data());

    // MLP layer 1
    unpack_bytes<uint32_t>(stream, mlp.mlp1Width);
    unpack_bytes<uint32_t>(stream, mlp.mlp1Height);
    const uint32_t mlp1Size = mlp.mlp1Width * mlp.mlp1Height + mlp.mlp1Width;
    mlp.mlp1Buffer.resize(mlp1Size);
    unpack_buffer(stream, mlp1Size * sizeof(float), (char*)mlp.mlp1Buffer.data());

    // MLP layer 2
    unpack_bytes<uint32_t>(stream, mlp.mlp2Width);
    unpack_bytes<uint32_t>(stream, mlp.mlp2Height);
    const uint32_t mlp2Size = mlp.mlp2Width * mlp.mlp2Height + mlp.mlp2Width;
    mlp.mlp2Buffer.resize(mlp2Size);
    unpack_buffer(stream, mlp2Size * sizeof(float), (char*)mlp.mlp2Buffer.data());

    // print metadata
	std::cout << "MLP info: " << std::endl;
	std::cout << "  nbMlp: " << mlp.nbMlp << std::endl;
	std::cout << "  finalChannelCount: " << mlp.finalChannelCount << std::endl;
	std::cout << "  finalBlockWidth: " << mlp.finalBlockWidth << std::endl;
	std::cout << "  MLP0: " << mlp.mlp0Width << " x " << mlp.mlp0Height << std::endl;
	std::cout << "  MLP1: " << mlp.mlp1Width << " x " << mlp.mlp1Height << std::endl;
	std::cout << "  MLP2: " << mlp.mlp2Width << " x " << mlp.mlp2Height << std::endl;
}
```

The first comes the MLP general info, then for each layer, it stores width, height, then weights + bias (weights: width * height, bias: width). 

For `mlp_0.bin` provided:
```
MLP info:
  nbMlp: 3
  finalChannelCount: 13
  finalBlockWidth: 1
  MLP0: 64 x 13
  MLP1: 64 x 64
  MLP2: 13 x 64
```

After alignment `sdk\src\network\tsnc.cpp:69`:
```cpp
        unpack_type(rawData, m_MLPArray[setIdx]);
69:     mlp::align_dimensions(m_MLPArray[setIdx]);


MLP0_IN_DIM: 16
MLP0_OUT_DIM: 64
MLP1_OUT_DIM: 64
MLP2_OUT_DIM: 16
```

`CPUMLP` is float32 weights and biases. When uploading to GPU, it converts to float16 `GPUMLP` see `sdk\src\network\mlp.cpp`. I gusses When training they use float32, but for inference on GPU they convert to float16 to save memory bandwidth.

MLP outputs' Channel Mapping(shaders\Material\MaterialPass.compute:128):
```hlsl
    // Fill the MLP's input
    #ifdef COOP_VECTOR_SUPPORTED
    vector<float16_t, MLP0_IN_DIM> infVector;
    #else
    float16_t infVector[16];
    #endif

#if !defined(LS_BC1_COMPRESSION)
    ...
#else
    sample_latent_space_bc1(infVector, uv, uvDX, uvDY, mat_id(v0)); // We are in this branch
#endif

    // Fill the rest with zeros
    infVector[12] = float16_t(compute_lod(uv, uvDX, uvDY));
    infVector[13] = float16_t(0.0);
    infVector[14] = float16_t(0.0);
    infVector[15] = float16_t(0.0);

    // Do the MLP Evaluation
    mlp_evaluation(infVector, matID);

    // Check the validity of the pixel
    if (!is_valid_visibility_value(visibilityData))
        return;

    // Fill the surface data
    SurfaceData surfaceData;
    surfaceData.baseColor = float3(infVector[DIFFUSE_OFFSET], infVector[DIFFUSE_OFFSET + 1], infVector[DIFFUSE_OFFSET + 2]);
    surfaceData.normalTS = float3(infVector[NORMAL_OFFSET], infVector[NORMAL_OFFSET + 1], infVector[NORMAL_OFFSET + 2]);
    surfaceData.ambientOcclusion = infVector[AO_OFFSET];
    surfaceData.perceptualRoughness = infVector[ROUGHNESS_OFFSET];
    surfaceData.metalness = infVector[METALNESS_OFFSET];
    surfaceData.thickness = infVector[THICKNESS_OFFSET];
    surfaceData.mask = float2(infVector[MASK_OFFSET], infVector[MASK_OFFSET + 1]);
```
---


## 3.Training

### Latent/MLP Structure for Training time
TODO: from above analysis, we can determine the latent structure and MLP structure for the uncompressed textures provided.

### Training Phase V.S Inference Phase
From the above and my understanding:
- Training Phase:
    - Latent = (e0_float, e1_float, alpha_float) per 4×4 block
    - Forward pass: `quant(sigmoid(latents))` → software BC1 decode → 12 channels → MLP
    - Backward pass: gradients flow back to the float latents (Quantization-Aware Training, STE)
    - Export the trained latents by quantizing them to BC1 format (This ensures training and inference see the exact same decoded values)

- Inference Phase
    - Latent = Exported BC1 binary (quantized e0, e1, alpha) per 4×4 block
    - GPU hardware BC1 decode → 12 channels → MLP (FP16)
    - The hardware decoder produces identical results to the software decoder used in training

