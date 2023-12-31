//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/BufferView.h>
#include <cuda/Light.h>


const unsigned int NUM_PAYLOAD_VALUES = 4u;


enum ObjectType
{
    PLANE_OBJECT  = 1,
    CUBE_OBJECT   = 1 << 1,
    VOLUME_OBJECT = 1 << 2,
    ANY_OBJECT = 0xFF,
};


enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};


struct LaunchParams
{
    unsigned int             width;
    unsigned int             height;
    unsigned int             subframe_index;
    float4*                  accum_buffer;
    uchar4*                  frame_buffer;
    int                      max_depth;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    BufferView<Light>        lights;
    float3                   miss_color;
    OptixTraversableHandle   handle;

    // Visbility masks
    unsigned int solid_objects;
    unsigned int volume_object;
};


struct MaterialData 
{
    struct Lambert
    {
        float3 base_color;
    };

    struct Volume
    {
        float  opacity; // effectively a scale factor for volume density
    };


    union
    {
        Lambert lambert;
        Volume  volume; 
    };
};


struct GeometryData
{
    struct Plane
    {
        float3 normal;
    };

    struct Volume
    {
        void* grid;
    };


    union
    {
        Plane  plane;
        Volume volume;
    };
};


struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};


struct PayloadRadiance
{
    float3 result;
    float  depth;
};


struct PayloadOcclusion
{
};
