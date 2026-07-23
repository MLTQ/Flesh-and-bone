import Foundation

let fleshMetalSource = """
#include <metal_stdlib>
using namespace metal;

constant int HIDDEN = 32;

struct SimulationUniforms {
    uint cellCount;
    uint boneCount;
    uint frameCount;
    uint physicsEnabled;
    float phase;
    float phaseDelta;
    float motionIntensity;
    float fps;
    float dt;
    float pitch;
    float neighborScale;
    float accelerationCap;
};

inline float wrappedPhase(float value, uint frames) {
    float count = float(frames);
    value = fmod(value, count);
    return value < 0.0f ? value + count : value;
}

inline float3 skinPoint(
    uint cell,
    float phase,
    const device float4 *points,
    const device ushort *indices,
    const device float *weights,
    const device float4x4 *matrices,
    constant SimulationUniforms &u
) {
    float sample = wrappedPhase(phase, u.frameCount);
    uint lower = uint(floor(sample));
    uint upper = (lower + 1) % u.frameCount;
    float fraction = sample - float(lower);
    float4 result = float4(0.0f);
    float4 point = points[cell];
    uint influenceBase = cell * 8;
    for (uint lane = 0; lane < 6; ++lane) {
        float weight = weights[influenceBase + lane];
        if (weight <= 0.0f) continue;
        uint bone = uint(indices[influenceBase + lane]);
        float4x4 left = matrices[lower * u.boneCount + bone];
        float4x4 right = matrices[upper * u.boneCount + bone];
        result += weight * ((left * (1.0f - fraction) + right * fraction) * point);
    }
    return mix(point.xyz, result.xyz, u.motionIntensity);
}

kernel void skin_motion(
    const device float4 *points [[buffer(0)]],
    const device ushort *indices [[buffer(1)]],
    const device float *weights [[buffer(2)]],
    const device float4x4 *matrices [[buffer(3)]],
    device float4 *lbsPrevious [[buffer(4)]],
    device float4 *lbsCurrent [[buffer(5)]],
    device float4 *lbsNext [[buffer(6)]],
    constant SimulationUniforms &u [[buffer(7)]],
    uint cell [[thread_position_in_grid]]
) {
    if (cell >= u.cellCount) return;
    lbsPrevious[cell] = float4(skinPoint(
        cell, u.phase - u.phaseDelta, points, indices, weights, matrices, u), 0.0f);
    lbsCurrent[cell] = float4(skinPoint(
        cell, u.phase, points, indices, weights, matrices, u), 0.0f);
    lbsNext[cell] = float4(skinPoint(
        cell, u.phase + u.phaseDelta, points, indices, weights, matrices, u), 0.0f);
}

kernel void observe_neighbors(
    const device float4 *residual [[buffer(0)]],
    const device float4 *velocity [[buffer(1)]],
    const device float4 *lbsCurrent [[buffer(2)]],
    const device float4 *material [[buffer(3)]],
    const device int *neighbors [[buffer(4)]],
    device float4 *neighborResidual [[buffer(5)]],
    device float4 *neighborVelocity [[buffer(6)]],
    device float4 *compressionVector [[buffer(7)]],
    device float4 *stretchVector [[buffer(8)]],
    device float4 *densityScalars [[buffer(9)]],
    constant SimulationUniforms &u [[buffer(10)]],
    uint cell [[thread_position_in_grid]]
) {
    if (cell >= u.cellCount) return;
    float3 ownResidual = residual[cell].xyz;
    float3 ownVelocity = velocity[cell].xyz;
    float3 meanResidual = float3(0.0f);
    float3 meanVelocity = float3(0.0f);
    float3 compressionDirection = float3(0.0f);
    float3 stretchDirection = float3(0.0f);
    float signedCompression = 0.0f;
    float compressionSquared = 0.0f;
    float stretchSquared = 0.0f;
    float degree = 0.0f;
    uint base = cell * 8;
    for (uint lane = 0; lane < 8; ++lane) {
        int neighbor = neighbors[base + lane];
        if (neighbor < 0) continue;
        uint selected = uint(neighbor);
        degree += 1.0f;
        meanResidual += residual[selected].xyz - ownResidual;
        meanVelocity += velocity[selected].xyz - ownVelocity;
        float3 equilibrium = lbsCurrent[selected].xyz - lbsCurrent[cell].xyz;
        float lengthValue = length(equilibrium);
        float3 unit = equilibrium / max(lengthValue, 1.0e-12f);
        float denominator = max(lengthValue, 0.5f * u.pitch);
        float3 difference = residual[selected].xyz - ownResidual;
        float strain = clamp(dot(difference, unit) / denominator, -1.0f, 1.0f);
        float compression = max(-strain - 0.05f, 0.0f);
        float stretch = max(strain - 0.08f, 0.0f);
        signedCompression += -strain;
        compressionSquared += compression * compression;
        stretchSquared += stretch * stretch;
        compressionDirection += -compression * compression * unit;
        stretchDirection += stretch * stretch * unit;
    }
    float inverseDegree = 1.0f / max(degree, 1.0f);
    neighborResidual[cell] = float4(meanResidual * inverseDegree, 0.0f);
    neighborVelocity[cell] = float4(meanVelocity * inverseDegree, 0.0f);
    compressionVector[cell] = float4(compressionDirection * inverseDegree, 0.0f);
    stretchVector[cell] = float4(stretchDirection * inverseDegree, 0.0f);
    densityScalars[cell] = float4(
        signedCompression * inverseDegree,
        sqrt(compressionSquared * inverseDegree),
        sqrt(stretchSquared * inverseDegree),
        material[cell].x
    );
}

inline float silu(float value) {
    return value / (1.0f + exp(-value));
}

kernel void integrate_flesh(
    const device float4 *residualIn [[buffer(0)]],
    const device float4 *velocityIn [[buffer(1)]],
    const device float4 *lbsPrevious [[buffer(2)]],
    const device float4 *lbsCurrent [[buffer(3)]],
    const device float4 *lbsNext [[buffer(4)]],
    const device float4 *material [[buffer(5)]],
    const device float4 *neighborResidual [[buffer(6)]],
    const device float4 *neighborVelocity [[buffer(7)]],
    const device float4 *compressionVector [[buffer(8)]],
    const device float4 *stretchVector [[buffer(9)]],
    const device float4 *densityScalars [[buffer(10)]],
    const device float *model [[buffer(11)]],
    device float4 *residualOut [[buffer(12)]],
    device float4 *velocityOut [[buffer(13)]],
    constant SimulationUniforms &u [[buffer(14)]],
    uint cell [[thread_position_in_grid]]
) {
    if (cell >= u.cellCount) return;
    if (u.physicsEnabled == 0) {
        residualOut[cell] = float4(0.0f);
        velocityOut[cell] = float4(0.0f);
        return;
    }
    float3 residual = residualIn[cell].xyz;
    float3 velocity = velocityIn[cell].xyz;
    float stiffness = material[cell].y;
    float3 lbsAcceleration = (
        lbsNext[cell].xyz - 2.0f * lbsCurrent[cell].xyz + lbsPrevious[cell].xyz
    ) * (u.fps * u.fps);

    float3 acceleration =
        model[0] * (-stiffness * residual)
        + model[1] * (-sqrt(max(stiffness, 0.0f)) * velocity)
        + model[2] * u.neighborScale * neighborResidual[cell].xyz
        + model[3] * neighborVelocity[cell].xyz
        + model[4] * (-lbsAcceleration);

    float4 observation = densityScalars[cell];
    float input[5] = {
        clamp(observation.x, -1.0f, 1.0f),
        clamp(observation.y, 0.0f, 1.0f),
        clamp(observation.z, 0.0f, 1.0f),
        observation.w,
        clamp(length(velocity) / (u.pitch * u.fps), 0.0f, 4.0f)
    };
    const uint W1 = 7;
    const uint B1 = W1 + 32 * 5;
    const uint W2 = B1 + 32;
    const uint B2 = W2 + 32 * 32;
    const uint W3 = B2 + 32;
    const uint B3 = W3 + 2 * 32;
    float hidden1[HIDDEN];
    float hidden2[HIDDEN];
    for (uint row = 0; row < 32; ++row) {
        float value = model[B1 + row];
        for (uint column = 0; column < 5; ++column) {
            value += model[W1 + row * 5 + column] * input[column];
        }
        hidden1[row] = silu(value);
    }
    for (uint row = 0; row < 32; ++row) {
        float value = model[B2 + row];
        for (uint column = 0; column < 32; ++column) {
            value += model[W2 + row * 32 + column] * hidden1[column];
        }
        hidden2[row] = silu(value);
    }
    float coefficient[2];
    for (uint row = 0; row < 2; ++row) {
        float value = model[B3 + row];
        for (uint column = 0; column < 32; ++column) {
            value += model[W3 + row * 32 + column] * hidden2[column];
        }
        coefficient[row] = (1.0f / (1.0f + exp(-value))) * model[5 + row];
    }
    float3 densityAcceleration =
        coefficient[0] * compressionVector[cell].xyz
        + coefficient[1] * stretchVector[cell].xyz;
    float densityNorm = length(densityAcceleration);
    if (densityNorm > 1.0e-12f) {
        float capRatio = min(
            densityNorm / u.accelerationCap,
            10.0f
        );
        densityAcceleration *= (
            u.accelerationCap * tanh(capRatio)
            / densityNorm
        );
    } else {
        densityAcceleration = float3(0.0f);
    }
    if (u.physicsEnabled == 2) densityAcceleration = float3(0.0f);
    acceleration += densityAcceleration;
    float3 nextVelocity = velocity + u.dt * acceleration;
    float3 nextResidual = residual + u.dt * nextVelocity;
    velocityOut[cell] = float4(nextVelocity, 0.0f);
    residualOut[cell] = float4(nextResidual, 0.0f);
}

struct RenderUniforms {
    float4x4 viewProjection;
    float4 cameraRight;
    float4 cameraUp;
    float baseRadius;
    float radiusMultiplier;
    float opacity;
    uint cellCount;
};

struct VertexOut {
    float4 position [[position]];
    float2 local;
    float4 color;
    float opacity;
};

vertex VertexOut splat_vertex(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    const device float4 *lbsCurrent [[buffer(0)]],
    const device float4 *residual [[buffer(1)]],
    const device float4 *material [[buffer(2)]],
    const device uchar4 *colors [[buffer(3)]],
    const device uint *renderOrder [[buffer(4)]],
    constant RenderUniforms &u [[buffer(5)]]
) {
    const float2 corners[6] = {
        float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(1.0f, 1.0f),
        float2(-1.0f, -1.0f), float2(1.0f, 1.0f), float2(-1.0f, 1.0f)
    };
    uint cell = renderOrder[min(instanceID, u.cellCount - 1)];
    float2 corner = corners[vertexID];
    float radius = u.baseRadius * u.radiusMultiplier * material[cell].w * 2.7f;
    float3 center = lbsCurrent[cell].xyz + residual[cell].xyz;
    float3 world = center
        + u.cameraRight.xyz * corner.x * radius
        + u.cameraUp.xyz * corner.y * radius;
    VertexOut output;
    output.position = u.viewProjection * float4(world, 1.0f);
    output.local = corner * 2.7f;
    output.color = float4(colors[cell]) / 255.0f;
    output.opacity = u.opacity;
    return output;
}

fragment float4 splat_fragment(VertexOut input [[stage_in]]) {
    float alpha = input.opacity * exp(-0.5f * dot(input.local, input.local));
    if (alpha < 0.002f) discard_fragment();
    return float4(input.color.rgb * alpha, alpha);
}
"""
