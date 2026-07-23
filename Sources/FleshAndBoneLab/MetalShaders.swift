import Foundation

let fleshMetalSource = """
#include <metal_stdlib>
using namespace metal;

constant int HIDDEN = 32;

struct SimulationUniforms {
    uint cellCount;
    uint baseCellCount;
    uint boneCount;
    uint frameCount;
    uint physicsEnabled;
    uint layerCount;
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
    const device float4 *sourceAnchors [[buffer(1)]],
    const device ushort *indices [[buffer(2)]],
    const device float *weights [[buffer(3)]],
    const device float4x4 *matrices [[buffer(4)]],
    device float4 *lbsPrevious [[buffer(5)]],
    device float4 *lbsCurrent [[buffer(6)]],
    device float4 *lbsNext [[buffer(7)]],
    device float4 *lbsSource [[buffer(8)]],
    constant SimulationUniforms &u [[buffer(9)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= u.cellCount) return;
    uint cell = slot % u.baseCellCount;
    lbsPrevious[slot] = float4(skinPoint(
        cell, u.phase - u.phaseDelta, points, indices, weights, matrices, u), 0.0f);
    lbsCurrent[slot] = float4(skinPoint(
        cell, u.phase, points, indices, weights, matrices, u), 0.0f);
    lbsNext[slot] = float4(skinPoint(
        cell, u.phase + u.phaseDelta, points, indices, weights, matrices, u), 0.0f);
    lbsSource[slot] = float4(skinPoint(
        cell, u.phase, sourceAnchors, indices, weights, matrices, u), 0.0f);
}

inline float3 nicheAxis(uint cell) {
    uint hash = cell * 1664525u + 1013904223u;
    float3 axis = float3(
        float(hash & 1023u) / 511.5f - 1.0f,
        float((hash >> 10) & 1023u) / 511.5f - 1.0f,
        float((hash >> 20) & 1023u) / 511.5f - 1.0f
    );
    return axis / max(length(axis), 1.0e-6f);
}

kernel void observe_neighbors(
    const device float4 *residual [[buffer(0)]],
    const device float4 *velocity [[buffer(1)]],
    const device float4 *lbsCurrent [[buffer(2)]],
    const device float4 *material [[buffer(3)]],
    const device int *neighbors [[buffer(4)]],
    const device float *population [[buffer(5)]],
    device float4 *neighborResidual [[buffer(6)]],
    device float4 *neighborVelocity [[buffer(7)]],
    device float4 *compressionVector [[buffer(8)]],
    device float4 *stretchVector [[buffer(9)]],
    device float4 *densityScalars [[buffer(10)]],
    constant SimulationUniforms &u [[buffer(11)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= u.cellCount) return;
    uint cell = slot % u.baseCellCount;
    uint layer = slot / u.baseCellCount;
    if (population[slot] < 0.5f) {
        neighborResidual[slot] = float4(0.0f);
        neighborVelocity[slot] = float4(0.0f);
        compressionVector[slot] = float4(0.0f);
        stretchVector[slot] = float4(0.0f);
        densityScalars[slot] = float4(0.0f);
        return;
    }
    float3 ownResidual = residual[slot].xyz;
    float3 ownVelocity = velocity[slot].xyz;
    float3 meanResidual = float3(0.0f);
    float3 meanVelocity = float3(0.0f);
    float3 compressionDirection = float3(0.0f);
    float3 stretchDirection = float3(0.0f);
    float signedCompression = 0.0f;
    float compressionSquared = 0.0f;
    float stretchSquared = 0.0f;
    float graphDegree = 0.0f;
    float densityDegree = 0.0f;
    uint base = cell * 8;
    for (uint lane = 0; lane < 8; ++lane) {
        int neighbor = neighbors[base + lane];
        if (neighbor < 0) continue;
        uint selected = layer * u.baseCellCount + uint(neighbor);
        if (population[selected] < 0.5f) continue;
        graphDegree += 1.0f;
        densityDegree += 1.0f;
        meanResidual += residual[selected].xyz - ownResidual;
        meanVelocity += velocity[selected].xyz - ownVelocity;
        float3 equilibrium = lbsCurrent[selected].xyz - lbsCurrent[slot].xyz;
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
    if (u.layerCount == 2) {
        uint other = (1u - layer) * u.baseCellCount + cell;
        if (population[other] >= 0.5f) {
            float3 unit = nicheAxis(cell);
            if (layer != 0) unit = -unit;
            float desiredLength = 0.65f * u.pitch;
            float3 actual = (
                lbsCurrent[other].xyz + residual[other].xyz
            ) - (
                lbsCurrent[slot].xyz + ownResidual
            );
            float3 difference = actual - unit * desiredLength;
            float strain = clamp(
                dot(difference, unit) / max(desiredLength, 1.0e-6f),
                -1.0f,
                1.0f
            );
            float compression = max(-strain - 0.05f, 0.0f);
            float stretch = max(strain - 0.08f, 0.0f);
            densityDegree += 1.0f;
            signedCompression += -strain;
            compressionSquared += compression * compression;
            stretchSquared += stretch * stretch;
            compressionDirection += -compression * compression * unit;
            stretchDirection += stretch * stretch * unit;
        }
    }
    float inverseGraphDegree = 1.0f / max(graphDegree, 1.0f);
    float inverseDensityDegree = 1.0f / max(densityDegree, 1.0f);
    neighborResidual[slot] = float4(meanResidual * inverseGraphDegree, 0.0f);
    neighborVelocity[slot] = float4(meanVelocity * inverseGraphDegree, 0.0f);
    compressionVector[slot] = float4(
        compressionDirection * inverseDensityDegree, 0.0f);
    stretchVector[slot] = float4(
        stretchDirection * inverseDensityDegree, 0.0f);
    densityScalars[slot] = float4(
        signedCompression * inverseDensityDegree,
        sqrt(compressionSquared * inverseDensityDegree),
        sqrt(stretchSquared * inverseDensityDegree),
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
    const device float *population [[buffer(11)]],
    const device float *model [[buffer(12)]],
    device float4 *residualOut [[buffer(13)]],
    device float4 *velocityOut [[buffer(14)]],
    constant SimulationUniforms &u [[buffer(15)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= u.cellCount) return;
    uint cell = slot % u.baseCellCount;
    if (population[slot] < 0.5f) {
        residualOut[slot] = float4(0.0f);
        velocityOut[slot] = float4(0.0f);
        return;
    }
    if (u.physicsEnabled == 0) {
        residualOut[slot] = float4(0.0f);
        velocityOut[slot] = float4(0.0f);
        return;
    }
    float3 residual = residualIn[slot].xyz;
    float3 velocity = velocityIn[slot].xyz;
    float stiffness = material[cell].y;
    float3 lbsAcceleration = (
        lbsNext[slot].xyz - 2.0f * lbsCurrent[slot].xyz + lbsPrevious[slot].xyz
    ) * (u.fps * u.fps);

    float3 acceleration =
        model[0] * (-stiffness * residual)
        + model[1] * (-sqrt(max(stiffness, 0.0f)) * velocity)
        + model[2] * u.neighborScale * neighborResidual[slot].xyz
        + model[3] * neighborVelocity[slot].xyz
        + model[4] * (-lbsAcceleration);

    float4 observation = densityScalars[slot];
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
        coefficient[0] * compressionVector[slot].xyz
        + coefficient[1] * stretchVector[slot].xyz;
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
    velocityOut[slot] = float4(nextVelocity, 0.0f);
    residualOut[slot] = float4(nextResidual, 0.0f);
}

struct PopulationUniforms {
    uint cellCount;
    uint baseCellCount;
    uint changeCount;
    uint activate;
};

kernel void apply_population_change(
    const device uint *changeIndices [[buffer(0)]],
    device float *population [[buffer(1)]],
    const device float4 *lbsCurrent [[buffer(2)]],
    const device float4 *lbsSource [[buffer(3)]],
    device float4 *residualA [[buffer(4)]],
    device float4 *residualB [[buffer(5)]],
    device float4 *velocityA [[buffer(6)]],
    device float4 *velocityB [[buffer(7)]],
    constant PopulationUniforms &u [[buffer(8)]],
    uint offset [[thread_position_in_grid]]
) {
    if (offset >= u.changeCount) return;
    uint cell = changeIndices[offset];
    if (cell >= u.cellCount) return;
    float4 zero = float4(0.0f);
    float4 spawn = u.activate != 0
        ? float4(lbsSource[cell].xyz - lbsCurrent[cell].xyz, 0.0f)
        : zero;
    residualA[cell] = spawn;
    residualB[cell] = spawn;
    velocityA[cell] = zero;
    velocityB[cell] = zero;
    population[cell] = u.activate != 0 ? 1.0f : 0.0f;
}

struct RenderUniforms {
    float4x4 viewProjection;
    float4 cameraRight;
    float4 cameraUp;
    float baseRadius;
    float radiusMultiplier;
    float opacity;
    uint baseCellCount;
    uint renderCount;
    uint layerCount;
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
    const device float *population [[buffer(5)]],
    constant RenderUniforms &u [[buffer(6)]]
) {
    const float2 corners[6] = {
        float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(1.0f, 1.0f),
        float2(-1.0f, -1.0f), float2(1.0f, 1.0f), float2(-1.0f, 1.0f)
    };
    uint orderIndex = min(instanceID / u.layerCount, u.renderCount - 1);
    uint layer = instanceID % u.layerCount;
    uint cell = renderOrder[orderIndex];
    uint slot = layer * u.baseCellCount + cell;
    float2 corner = corners[vertexID];
    float radius = u.baseRadius * u.radiusMultiplier * material[cell].w * 2.7f;
    float3 center = lbsCurrent[slot].xyz + residual[slot].xyz;
    float3 world = center
        + u.cameraRight.xyz * corner.x * radius
        + u.cameraUp.xyz * corner.y * radius;
    VertexOut output;
    output.position = u.viewProjection * float4(world, 1.0f);
    output.local = corner * 2.7f;
    output.color = float4(colors[cell]) / 255.0f;
    output.opacity = u.opacity * population[slot];
    return output;
}

fragment float4 splat_fragment(VertexOut input [[stage_in]]) {
    float alpha = input.opacity * exp(-0.5f * dot(input.local, input.local));
    if (alpha < 0.002f) discard_fragment();
    return float4(input.color.rgb * alpha, alpha);
}
"""
