import Foundation
import CoreML
import Accelerate

/// Optimized sine wave generator for TTS - experimental version for optimization
/// This is a copy of SineGen for testing optimizations without affecting the main pipeline
class SineGenOptimized {

    // Profiling data
    private var profilingEnabled = false
    private var profileData: [String: Double] = [:]
    private let sineAmp: Float = 0.1
    private let noiseStd: Float = 0.003
    private let harmonicNum: Int = 8
    private let dim: Int = 9 // harmonicNum + 1
    private let samplingRate: Float = 24000.0
    private let voicedThreshold: Float = 10.0
    private let upsampleScale: Float = 300.0

    /// Enable or disable phase randomization for debugging purposes
    /// Set to false to disable randomization (for deterministic testing)
    private let useRandomPhase: Bool

    /// Initialize with optional deterministic mode for testing
    /// - Parameter deterministic: If true, disables random phase for reproducible results
    init(deterministic: Bool = false) {
        self.useRandomPhase = !deterministic
        self.sineAmpDiv3 = sineAmp / 3.0
        self.harmonicMultipliers = (1...dim).map { Float($0) }
        self.invSamplingRate = 1.0 / samplingRate
        self.invUpsampleScale = 1.0 / upsampleScale
    }

    // Precomputed constants
    private let twoPi: Float = 2.0 * Float.pi
    private let sineAmpDiv3: Float
    private let harmonicMultipliers: [Float]
    private var invSamplingRate: Float
    private var invUpsampleScale: Float
    private let halfUpsampleScale: Float = 0.5

    // MARK: - Profiling

    func enableProfiling() {
        profilingEnabled = true
        profileData = [:]
    }

    func getProfilingReport() -> String? {
        guard profilingEnabled, !profileData.isEmpty else { return nil }

        var report = "=== SineGen Profiling ===\n"
        let sorted = profileData.sorted { $0.value > $1.value }
        let total = sorted.reduce(0) { $0 + $1.value }

        for (name, time) in sorted {
            let percent = (time / total) * 100
            report += String(format: "  %@: %.2f ms (%.1f%%)\n", name, time * 1000, percent)
        }
        report += String(format: "  TOTAL: %.2f ms\n", total * 1000)
        return report
    }

    private func profileStart() -> CFAbsoluteTime {
        return profilingEnabled ? CFAbsoluteTimeGetCurrent() : 0
    }

    private func profileEnd(_ start: CFAbsoluteTime, name: String) {
        guard profilingEnabled else { return }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        profileData[name, default: 0] += elapsed
    }

    /// Generate UV (unvoiced/voiced) signal
    private func f02uv(_ f0: MLMultiArray) throws -> MLMultiArray {
        let shape = f0.shape
        let uv = try MLMultiArray(shape: shape, dataType: .float32)

        // Use pointer-based access for better performance
        let f0Pointer = f0.dataPointer.bindMemory(to: Float32.self, capacity: f0.count)
        let uvPointer = uv.dataPointer.bindMemory(to: Float32.self, capacity: uv.count)

        // Vectorized threshold comparison
        var threshold = voicedThreshold
        var one: Float = 1.0
        var zero: Float = 0.0

        for i in 0..<f0.count {
            uvPointer[i] = f0Pointer[i] > threshold ? one : zero
        }

        return uv
    }

    /// Convert F0 to sine waves with optimizations
    private func f02sine(_ f0Values: MLMultiArray) throws -> MLMultiArray {
        let batchSize = f0Values.shape[0].intValue
        let length = f0Values.shape[1].intValue
        let harmonics = f0Values.shape[2].intValue

        // Step 1: Convert to radians (normalized by sampling rate)
        var t = profileStart()
        let radValues = try MLMultiArray(shape: f0Values.shape, dataType: .float32)
        let f0Pointer = f0Values.dataPointer.bindMemory(to: Float32.self, capacity: f0Values.count)
        let radPointer = radValues.dataPointer.bindMemory(to: Float32.self, capacity: radValues.count)

        // Vectorized normalization
        vDSP_vsmul(f0Pointer, 1, &invSamplingRate, radPointer, 1, vDSP_Length(f0Values.count))

        // Apply modulo 1 using vectorized: x - floor(x)
        var floored = [Float](repeating: 0, count: f0Values.count)
        vvfloorf(&floored, radPointer, [Int32(f0Values.count)])
        // radPointer = radPointer - floored (note: vDSP_vsub is B - A)
        vDSP_vsub(&floored, 1, radPointer, 1, radPointer, 1, vDSP_Length(f0Values.count))
        profileEnd(t, name: "1_radians+fmod")

        // Step 2: Add initial phase noise
        t = profileStart()
        if useRandomPhase {
            for b in 0..<batchSize {
                for h in 1..<harmonics { // Skip h=0 (fundamental)
                    let index = b * length * harmonics + h
                    radPointer[index] += Float.random(in: 0..<1)
                }
            }
        } else {
            // Static phase for debugging
            for b in 0..<batchSize {
                for h in 1..<harmonics { // Skip h=0 (fundamental)
                    let index = b * length * harmonics + h
                    radPointer[index] += 0.5
                }
            }
        }
        profileEnd(t, name: "2_phase_noise")

        // Step 3: Downsample using optimized linear interpolation
        t = profileStart()
        let downsampledLength = Int(Float(length) * invUpsampleScale)
        let downsampledRad = try downsampleLinearOptimized(radValues, targetLength: downsampledLength)
        profileEnd(t, name: "3_downsample")

        // Step 4: Cumulative sum for phase
        t = profileStart()
        let phase = try cumulativeSumOptimized(downsampledRad)

        // Scale by 2π using vectorized operations
        let phasePointer = phase.dataPointer.bindMemory(to: Float32.self, capacity: phase.count)
        var twoPiVar = twoPi
        vDSP_vsmul(phasePointer, 1, &twoPiVar, phasePointer, 1, vDSP_Length(phase.count))
        profileEnd(t, name: "4_cumsum+scale")

        // Step 5: Upsample back to original length
        t = profileStart()
        let scaledPhase = try upsampleLinearOptimized(phase, originalLength: length)
        profileEnd(t, name: "5_upsample")

        // Step 6: Generate sine waves using Accelerate
        t = profileStart()
        let sines = try MLMultiArray(shape: f0Values.shape, dataType: .float32)
        let sinesPointer = sines.dataPointer.bindMemory(to: Float32.self, capacity: sines.count)
        let scaledPhasePointer = scaledPhase.dataPointer.bindMemory(to: Float32.self, capacity: scaledPhase.count)

        // Vectorized sine calculation
        vvsinf(sinesPointer, scaledPhasePointer, [Int32(sines.count)])
        profileEnd(t, name: "6_vvsinf")

        return sines
    }

    /// Downsample using SIMD - process all 9 dims at once per time step
    private func downsampleLinearOptimized(_ array: MLMultiArray, targetLength: Int) throws -> MLMultiArray {
        let batchSize = array.shape[0].intValue
        let originalLength = array.shape[1].intValue
        let dim = array.shape[2].intValue

        let downsampled = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: targetLength), NSNumber(value: dim)], dataType: .float32)

        let scale = Float(originalLength) / Float(targetLength)
        let arrayPointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let downsampledPointer = downsampled.dataPointer.bindMemory(to: Float32.self, capacity: downsampled.count)

        // Process all dims at once using SIMD
        for b in 0..<batchSize {
            let srcBase = b * originalLength * dim
            let dstBase = b * targetLength * dim

            for i in 0..<targetLength {
                let srcIndex = (Float(i) + 0.5) * scale - 0.5
                let srcIndexClamped = max(0, min(srcIndex, Float(originalLength - 1)))

                let lowerIndex = Int(floor(srcIndexClamped))
                let upperIndex = min(lowerIndex + 1, originalLength - 1)
                let fraction = srcIndexClamped - Float(lowerIndex)
                let oneMinusFrac = 1.0 - fraction

                let lowerBase = srcBase + lowerIndex * dim
                let upperBase = srcBase + upperIndex * dim
                let dstIdx = dstBase + i * dim

                // Load 8 values from lower and upper positions
                let lowerVec = SIMD8<Float>(
                    arrayPointer[lowerBase], arrayPointer[lowerBase + 1],
                    arrayPointer[lowerBase + 2], arrayPointer[lowerBase + 3],
                    arrayPointer[lowerBase + 4], arrayPointer[lowerBase + 5],
                    arrayPointer[lowerBase + 6], arrayPointer[lowerBase + 7]
                )
                let upperVec = SIMD8<Float>(
                    arrayPointer[upperBase], arrayPointer[upperBase + 1],
                    arrayPointer[upperBase + 2], arrayPointer[upperBase + 3],
                    arrayPointer[upperBase + 4], arrayPointer[upperBase + 5],
                    arrayPointer[upperBase + 6], arrayPointer[upperBase + 7]
                )

                // Interpolate: lower * (1-frac) + upper * frac
                let resultVec = lowerVec * oneMinusFrac + upperVec * fraction

                // Store 8 results
                downsampledPointer[dstIdx] = resultVec[0]
                downsampledPointer[dstIdx + 1] = resultVec[1]
                downsampledPointer[dstIdx + 2] = resultVec[2]
                downsampledPointer[dstIdx + 3] = resultVec[3]
                downsampledPointer[dstIdx + 4] = resultVec[4]
                downsampledPointer[dstIdx + 5] = resultVec[5]
                downsampledPointer[dstIdx + 6] = resultVec[6]
                downsampledPointer[dstIdx + 7] = resultVec[7]

                // Handle 9th dim
                let lower9 = arrayPointer[lowerBase + 8]
                let upper9 = arrayPointer[upperBase + 8]
                downsampledPointer[dstIdx + 8] = lower9 * oneMinusFrac + upper9 * fraction
            }
        }

        return downsampled
    }

    /// Upsample using SIMD - process all 9 dims at once per time step
    private func upsampleLinearOptimized(_ array: MLMultiArray, originalLength: Int) throws -> MLMultiArray {
        let batchSize = array.shape[0].intValue
        let downsampledLength = array.shape[1].intValue
        let dim = array.shape[2].intValue

        // First scale by upsample_scale
        let scaled = try MLMultiArray(shape: array.shape, dataType: .float32)
        let arrayPointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let scaledPointer = scaled.dataPointer.bindMemory(to: Float32.self, capacity: scaled.count)

        // Vectorized scaling
        var scaleVar = upsampleScale
        vDSP_vsmul(arrayPointer, 1, &scaleVar, scaledPointer, 1, vDSP_Length(array.count))

        // Then upsample to original length
        let upsampled = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: originalLength), NSNumber(value: dim)], dataType: .float32)
        let upsampledPointer = upsampled.dataPointer.bindMemory(to: Float32.self, capacity: upsampled.count)

        let scale = Float(downsampledLength) / Float(originalLength)

        // Process all dims at once using SIMD
        for b in 0..<batchSize {
            let srcBase = b * downsampledLength * dim
            let dstBase = b * originalLength * dim

            for t in 0..<originalLength {
                let srcIndex = (Float(t) + 0.5) * scale - 0.5
                let srcIndexClamped = max(0, min(srcIndex, Float(downsampledLength - 1)))

                let lowerIndex = Int(floor(srcIndexClamped))
                let upperIndex = min(lowerIndex + 1, downsampledLength - 1)
                let fraction = srcIndexClamped - Float(lowerIndex)
                let oneMinusFrac = 1.0 - fraction

                let lowerBase = srcBase + lowerIndex * dim
                let upperBase = srcBase + upperIndex * dim
                let dstIdx = dstBase + t * dim

                // Load 8 values from lower and upper positions
                let lowerVec = SIMD8<Float>(
                    scaledPointer[lowerBase], scaledPointer[lowerBase + 1],
                    scaledPointer[lowerBase + 2], scaledPointer[lowerBase + 3],
                    scaledPointer[lowerBase + 4], scaledPointer[lowerBase + 5],
                    scaledPointer[lowerBase + 6], scaledPointer[lowerBase + 7]
                )
                let upperVec = SIMD8<Float>(
                    scaledPointer[upperBase], scaledPointer[upperBase + 1],
                    scaledPointer[upperBase + 2], scaledPointer[upperBase + 3],
                    scaledPointer[upperBase + 4], scaledPointer[upperBase + 5],
                    scaledPointer[upperBase + 6], scaledPointer[upperBase + 7]
                )

                // Interpolate: lower * (1-frac) + upper * frac
                let resultVec = lowerVec * oneMinusFrac + upperVec * fraction

                // Store 8 results
                upsampledPointer[dstIdx] = resultVec[0]
                upsampledPointer[dstIdx + 1] = resultVec[1]
                upsampledPointer[dstIdx + 2] = resultVec[2]
                upsampledPointer[dstIdx + 3] = resultVec[3]
                upsampledPointer[dstIdx + 4] = resultVec[4]
                upsampledPointer[dstIdx + 5] = resultVec[5]
                upsampledPointer[dstIdx + 6] = resultVec[6]
                upsampledPointer[dstIdx + 7] = resultVec[7]

                // Handle 9th dim
                let lower9 = scaledPointer[lowerBase + 8]
                let upper9 = scaledPointer[upperBase + 8]
                upsampledPointer[dstIdx + 8] = lower9 * oneMinusFrac + upper9 * fraction
            }
        }

        return upsampled
    }

    /// Cumulative sum - using manual implementation (same as original)
    /// vDSP_vrsum has different semantics, keeping manual for correctness
    private func cumulativeSumOptimized(_ array: MLMultiArray) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: array.shape, dataType: .float32)
        let batchSize = array.shape[0].intValue
        let length = array.shape[1].intValue
        let dim = array.shape[2].intValue

        let arrayPointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let resultPointer = result.dataPointer.bindMemory(to: Float32.self, capacity: result.count)

        // Process each batch and dimension
        for b in 0..<batchSize {
            for d in 0..<dim {
                let startIdx = b * length * dim + d
                var sum: Float = 0

                for t in 0..<length {
                    let idx = startIdx + t * dim
                    sum += arrayPointer[idx]
                    resultPointer[idx] = sum
                }
            }
        }

        return result
    }

    /// Main forward function with optimizations
    func forward(_ f0: MLMultiArray) throws -> MLMultiArray {
        let batchSize = f0.shape[0].intValue
        let length = f0.shape[1].intValue

        // Step 1: Generate harmonics using SIMD
        var t = profileStart()
        let fn = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: length), NSNumber(value: dim)], dataType: .float32)
        let f0Pointer = f0.dataPointer.bindMemory(to: Float32.self, capacity: f0.count)
        let fnPointer = fn.dataPointer.bindMemory(to: Float32.self, capacity: fn.count)

        // Preload harmonic multipliers as SIMD8 + scalar
        let harmonicVec = SIMD8<Float>(
            harmonicMultipliers[0], harmonicMultipliers[1],
            harmonicMultipliers[2], harmonicMultipliers[3],
            harmonicMultipliers[4], harmonicMultipliers[5],
            harmonicMultipliers[6], harmonicMultipliers[7]
        )
        let harmonic9 = harmonicMultipliers[8]

        for b in 0..<batchSize {
            for ti in 0..<length {
                let f0Value = f0Pointer[b * length + ti]
                let baseIdx = b * length * dim + ti * dim

                // Compute 8 harmonics at once
                let resultVec = harmonicVec * f0Value

                // Store results
                fnPointer[baseIdx] = resultVec[0]
                fnPointer[baseIdx + 1] = resultVec[1]
                fnPointer[baseIdx + 2] = resultVec[2]
                fnPointer[baseIdx + 3] = resultVec[3]
                fnPointer[baseIdx + 4] = resultVec[4]
                fnPointer[baseIdx + 5] = resultVec[5]
                fnPointer[baseIdx + 6] = resultVec[6]
                fnPointer[baseIdx + 7] = resultVec[7]
                fnPointer[baseIdx + 8] = f0Value * harmonic9
            }
        }
        profileEnd(t, name: "fwd_1_harmonics")

        // Step 2: Generate sine waveforms (includes its own profiling)
        let sineWaves = try f02sine(fn)

        // Step 3: Apply amplitude using vectorized operations
        t = profileStart()
        let sinePointer = sineWaves.dataPointer.bindMemory(to: Float32.self, capacity: sineWaves.count)
        var ampVar = sineAmp
        vDSP_vsmul(sinePointer, 1, &ampVar, sinePointer, 1, vDSP_Length(sineWaves.count))
        profileEnd(t, name: "fwd_2_amplitude")

        // Step 4: Generate UV signal
        t = profileStart()
        let uv = try f02uv(f0)
        let uvPointer = uv.dataPointer.bindMemory(to: Float32.self, capacity: uv.count)
        profileEnd(t, name: "fwd_3_uv")

        // Step 5: Apply UV and noise - SIMD optimized with fast random
        t = profileStart()

        let totalElements = batchSize * length * dim

        if useRandomPhase {
            // Pre-generate all random numbers at once using arc4random_buf
            // This is MUCH faster than calling Float.random() 432000 times
            var randomInts = [UInt32](repeating: 0, count: totalElements)
            arc4random_buf(&randomInts, totalElements * MemoryLayout<UInt32>.size)

            // Convert UInt32 to Float using vectorized operations
            // Combined formula: (uint / max) * 2 - 1 = uint * (2/max) - 1
            var randomFloats = [Float](repeating: 0, count: totalElements)

            // Convert UInt32 to Float
            vDSP_vfltu32(&randomInts, 1, &randomFloats, 1, vDSP_Length(totalElements))

            // Scale and shift in one pass using unsafe pointer to allow in-place
            var scale = 2.0 / Float(UInt32.max)
            var offset: Float = -1.0
            randomFloats.withUnsafeMutableBufferPointer { buffer in
                vDSP_vsmsa(buffer.baseAddress!, 1, &scale, &offset, buffer.baseAddress!, 1, vDSP_Length(totalElements))
            }

            // Process using SIMD8
            for b in 0..<batchSize {
                for ti in 0..<length {
                    let uvValue = uvPointer[b * length + ti]
                    let noiseAmp = uvValue * noiseStd + (1.0 - uvValue) * sineAmpDiv3
                    let baseIdx = b * length * dim + ti * dim

                    // Load 8 sine values as SIMD8
                    let sineVec = SIMD8<Float>(
                        sinePointer[baseIdx], sinePointer[baseIdx + 1],
                        sinePointer[baseIdx + 2], sinePointer[baseIdx + 3],
                        sinePointer[baseIdx + 4], sinePointer[baseIdx + 5],
                        sinePointer[baseIdx + 6], sinePointer[baseIdx + 7]
                    )

                    // Load pre-generated random values
                    let noiseVec = SIMD8<Float>(
                        randomFloats[baseIdx], randomFloats[baseIdx + 1],
                        randomFloats[baseIdx + 2], randomFloats[baseIdx + 3],
                        randomFloats[baseIdx + 4], randomFloats[baseIdx + 5],
                        randomFloats[baseIdx + 6], randomFloats[baseIdx + 7]
                    ) * noiseAmp

                    let resultVec = sineVec * uvValue + noiseVec

                    // Store back
                    sinePointer[baseIdx] = resultVec[0]
                    sinePointer[baseIdx + 1] = resultVec[1]
                    sinePointer[baseIdx + 2] = resultVec[2]
                    sinePointer[baseIdx + 3] = resultVec[3]
                    sinePointer[baseIdx + 4] = resultVec[4]
                    sinePointer[baseIdx + 5] = resultVec[5]
                    sinePointer[baseIdx + 6] = resultVec[6]
                    sinePointer[baseIdx + 7] = resultVec[7]

                    // Handle 9th harmonic
                    sinePointer[baseIdx + 8] = sinePointer[baseIdx + 8] * uvValue + randomFloats[baseIdx + 8] * noiseAmp
                }
            }
        } else {
            // Deterministic mode - no random needed
            for b in 0..<batchSize {
                for ti in 0..<length {
                    let uvValue = uvPointer[b * length + ti]
                    let noiseAmp = uvValue * noiseStd + (1.0 - uvValue) * sineAmpDiv3
                    let baseIdx = b * length * dim + ti * dim

                    let sineVec = SIMD8<Float>(
                        sinePointer[baseIdx], sinePointer[baseIdx + 1],
                        sinePointer[baseIdx + 2], sinePointer[baseIdx + 3],
                        sinePointer[baseIdx + 4], sinePointer[baseIdx + 5],
                        sinePointer[baseIdx + 6], sinePointer[baseIdx + 7]
                    )

                    let noiseVec = SIMD8<Float>(repeating: 0.1 * noiseAmp)
                    let resultVec = sineVec * uvValue + noiseVec

                    sinePointer[baseIdx] = resultVec[0]
                    sinePointer[baseIdx + 1] = resultVec[1]
                    sinePointer[baseIdx + 2] = resultVec[2]
                    sinePointer[baseIdx + 3] = resultVec[3]
                    sinePointer[baseIdx + 4] = resultVec[4]
                    sinePointer[baseIdx + 5] = resultVec[5]
                    sinePointer[baseIdx + 6] = resultVec[6]
                    sinePointer[baseIdx + 7] = resultVec[7]

                    sinePointer[baseIdx + 8] = sinePointer[baseIdx + 8] * uvValue + 0.1 * noiseAmp
                }
            }
        }
        profileEnd(t, name: "fwd_4_uv_noise")

        return sineWaves
    }
}
