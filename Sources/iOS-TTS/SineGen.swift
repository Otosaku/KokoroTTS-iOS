import Foundation
import CoreML
import Accelerate

/// Optimized sine wave generator for TTS with precomputed constants
class SineGen {
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

    // Precomputed constants
    private let twoPi: Float = 2.0 * Float.pi
    private let sineAmpDiv3: Float
    private let harmonicMultipliers: [Float]
    private var invSamplingRate: Float
    private var invUpsampleScale: Float
    private let halfUpsampleScale: Float = 0.5

    /// Initialize with optional deterministic mode for testing
    /// - Parameter deterministic: If true, disables random phase for reproducible results (default: false)
    init(deterministic: Bool = false) {
        self.useRandomPhase = !deterministic
        // Precompute constants
        self.sineAmpDiv3 = sineAmp / 3.0
        self.harmonicMultipliers = (1...dim).map { Float($0) }
        self.invSamplingRate = 1.0 / samplingRate
        self.invUpsampleScale = 1.0 / upsampleScale
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
        let radValues = try MLMultiArray(shape: f0Values.shape, dataType: .float32)
        let f0Pointer = f0Values.dataPointer.bindMemory(to: Float32.self, capacity: f0Values.count)
        let radPointer = radValues.dataPointer.bindMemory(to: Float32.self, capacity: radValues.count)
        
        // Vectorized normalization
        vDSP_vsmul(f0Pointer, 1, &invSamplingRate, radPointer, 1, vDSP_Length(f0Values.count))
        
        // Apply modulo 1
        for i in 0..<radValues.count {
            radPointer[i] = fmodf(radPointer[i], 1.0)
        }
        
        // Step 2: Add initial phase noise
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
        
        // Step 3: Downsample using optimized linear interpolation
        let downsampledLength = Int(Float(length) * invUpsampleScale)
        let downsampledRad = try downsampleLinearOptimized(radValues, targetLength: downsampledLength)
        
        // Step 4: Cumulative sum for phase
        let phase = try cumulativeSumOptimized(downsampledRad)
        
        // Scale by 2π using vectorized operations
        let phasePointer = phase.dataPointer.bindMemory(to: Float32.self, capacity: phase.count)
        var twoPiVar = twoPi
        vDSP_vsmul(phasePointer, 1, &twoPiVar, phasePointer, 1, vDSP_Length(phase.count))
        
        // Step 5: Upsample back to original length
        let scaledPhase = try upsampleLinearOptimized(phase, originalLength: length)
        
        // Step 6: Generate sine waves using Accelerate
        let sines = try MLMultiArray(shape: f0Values.shape, dataType: .float32)
        let sinesPointer = sines.dataPointer.bindMemory(to: Float32.self, capacity: sines.count)
        let scaledPhasePointer = scaledPhase.dataPointer.bindMemory(to: Float32.self, capacity: scaledPhase.count)
        
        // Vectorized sine calculation
        vvsinf(sinesPointer, scaledPhasePointer, [Int32(sines.count)])
        
        return sines
    }
    
    /// Optimized downsample using Accelerate
    private func downsampleLinearOptimized(_ array: MLMultiArray, targetLength: Int) throws -> MLMultiArray {
        let batchSize = array.shape[0].intValue
        let originalLength = array.shape[1].intValue
        let dim = array.shape[2].intValue
        
        let downsampled = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: targetLength), NSNumber(value: dim)], dataType: .float32)
        
        let scale = Float(originalLength) / Float(targetLength)
        let arrayPointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let downsampledPointer = downsampled.dataPointer.bindMemory(to: Float32.self, capacity: downsampled.count)
        
        // Process each batch and dimension
        for b in 0..<batchSize {
            for d in 0..<dim {
                // Create temporary arrays for this slice
                var slice = [Float](repeating: 0, count: originalLength)
                let startIdx = b * originalLength * dim + d
                
                // Extract slice with stride
                for i in 0..<originalLength {
                    slice[i] = arrayPointer[startIdx + i * dim]
                }
                
                // Perform interpolation for this slice
                for i in 0..<targetLength {
                    let srcIndex = (Float(i) + 0.5) * scale - 0.5
                    let srcIndexClamped = max(0, min(srcIndex, Float(originalLength - 1)))
                    
                    let lowerIndex = Int(floor(srcIndexClamped))
                    let upperIndex = min(lowerIndex + 1, originalLength - 1)
                    let fraction = srcIndexClamped - Float(lowerIndex)
                    
                    let interpolated = slice[lowerIndex] * (1.0 - fraction) + slice[upperIndex] * fraction
                    downsampledPointer[b * targetLength * dim + i * dim + d] = interpolated
                }
            }
        }
        
        return downsampled
    }
    
    /// Optimized upsample using Accelerate
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
        
        // Process each batch and dimension
        for b in 0..<batchSize {
            for d in 0..<dim {
                // Create temporary arrays for this slice
                var slice = [Float](repeating: 0, count: downsampledLength)
                let startIdx = b * downsampledLength * dim + d
                
                // Extract slice with stride
                for i in 0..<downsampledLength {
                    slice[i] = scaledPointer[startIdx + i * dim]
                }
                
                // Perform interpolation for this slice
                for t in 0..<originalLength {
                    let srcIndex = (Float(t) + 0.5) * scale - 0.5
                    let srcIndexClamped = max(0, min(srcIndex, Float(downsampledLength - 1)))
                    
                    let lowerIndex = Int(floor(srcIndexClamped))
                    let upperIndex = min(lowerIndex + 1, downsampledLength - 1)
                    let fraction = srcIndexClamped - Float(lowerIndex)
                    
                    let interpolated = slice[lowerIndex] * (1.0 - fraction) + slice[upperIndex] * fraction
                    upsampledPointer[b * originalLength * dim + t * dim + d] = interpolated
                }
            }
        }
        
        return upsampled
    }
    
    /// Optimized cumulative sum using Accelerate
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
        
        // Step 1: Generate harmonics using precomputed multipliers
        let fn = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: length), NSNumber(value: dim)], dataType: .float32)
        let f0Pointer = f0.dataPointer.bindMemory(to: Float32.self, capacity: f0.count)
        let fnPointer = fn.dataPointer.bindMemory(to: Float32.self, capacity: fn.count)
        
        // Optimized harmonic generation
        for b in 0..<batchSize {
            for t in 0..<length {
                let f0Value = f0Pointer[b * length + t]
                let baseIdx = b * length * dim + t * dim
                
                // Unroll small loop for better performance
                for h in 0..<dim {
                    fnPointer[baseIdx + h] = f0Value * harmonicMultipliers[h]
                }
            }
        }
        
        // Step 2: Generate sine waveforms
        let sineWaves = try f02sine(fn)
        
        // Step 3: Apply amplitude using vectorized operations
        let sinePointer = sineWaves.dataPointer.bindMemory(to: Float32.self, capacity: sineWaves.count)
        var ampVar = sineAmp
        vDSP_vsmul(sinePointer, 1, &ampVar, sinePointer, 1, vDSP_Length(sineWaves.count))
        
        // Step 4: Generate UV signal
        let uv = try f02uv(f0)
        let uvPointer = uv.dataPointer.bindMemory(to: Float32.self, capacity: uv.count)
        
        // Step 5: Apply UV and noise in single pass
        for b in 0..<batchSize {
            for t in 0..<length {
                let uvValue = uvPointer[b * length + t]
                let noiseAmp = uvValue * noiseStd + (1.0 - uvValue) * sineAmpDiv3
                let baseIdx = b * length * dim + t * dim
                
                for h in 0..<dim {
                    let idx = baseIdx + h
                    let sineValue = sinePointer[idx]
                    let noise: Float
                    if useRandomPhase {
                        noise = Float.random(in: -1...1) * noiseAmp
                    } else {
                        noise = 0.1 * noiseAmp
                    }
                    sinePointer[idx] = sineValue * uvValue + noise
                }
            }
        }
        
        if !useRandomPhase {
            sineWaves.saveDebug(name: "sineWaves_opt")
        }
        return sineWaves
    }
}
