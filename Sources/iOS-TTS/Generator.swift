import Foundation
import CoreML

/// Generator model for converting decoder output to audio
public class Generator {
    private let generator: MLModel
    private let f0Upsample: MLModel
    private let sourceModule: MLModel
    private let sineGen: SineGenOptimized
    private let rosaStft: RosaKitSTFT

    /// Initialize generator with Core ML models
    /// - Parameters:
    ///   - generatorModel: The main generator model
    ///   - f0UpsampleModel: The F0 upsampling model
    ///   - sourceModuleModel: The source module model
    public init(generatorModel: MLModel, f0UpsampleModel: MLModel, sourceModuleModel: MLModel) {
        self.generator = generatorModel
        self.f0Upsample = f0UpsampleModel
        self.sourceModule = sourceModuleModel
        self.sineGen = SineGenOptimized()
        self.rosaStft = RosaKitSTFT(filterLength: 20, hopLength: 5, winLength: 20)
    }

    /// Initialize generator by loading models from path
    /// - Parameters:
    ///   - modelPath: Path to the directory containing model files
    ///   - configuration: Model configuration
    /// - Throws: Error if models cannot be loaded
    public convenience init(modelPath: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let generatorURL = modelPath.appendingPathComponent("Generator.mlmodelc")
        let f0UpsampleURL = modelPath.appendingPathComponent("F0Upsample.mlmodelc")
        let sourceModuleURL = modelPath.appendingPathComponent("SourceModuleHnNSF.mlmodelc")
        // Generator Core: CPU-only is fastest
        // ANE/GPU cause slowdown due to dilated convolutions and Snake activation
        let genConf = MLModelConfiguration()
        genConf.computeUnits = .cpuOnly
        let generatorModel = try MLModel(contentsOf: generatorURL, configuration: genConf)
        let f0UpsampleModel = try MLModel(contentsOf: f0UpsampleURL, configuration: configuration)
        let sourceModuleModel = try MLModel(contentsOf: sourceModuleURL, configuration: configuration)

        self.init(generatorModel: generatorModel, f0UpsampleModel: f0UpsampleModel, sourceModuleModel: sourceModuleModel)
    }

    /// Generate audio from decoder output
    /// - Parameters:
    ///   - x: Decoder output tensor
    ///   - s: Style vector
    ///   - f0Curve: F0 curve from decoder
    /// - Returns: Generated audio samples
    /// - Throws: Error if generation fails
    public func generate(x: MLMultiArray, s: MLMultiArray, f0Curve: MLMultiArray) throws -> [Float] {
        let monitor = PerformanceMonitor.shared

        // Step 1: Upsample F0
        let f0UpsampleOutput = try monitor.measure(PerformanceMonitor.Module.f0Upsample) {
            let f0Input = try reshapeF0ForUpsample(f0Curve)
            let f0UpsampleInput = try MLDictionaryFeatureProvider(dictionary: [
                "f0": MLFeatureValue(multiArray: f0Input)
            ])
            return try f0Upsample.prediction(from: f0UpsampleInput)
        }
        guard let f0Upsampled = f0UpsampleOutput.featureValue(for: "f0_out")?.multiArrayValue else {
            throw TTSError.predictionFailed("Failed to get F0 upsample output")
        }

        // Transpose f0 from [batch, 1, time] to [batch, time, 1]
        let f0Transposed = try transposeF0(f0Upsampled)

        // Step 2: Generate sine waves using SineGenOptimized
        let sineWaves = try monitor.measure(PerformanceMonitor.Module.sineGen) {
            return try sineGen.forward(f0Transposed)
        }

        // Step 3: Process through source module
        let sourceOutput = try monitor.measure(PerformanceMonitor.Module.sourceModule) {
            let sourceInput = try MLDictionaryFeatureProvider(dictionary: [
                "sine_wavs": MLFeatureValue(multiArray: sineWaves)
            ])
            return try sourceModule.prediction(from: sourceInput)
        }
        guard let sineMerge = sourceOutput.featureValue(for: "sine_merge")?.multiArrayValue else {
            throw TTSError.predictionFailed("Failed to get source module output")
        }

        // Apply transpose(1, 2).squeeze(1) equivalent operations
        let harSource = try transposeAndSqueeze(sineMerge)

        // Step 4: Apply STFT to get harmonics using RosaKit
        let (harSpec, harPhase) = try monitor.measure(PerformanceMonitor.Module.stft) {
            return try rosaStft.transform(harSource)
        }

        // Concatenate spec and phase: har = torch.cat([har_spec, har_phase], dim=1)
        let har = try concatenateSpectrograms(harSpec, harPhase)

        // Step 5: Generate through main generator
        let generatorOutput = try monitor.measure("Generator Core") {
            let generatorInput = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: x),
                "s": MLFeatureValue(multiArray: s),
                "har": MLFeatureValue(multiArray: har)
            ])
            return try generator.prediction(from: generatorInput)
        }
        guard let spec = generatorOutput.featureValue(for: "spec")?.multiArrayValue,
              let phase = generatorOutput.featureValue(for: "phase")?.multiArrayValue else {
            throw TTSError.predictionFailed("Failed to get generator output")
        }

        // Step 6: Apply inverse STFT to get audio
        let audio = try monitor.measure(PerformanceMonitor.Module.inverseSTFT) {
            return try rosaStft.inverse(spec, phase)
        }

        // Convert to 1D array
        let audioLength = audio.shape[2].intValue
        var audioArray = [Float](repeating: 0, count: audioLength)
        for i in 0..<audioLength {
            audioArray[i] = audio[[0, 0, i as NSNumber]].floatValue
        }

        return audioArray
    }

    // MARK: - Private Methods

    private func reshapeF0ForUpsample(_ f0Curve: MLMultiArray) throws -> MLMultiArray {
        // F0 curve shape: [1, sequence_length] -> [1, 1, sequence_length]
        let batchSize = 1
        let channels = 1
        let sequenceLength = f0Curve.shape[1].intValue

        let reshaped = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: channels), NSNumber(value: sequenceLength)], dataType: .float32)

        for i in 0..<sequenceLength {
            reshaped[[0, 0, i as NSNumber]] = f0Curve[[0, i as NSNumber]]
        }

        return reshaped
    }

    private func transposeF0(_ f0: MLMultiArray) throws -> MLMultiArray {
        // Transpose from [batch, 1, time] to [batch, time, 1]
        let batchSize = f0.shape[0].intValue
        let channels = f0.shape[1].intValue // Should be 1
        let time = f0.shape[2].intValue

        let transposed = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: time), NSNumber(value: channels)], dataType: .float32)

        for b in 0..<batchSize {
            for t in 0..<time {
                for c in 0..<channels {
                    transposed[[b as NSNumber, t as NSNumber, c as NSNumber]] = f0[[b as NSNumber, c as NSNumber, t as NSNumber]]
                }
            }
        }

        return transposed
    }

    private func concatenateSpectrograms(_ spec: MLMultiArray, _ phase: MLMultiArray) throws -> MLMultiArray {
        // Concatenate along channel dimension: [batch, freqBins, frames] + [batch, freqBins, frames] -> [batch, freqBins*2, frames]
        let batchSize = spec.shape[0].intValue
        let freqBins = spec.shape[1].intValue
        let frames = spec.shape[2].intValue

        let concatenated = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: freqBins * 2), NSNumber(value: frames)], dataType: .float32)

        // Copy spec to first half of channels
        for b in 0..<batchSize {
            for f in 0..<freqBins {
                for t in 0..<frames {
                    concatenated[[b as NSNumber, f as NSNumber, t as NSNumber]] = spec[[b as NSNumber, f as NSNumber, t as NSNumber]]
                }
            }
        }

        // Copy phase to second half of channels
        for b in 0..<batchSize {
            for f in 0..<freqBins {
                for t in 0..<frames {
                    concatenated[[b as NSNumber, (f + freqBins) as NSNumber, t as NSNumber]] = phase[[b as NSNumber, f as NSNumber, t as NSNumber]]
                }
            }
        }

        return concatenated
    }

    private func transposeAndSqueeze(_ input: MLMultiArray) throws -> MLMultiArray {
        // Apply transpose(1, 2).squeeze(1)
        // Input shape: [1, length, 1] -> transpose(1,2) -> [1, 1, length] -> squeeze(1) -> [length]
        // But we need to keep batch dimension for STFT, so result: [1, length]

        let length = input.shape[1].intValue
        let result = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float32)

        for i in 0..<length {
            result[[0, i as NSNumber]] = input[[0, i as NSNumber, 0]]
        }

        return result
    }

}
