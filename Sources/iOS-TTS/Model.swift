//
//  Model.swift
//  iOS-TTS
//
//  Created by Marat Zainullin on 10/07/2025.
//

import Foundation
import CoreML
import Accelerate

public class TTSModel {
    private let bert: MLModel
    private let bertEncoder: MLModel
    private let durationEncoder: MLModel
    private let prosodyPredictor: MLModel
    private let f0Predictor: MLModel
    private let textEncoder: MLModel
    private let decoder: MLModel
    private let generator: Generator

    private let configuration: MLModelConfiguration

    public init(modelPath: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        self.configuration = configuration

        // Load BERT model
        let bertURL = modelPath.appendingPathComponent("Albert.mlmodelc")
        self.bert = try MLModel(contentsOf: bertURL, configuration: configuration)

        // Load BERT Encoder
        let bertEncoderURL = modelPath.appendingPathComponent("BertEncoder.mlmodelc")
        self.bertEncoder = try MLModel(contentsOf: bertEncoderURL, configuration: configuration)

        // Load Duration Encoder
        let durationEncoderURL = modelPath.appendingPathComponent("DurationEncoder.mlmodelc")
        self.durationEncoder = try MLModel(contentsOf: durationEncoderURL, configuration: configuration)

        // Load Prosody Predictor
        let prosodyPredictorURL = modelPath.appendingPathComponent("ProsodyPredictor.mlmodelc")
        self.prosodyPredictor = try MLModel(contentsOf: prosodyPredictorURL, configuration: configuration)

        // Load F0 Predictor
        let f0PredictorURL = modelPath.appendingPathComponent("F0Predictor.mlmodelc")
        self.f0Predictor = try MLModel(contentsOf: f0PredictorURL, configuration: configuration)

        // Load Text Encoder
        let textEncoderURL = modelPath.appendingPathComponent("TextEncoder.mlmodelc")
        self.textEncoder = try MLModel(contentsOf: textEncoderURL, configuration: configuration)

        // Load Decoder
        let decoderURL = modelPath.appendingPathComponent("Decoder.mlmodelc")
        self.decoder = try MLModel(contentsOf: decoderURL, configuration: configuration)

        // Load Generator
        self.generator = try Generator(modelPath: modelPath, configuration: configuration)
    }

    func infer(inputIds: [Int], refS: [Float], speed: Float) throws -> [Float] {
        let monitor = PerformanceMonitor.shared

        return try monitor.measure(PerformanceMonitor.Module.total) {
            // Batch size is always 1
            let seqLen = inputIds.count

            // Create attention mask (all 1s since we have real tokens, no padding)
            let attentionMask = Array(repeating: 1, count: seqLen)

            // Create text mask for other models (0s for real tokens, 1s for padding)
            // Since we have no padding, all values are 0
            let textMask = Array(repeating: Float(0), count: seqLen)

            // Prepare BERT inputs
            let inputIdsArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .float32)
            let attentionMaskArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .float32)

            // Convert input_ids to Float16
            for i in 0..<seqLen {
                inputIdsArray[[0, i as NSNumber]] = NSNumber(value: Float(inputIds[i]))
                attentionMaskArray[[0, i as NSNumber]] = NSNumber(value: Float(attentionMask[i]))
            }

            // Call BERT model
            let bertInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                "attention_mask": MLFeatureValue(multiArray: attentionMaskArray)
            ])

            let bertOutput = try monitor.measure(PerformanceMonitor.Module.bert) {
                return try bert.prediction(from: bertInput)
            }
            guard let lastHiddenState = bertOutput.featureValue(for: "last_hidden_state")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get BERT output"])
            }

            // Call BERT encoder
            let bertEncoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "bert_dur": MLFeatureValue(multiArray: lastHiddenState)
            ])

            let bertEncoderOutput = try monitor.measure(PerformanceMonitor.Module.bertEncoder) {
                return try bertEncoder.prediction(from: bertEncoderInput)
            }
            guard let dEnRaw = bertEncoderOutput.featureValue(for: "d_en")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to get BERT encoder output"])
            }

            // In Python: d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            // Transpose last two dimensions
            let dEn = try transposeLastTwoDimensions(dEnRaw)

            // Split style vector
            let (refAudio, style) = splitStyleVector(refS)

            // Prepare text mask for prosody predictor
            let textMaskArray = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .float32)
            for i in 0..<seqLen {
                textMaskArray[[0, i as NSNumber]] = NSNumber(value: textMask[i])
            }

            // Prepare style array
            let styleArray = try MLMultiArray(shape: [1, 128], dataType: .float32)
            for i in 0..<128 {
                styleArray[[0, i as NSNumber]] = NSNumber(value: style[i])
            }

            // Call Duration Encoder (without speed)
            let durationInput = try MLDictionaryFeatureProvider(dictionary: [
                "text": MLFeatureValue(multiArray: dEn),
                "style": MLFeatureValue(multiArray: styleArray),
                "mask": MLFeatureValue(multiArray: textMaskArray),
            ])

            let durationOutput = try monitor.measure(PerformanceMonitor.Module.durationEncoder) {
                return try durationEncoder.prediction(from: durationInput)
            }
            guard let d = durationOutput.featureValue(for: "d")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get duration predictor output"])
            }

            // Prepare speed array (tensor of size (1,))
            let speedArray = try MLMultiArray(shape: [1], dataType: .float32)
            speedArray[[0]] = NSNumber(value: speed)

            let prosodyInput = try MLDictionaryFeatureProvider(dictionary: [
                "d": MLFeatureValue(multiArray: d),
                "speed": MLFeatureValue(multiArray: speedArray)
            ])

            let prosodyOutput = try monitor.measure(PerformanceMonitor.Module.prosodyPredictor) {
                return try prosodyPredictor.prediction(from: prosodyInput)
            }

            guard let predDur = prosodyOutput.featureValue(for: "pred_dur")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get prosody predictor output"])
            }

            // Create alignment indices
            let (_, alignmentMatrix, en) = try monitor.measure(PerformanceMonitor.Module.alignment) {
                let indices = try createAlignmentIndices(predDur: predDur, seqLen: seqLen)
                let alignmentMatrix = try createAlignmentMatrix(indices: indices, seqLen: seqLen)
                let en = try applyAlignment(d: d, alignmentMatrix: alignmentMatrix)
                return (indices, alignmentMatrix, en)
            }

            // Call F0 Predictor
            let f0Input = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: en),
                "s": MLFeatureValue(multiArray: styleArray),
            ])

            let f0Output = try monitor.measure(PerformanceMonitor.Module.f0Predictor) {
                return try f0Predictor.prediction(from: f0Input)
            }
            guard let f0Pred = f0Output.featureValue(for: "F0_pred")?.multiArrayValue,
                  let nPred = f0Output.featureValue(for: "N_pred")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to get F0 predictor output"])
            }

            // Call Text Encoder
            let textEncoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: inputIdsArray),
                "m": MLFeatureValue(multiArray: textMaskArray),
            ])

            let textEncoderOutput = try monitor.measure(PerformanceMonitor.Module.textEncoder) {
                return try textEncoder.prediction(from: textEncoderInput)
            }
            guard let tEn = textEncoderOutput.featureValue(for: "t_en")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to get text encoder output"])
            }

            // Apply alignment to text encoder output
            let asr = try applyAlignmentDirect(input: tEn, alignmentMatrix: alignmentMatrix)

            // Prepare reference audio array
            let refAudioArray = try MLMultiArray(shape: [1, 128], dataType: .float32)
            for i in 0..<128 {
                refAudioArray[[0, i as NSNumber]] = NSNumber(value: refAudio[i])
            }

            // Call Decoder
            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "asr": MLFeatureValue(multiArray: asr),
                "F0_curve": MLFeatureValue(multiArray: f0Pred),
                "N": MLFeatureValue(multiArray: nPred),
                "s": MLFeatureValue(multiArray: refAudioArray)
            ])

            let decoderOutput = try monitor.measure(PerformanceMonitor.Module.decoder) {
                return try decoder.prediction(from: decoderInput)
            }
            guard let x = decoderOutput.featureValue(for: "x")?.multiArrayValue,
                  let _ = decoderOutput.featureValue(for: "s")?.multiArrayValue,
                  let _ = decoderOutput.featureValue(for: "F0_curve")?.multiArrayValue else {
                throw NSError(domain: "TTSModel", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to get decoder output"])
            }

            let s = refAudioArray
            let F0_curve = f0Pred

            // Generate audio using the generator
            let audio = try monitor.measure(PerformanceMonitor.Module.generator) {
                return try generator.generate(x: x, s: s, f0Curve: F0_curve)
            }

            return audio
        }
    }

    private func splitStyleVector(_ refS: [Float]) -> (refAudio: [Float], style: [Float]) {
        // ref_s[:, :128] - first 128 elements for reference audio
        // ref_s[:, 128:] - last 128 elements for style
        let refAudio = Array(refS.prefix(128))
        let style = Array(refS.suffix(128))
        return (refAudio, style)
    }

    private func createAlignmentIndices(predDur: MLMultiArray, seqLen: Int) throws -> [Int] {
        // Convert pred_dur to integer durations
        var durations: [Int] = []
        for i in 0..<seqLen {
            let durValue = predDur[[i as NSNumber]].floatValue
            durations.append(Int(durValue))
        }

        // Create indices array by repeating each index by its duration
        var indices: [Int] = []
        for (index, duration) in durations.enumerated() {
            for _ in 0..<duration {
                indices.append(index)
            }
        }

        return indices
    }

    private func createAlignmentMatrix(indices: [Int], seqLen: Int) throws -> MLMultiArray {
        // Create alignment matrix of shape [1, seqLen, totalDuration]
        let totalDuration = indices.count
        let alignmentMatrix = try MLMultiArray(shape: [1, NSNumber(value: seqLen), NSNumber(value: totalDuration)], dataType: .float32)

        // Initialize with zeros
        let dataPointer = alignmentMatrix.dataPointer.bindMemory(to: Float32.self, capacity: alignmentMatrix.count)
        for i in 0..<alignmentMatrix.count {
            dataPointer[i] = 0.0
        }

        // Set 1s at alignment positions
        for (i, index) in indices.enumerated() {
            alignmentMatrix[[0, index as NSNumber, i as NSNumber]] = NSNumber(value: 1.0)
        }

        return alignmentMatrix
    }

    private func applyAlignment(d: MLMultiArray, alignmentMatrix: MLMultiArray) throws -> MLMultiArray {
        let seqLen = d.shape[1].intValue
        let hiddenDim = d.shape[2].intValue
        let totalDuration = alignmentMatrix.shape[2].intValue

        // First transpose d from [1, seqLen, hiddenDim] to [1, hiddenDim, seqLen]
        let dTransposed = try transposeLastTwoDimensions(d)

        // Create result array
        let result = try MLMultiArray(shape: [1, NSNumber(value: hiddenDim), NSNumber(value: totalDuration)], dataType: .float32)

        // Get data pointers
        let dPointer = dTransposed.dataPointer.bindMemory(to: Float32.self, capacity: dTransposed.count)
        let alignPointer = alignmentMatrix.dataPointer.bindMemory(to: Float32.self, capacity: alignmentMatrix.count)
        let resultPointer = result.dataPointer.bindMemory(to: Float32.self, capacity: result.count)

        // Perform matrix multiplication using BLAS
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            Int32(hiddenDim),
            Int32(totalDuration),
            Int32(seqLen),
            1.0,
            dPointer,
            Int32(seqLen),
            alignPointer,
            Int32(totalDuration),
            0.0,
            resultPointer,
            Int32(totalDuration)
        )

        return result
    }

    private func applyAlignmentDirect(input: MLMultiArray, alignmentMatrix: MLMultiArray) throws -> MLMultiArray {
        let hiddenDim = input.shape[1].intValue
        let seqLen = input.shape[2].intValue
        let totalDuration = alignmentMatrix.shape[2].intValue

        // Create result array
        let result = try MLMultiArray(shape: [1, NSNumber(value: hiddenDim), NSNumber(value: totalDuration)], dataType: .float32)

        // Get data pointers
        let inputPointer = input.dataPointer.bindMemory(to: Float32.self, capacity: input.count)
        let alignPointer = alignmentMatrix.dataPointer.bindMemory(to: Float32.self, capacity: alignmentMatrix.count)
        let resultPointer = result.dataPointer.bindMemory(to: Float32.self, capacity: result.count)

        // Perform matrix multiplication using BLAS
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            Int32(hiddenDim),
            Int32(totalDuration),
            Int32(seqLen),
            1.0,
            inputPointer,
            Int32(seqLen),
            alignPointer,
            Int32(totalDuration),
            0.0,
            resultPointer,
            Int32(totalDuration)
        )

        return result
    }

    /// Optimized transpose using vDSP
    private func transposeLastTwoDimensions(_ array: MLMultiArray) throws -> MLMultiArray {
        let batch = array.shape[0].intValue
        let dim1 = array.shape[1].intValue
        let dim2 = array.shape[2].intValue

        let transposed = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: dim2), NSNumber(value: dim1)],
                                         dataType: array.dataType)

        let sourcePointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let destPointer = transposed.dataPointer.bindMemory(to: Float32.self, capacity: transposed.count)

        // Use vDSP_mtrans for matrix transpose
        vDSP_mtrans(sourcePointer, 1, destPointer, 1, vDSP_Length(dim2), vDSP_Length(dim1))

        return transposed
    }
}
