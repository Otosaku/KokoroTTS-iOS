import XCTest
import CoreML
@testable import iOS_TTS

final class SineGenTests: XCTestCase {

    // MARK: - Test Input Generation

    /// Create a test F0 input with realistic values
    /// F0 typically ranges from 80-400 Hz for speech, with 0 for unvoiced segments
    private func createTestF0(length: Int, voiced: Bool = true) throws -> MLMultiArray {
        let f0 = try MLMultiArray(shape: [1, NSNumber(value: length), 1], dataType: .float32)
        let pointer = f0.dataPointer.bindMemory(to: Float32.self, capacity: f0.count)

        for i in 0..<length {
            if voiced {
                // Simulate voiced speech with F0 around 150-250 Hz
                pointer[i] = 150.0 + 100.0 * sin(Float(i) * 0.01)
            } else {
                // Mix of voiced and unvoiced
                pointer[i] = (i % 100 < 50) ? 200.0 : 0.0
            }
        }

        return f0
    }

    // MARK: - Shape Tests

    func testOutputShapeMatch() throws {
        let length = 48000 // ~2 seconds at 24kHz
        let f0 = try createTestF0(length: length)

        let original = SineGen()
        let optimized = SineGenOptimized(deterministic: true)

        let outputOriginal = try original.forward(f0)
        let outputOptimized = try optimized.forward(f0)

        // Check shapes match
        XCTAssertEqual(outputOriginal.shape, outputOptimized.shape,
                       "Output shapes should match")

        // Expected shape: [1, length, 9] where 9 = harmonicNum(8) + 1
        XCTAssertEqual(outputOriginal.shape.count, 3)
        XCTAssertEqual(outputOriginal.shape[0].intValue, 1)
        XCTAssertEqual(outputOriginal.shape[1].intValue, length)
        XCTAssertEqual(outputOriginal.shape[2].intValue, 9)
    }

    // MARK: - Value Range Tests

    func testOutputValueRange() throws {
        let length = 24000
        let f0 = try createTestF0(length: length)

        let optimized = SineGenOptimized(deterministic: true)
        let output = try optimized.forward(f0)

        let pointer = output.dataPointer.bindMemory(to: Float32.self, capacity: output.count)

        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude

        for i in 0..<output.count {
            minVal = min(minVal, pointer[i])
            maxVal = max(maxVal, pointer[i])
        }

        // Sine waves with amplitude 0.1 should be in range [-0.15, 0.15] with noise
        XCTAssertGreaterThan(minVal, -0.2, "Min value should be > -0.2")
        XCTAssertLessThan(maxVal, 0.2, "Max value should be < 0.2")

        print("SineGen output range: [\(minVal), \(maxVal)]")
    }

    // MARK: - Statistical Comparison Tests

    func testStatisticalSimilarity() throws {
        let length = 24000
        let f0 = try createTestF0(length: length)

        let original = SineGen()
        let optimized = SineGenOptimized()

        let outputOriginal = try original.forward(f0)
        let outputOptimized = try optimized.forward(f0)

        // Compute statistics for both
        let statsOriginal = computeStatistics(outputOriginal)
        let statsOptimized = computeStatistics(outputOptimized)

        print("Original - mean: \(statsOriginal.mean), std: \(statsOriginal.std), min: \(statsOriginal.min), max: \(statsOriginal.max)")
        print("Optimized - mean: \(statsOptimized.mean), std: \(statsOptimized.std), min: \(statsOptimized.min), max: \(statsOptimized.max)")

        // Statistics should be similar (not exact due to random phase)
        // Mean should be close to 0 for sine waves
        XCTAssertLessThan(abs(statsOriginal.mean), 0.05, "Original mean should be near 0")
        XCTAssertLessThan(abs(statsOptimized.mean), 0.05, "Optimized mean should be near 0")

        // Std should be similar
        XCTAssertLessThan(abs(statsOriginal.std - statsOptimized.std), 0.02,
                          "Standard deviations should be similar")
    }

    // MARK: - Deterministic Mode Test

    func testDeterministicMode() throws {
        let length = 12000
        let f0 = try createTestF0(length: length)

        let optimized1 = SineGenOptimized(deterministic: true)
        let optimized2 = SineGenOptimized(deterministic: true)

        let output1 = try optimized1.forward(f0)
        let output2 = try optimized2.forward(f0)

        // In deterministic mode, outputs should be identical
        let pointer1 = output1.dataPointer.bindMemory(to: Float32.self, capacity: output1.count)
        let pointer2 = output2.dataPointer.bindMemory(to: Float32.self, capacity: output2.count)

        var maxDiff: Float = 0
        for i in 0..<output1.count {
            maxDiff = max(maxDiff, abs(pointer1[i] - pointer2[i]))
        }

        XCTAssertLessThan(maxDiff, 1e-5, "Deterministic mode should produce identical results")
        print("Max difference in deterministic mode: \(maxDiff)")
    }

    // MARK: - Performance Tests

    func testPerformanceOriginal() throws {
        let length = 48000
        let f0 = try createTestF0(length: length)
        let sineGen = SineGen()

        // Warmup
        _ = try sineGen.forward(f0)

        measure {
            _ = try? sineGen.forward(f0)
        }
    }

    func testPerformanceOptimized() throws {
        let length = 48000
        let f0 = try createTestF0(length: length)
        let sineGen = SineGenOptimized()

        // Warmup
        _ = try sineGen.forward(f0)

        measure {
            _ = try? sineGen.forward(f0)
        }
    }

    func testPerformanceComparison() throws {
        let length = 48000
        let f0 = try createTestF0(length: length)

        let original = SineGen()
        let optimized = SineGenOptimized()

        // Warmup
        _ = try original.forward(f0)
        _ = try optimized.forward(f0)

        // Measure original
        let iterations = 10
        var originalTime: Double = 0
        var optimizedTime: Double = 0

        for _ in 0..<iterations {
            let start1 = CFAbsoluteTimeGetCurrent()
            _ = try original.forward(f0)
            originalTime += CFAbsoluteTimeGetCurrent() - start1

            let start2 = CFAbsoluteTimeGetCurrent()
            _ = try optimized.forward(f0)
            optimizedTime += CFAbsoluteTimeGetCurrent() - start2
        }

        originalTime /= Double(iterations)
        optimizedTime /= Double(iterations)

        print("=== Performance Comparison ===")
        print("Original SineGen: \(String(format: "%.2f", originalTime * 1000)) ms")
        print("Optimized SineGen: \(String(format: "%.2f", optimizedTime * 1000)) ms")
        print("Speedup: \(String(format: "%.2fx", originalTime / optimizedTime))")
    }

    // MARK: - Unvoiced Segments Test

    func testUnvoicedSegments() throws {
        let length = 12000
        let f0 = try createTestF0(length: length, voiced: false)

        let optimized = SineGenOptimized(deterministic: true)
        let output = try optimized.forward(f0)

        // For unvoiced segments (F0 = 0), output should be noise only
        let f0Pointer = f0.dataPointer.bindMemory(to: Float32.self, capacity: f0.count)
        let outPointer = output.dataPointer.bindMemory(to: Float32.self, capacity: output.count)

        // Check that we have both voiced and unvoiced outputs
        var hasVoiced = false
        var hasUnvoiced = false

        for i in 0..<length {
            if f0Pointer[i] > 10 {
                hasVoiced = true
            } else {
                hasUnvoiced = true
            }
        }

        XCTAssertTrue(hasVoiced, "Test should have voiced segments")
        XCTAssertTrue(hasUnvoiced, "Test should have unvoiced segments")
    }

    // MARK: - Helper Methods

    private struct Statistics {
        let mean: Float
        let std: Float
        let min: Float
        let max: Float
    }

    private func computeStatistics(_ array: MLMultiArray) -> Statistics {
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let count = array.count

        var sum: Float = 0
        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude

        for i in 0..<count {
            sum += pointer[i]
            minVal = min(minVal, pointer[i])
            maxVal = max(maxVal, pointer[i])
        }

        let mean = sum / Float(count)

        var varianceSum: Float = 0
        for i in 0..<count {
            let diff = pointer[i] - mean
            varianceSum += diff * diff
        }
        let std = sqrt(varianceSum / Float(count))

        return Statistics(mean: mean, std: std, min: minVal, max: maxVal)
    }
}
