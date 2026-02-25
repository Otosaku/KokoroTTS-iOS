import Foundation
import CoreML
import Accelerate

/// Temporary class for testing SineGen optimizations
/// TODO: Remove after optimization is complete
public class SineGenTester {

    public struct TestResult {
        public let originalTimeMs: Double
        public let optimizedTimeMs: Double
        public let speedup: Double
        public let shapesMatch: Bool
        public let shapeOriginal: [Int]
        public let shapeOptimized: [Int]
        public let elementCountOriginal: Int
        public let elementCountOptimized: Int
        public let statsOriginal: Stats
        public let statsOptimized: Stats
        public let maxDiffVsOriginal: Float  // Difference between original and optimized in deterministic mode
        public let maxDiffOptimizedRuns: Float  // Difference between two optimized runs (should be 0)
        public let profilingReport: String?  // Detailed timing breakdown

        public struct Stats {
            public let mean: Float
            public let std: Float
            public let min: Float
            public let max: Float
        }
    }

    public init() {}

    /// Run comparison test between SineGen and SineGenOptimized
    /// - Parameter length: Length of test F0 input (default ~2 seconds at 24kHz)
    /// - Returns: Test results with timing and accuracy info
    public func runTest(length: Int = 48000) throws -> TestResult {
        let f0 = try createTestF0(length: length)

        let original = SineGen()
        let optimized = SineGenOptimized()

        // Enable profiling for optimized version
        optimized.enableProfiling()

        // Warmup
        _ = try original.forward(f0)
        _ = try optimized.forward(f0)

        // Measure original
        let iterations = 5
        var originalTime: Double = 0
        var optimizedTime: Double = 0

        var outputOriginal: MLMultiArray?
        var outputOptimized: MLMultiArray?

        for i in 0..<iterations {
            let start1 = CFAbsoluteTimeGetCurrent()
            outputOriginal = try original.forward(f0)
            originalTime += CFAbsoluteTimeGetCurrent() - start1

            let start2 = CFAbsoluteTimeGetCurrent()
            outputOptimized = try optimized.forward(f0)
            optimizedTime += CFAbsoluteTimeGetCurrent() - start2
        }

        originalTime = (originalTime / Double(iterations)) * 1000 // to ms
        optimizedTime = (optimizedTime / Double(iterations)) * 1000 // to ms

        // Check shapes
        let shapeOrig = outputOriginal!.shape.map { $0.intValue }
        let shapeOpt = outputOptimized!.shape.map { $0.intValue }
        let shapesMatch = shapeOrig == shapeOpt

        // Compute stats
        let statsOrig = computeStats(outputOriginal!)
        let statsOpt = computeStats(outputOptimized!)

        // Test deterministic mode - compare original vs optimized
        let (diffVsOriginal, diffOptRuns) = try testDeterministicMode(length: length / 4)

        // Get profiling report
        let profilingReport = optimized.getProfilingReport()

        return TestResult(
            originalTimeMs: originalTime,
            optimizedTimeMs: optimizedTime,
            speedup: originalTime / optimizedTime,
            shapesMatch: shapesMatch,
            shapeOriginal: shapeOrig,
            shapeOptimized: shapeOpt,
            elementCountOriginal: outputOriginal!.count,
            elementCountOptimized: outputOptimized!.count,
            statsOriginal: statsOrig,
            statsOptimized: statsOpt,
            maxDiffVsOriginal: diffVsOriginal,
            maxDiffOptimizedRuns: diffOptRuns,
            profilingReport: profilingReport
        )
    }

    /// Generate report string from test results
    public func generateReport(_ result: TestResult) -> String {
        var report = "=== SineGen Test Report ===\n\n"

        report += "Performance:\n"
        report += String(format: "  Original:  %.2f ms\n", result.originalTimeMs)
        report += String(format: "  Optimized: %.2f ms\n", result.optimizedTimeMs)
        report += String(format: "  Speedup:   %.2fx\n\n", result.speedup)

        report += "Shapes:\n"
        report += "  Original:  \(result.shapeOriginal) (\(result.elementCountOriginal) elements)\n"
        report += "  Optimized: \(result.shapeOptimized) (\(result.elementCountOptimized) elements)\n"
        report += "  Match: \(result.shapesMatch ? "YES" : "NO")\n\n"

        report += "Statistics (Original):\n"
        report += String(format: "  Mean: %.6f\n", result.statsOriginal.mean)
        report += String(format: "  Std:  %.6f\n", result.statsOriginal.std)
        report += String(format: "  Range: [%.4f, %.4f]\n\n", result.statsOriginal.min, result.statsOriginal.max)

        report += "Statistics (Optimized):\n"
        report += String(format: "  Mean: %.6f\n", result.statsOptimized.mean)
        report += String(format: "  Std:  %.6f\n", result.statsOptimized.std)
        report += String(format: "  Range: [%.4f, %.4f]\n\n", result.statsOptimized.min, result.statsOptimized.max)

        report += "Deterministic mode:\n"
        report += String(format: "  Max diff vs original: %.2e\n", result.maxDiffVsOriginal)
        report += String(format: "  Max diff optimized runs: %.2e\n", result.maxDiffOptimizedRuns)
        let pass = result.maxDiffVsOriginal < 1e-5 && result.maxDiffOptimizedRuns < 1e-5
        report += "  Status: \(pass ? "PASS" : "FAIL")\n"

        if let profiling = result.profilingReport {
            report += "\n" + profiling
        }

        return report
    }

    // MARK: - Private

    private func createTestF0(length: Int) throws -> MLMultiArray {
        let f0 = try MLMultiArray(shape: [1, NSNumber(value: length), 1], dataType: .float32)
        let pointer = f0.dataPointer.bindMemory(to: Float32.self, capacity: f0.count)

        for i in 0..<length {
            // Simulate voiced speech with F0 around 150-250 Hz
            pointer[i] = 150.0 + 100.0 * sin(Float(i) * 0.01)
        }

        return f0
    }

    private func computeStats(_ array: MLMultiArray) -> TestResult.Stats {
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

        return TestResult.Stats(mean: mean, std: std, min: minVal, max: maxVal)
    }

    private func testDeterministicMode(length: Int) throws -> (vsOriginal: Float, optimizedRuns: Float) {
        let f0 = try createTestF0(length: length)

        // Compare original vs optimized (both deterministic)
        let original = SineGen(deterministic: true)
        let optimized = SineGenOptimized(deterministic: true)

        let outOriginal = try original.forward(f0)
        let outOptimized = try optimized.forward(f0)

        let ptrOrig = outOriginal.dataPointer.bindMemory(to: Float32.self, capacity: outOriginal.count)
        let ptrOpt = outOptimized.dataPointer.bindMemory(to: Float32.self, capacity: outOptimized.count)

        var maxDiffVsOriginal: Float = 0
        for i in 0..<outOriginal.count {
            maxDiffVsOriginal = max(maxDiffVsOriginal, abs(ptrOrig[i] - ptrOpt[i]))
        }

        // Compare two optimized runs (should be identical)
        let opt2 = SineGenOptimized(deterministic: true)
        let outOpt2 = try opt2.forward(f0)
        let ptrOpt2 = outOpt2.dataPointer.bindMemory(to: Float32.self, capacity: outOpt2.count)

        var maxDiffOptRuns: Float = 0
        for i in 0..<outOptimized.count {
            maxDiffOptRuns = max(maxDiffOptRuns, abs(ptrOpt[i] - ptrOpt2[i]))
        }

        return (maxDiffVsOriginal, maxDiffOptRuns)
    }
}
