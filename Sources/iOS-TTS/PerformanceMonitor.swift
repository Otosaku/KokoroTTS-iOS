import Foundation

/// Performance monitor for measuring execution time between TTS pipeline modules
public final class PerformanceMonitor: @unchecked Sendable {
    private var measurements: [String: TimeInterval] = [:]
    private var startTimes: [String: Date] = [:]
    private let queue = DispatchQueue(label: "com.ios-tts.performance", attributes: .concurrent)
    
    /// Singleton instance for global access
    public static let shared = PerformanceMonitor()
    
    /// Enable or disable performance monitoring
    public var isEnabled: Bool = true
    
    private init() {}
    
    /// Start timing for a specific module
    /// - Parameter module: The name of the module to start timing
    public func startMeasurement(_ module: String) {
        guard isEnabled else { return }
        
        queue.async(flags: .barrier) {
            self.startTimes[module] = Date()
        }
    }
    
    /// End timing for a specific module
    /// - Parameter module: The name of the module to end timing
    public func endMeasurement(_ module: String) {
        guard isEnabled else { return }
        
        let endTime = Date()
        
        queue.async(flags: .barrier) {
            if let startTime = self.startTimes[module] {
                let duration = endTime.timeIntervalSince(startTime)
                self.measurements[module] = duration
                self.startTimes.removeValue(forKey: module)
            }
        }
    }
    
    /// Measure the execution time of a closure
    /// - Parameters:
    ///   - module: The name of the module being measured
    ///   - operation: The closure to measure
    /// - Returns: The result of the closure
    /// - Throws: Any error thrown by the closure
    public func measure<T>(_ module: String, operation: () throws -> T) rethrows -> T {
        guard isEnabled else { return try operation() }
        
        startMeasurement(module)
        defer { endMeasurement(module) }
        return try operation()
    }
    
    /// Measure the execution time of an async closure
    /// - Parameters:
    ///   - module: The name of the module being measured
    ///   - operation: The async closure to measure
    /// - Returns: The result of the closure
    /// - Throws: Any error thrown by the closure
    public func measureAsync<T>(_ module: String, operation: () async throws -> T) async rethrows -> T {
        guard isEnabled else { return try await operation() }
        
        startMeasurement(module)
        defer { endMeasurement(module) }
        return try await operation()
    }
    
    /// Get all measurements
    /// - Returns: Dictionary of module names to execution times in seconds
    public func getAllMeasurements() -> [String: TimeInterval] {
        queue.sync {
            return measurements
        }
    }
    
    /// Get measurement for a specific module
    /// - Parameter module: The name of the module
    /// - Returns: The execution time in seconds, or nil if not measured
    public func getMeasurement(for module: String) -> TimeInterval? {
        queue.sync {
            return measurements[module]
        }
    }
    
    /// Clear all measurements
    public func clearMeasurements() {
        queue.async(flags: .barrier) {
            self.measurements.removeAll()
            self.startTimes.removeAll()
        }
    }
    
    /// Generate a performance report
    /// - Returns: A formatted string with performance metrics
    public func generateReport() -> String {
        let measurements = getAllMeasurements()
        guard !measurements.isEmpty else {
            return "No performance measurements available"
        }
        
        var report = "=== TTS Performance Report ===\n"
        report += "Module".padding(toLength: 30, withPad: " ", startingAt: 0) + " | "
        report += "Time (ms)".padding(toLength: 15, withPad: " ", startingAt: 0) + " | "
        report += "Time (s)".padding(toLength: 10, withPad: " ", startingAt: 0) + "\n"
        report += String(repeating: "-", count: 60) + "\n"
        
        let sortedMeasurements = measurements.sorted { $0.key < $1.key }
        var totalTime: TimeInterval = 0
        
        for (module, time) in sortedMeasurements {
            let timeMs = time * 1000
            let moduleName = module.padding(toLength: 30, withPad: " ", startingAt: 0)
            let timeMsStr = String(format: "%.2f", timeMs).padding(toLength: 15, withPad: " ", startingAt: 0)
            let timeSecStr = String(format: "%.3f", time).padding(toLength: 10, withPad: " ", startingAt: 0)
            report += "\(moduleName) | \(timeMsStr) | \(timeSecStr)\n"
            totalTime += time
        }
        
        report += String(repeating: "-", count: 60) + "\n"
        let totalName = "TOTAL".padding(toLength: 30, withPad: " ", startingAt: 0)
        let totalMsStr = String(format: "%.2f", totalTime * 1000).padding(toLength: 15, withPad: " ", startingAt: 0)
        let totalSecStr = String(format: "%.3f", totalTime).padding(toLength: 10, withPad: " ", startingAt: 0)
        report += "\(totalName) | \(totalMsStr) | \(totalSecStr)\n"
        
        return report
    }
    
    /// Print the performance report to console
    public func printReport() {
        Swift.print(generateReport())
    }
}

/// Convenience extension for module names
public extension PerformanceMonitor {
    enum Module {
        static let bert = "BERT"
        static let bertEncoder = "BERT Encoder"
        static let durationEncoder = "Duration Encoder"
        static let prosodyPredictor = "Prosody Predictor"
        static let f0Predictor = "F0 Predictor"
        static let textEncoder = "Text Encoder"
        static let decoder = "Decoder"
        static let generator = "Generator"
        static let f0Upsample = "F0 Upsample"
        static let sourceModule = "Source Module"
        static let sineGen = "Sine Generator"
        static let stft = "STFT"
        static let inverseSTFT = "Inverse STFT"
        static let alignment = "Alignment"
        static let total = "Total Pipeline"
    }
}