# KokoroTTS-iOS

Swift Package for text-to-speech on iOS using the Kokoro TTS model.

> **Note:** Currently, the model is adapted and tested only for **English (en_us, en_gb)** and **French (fr)** languages. All other languages require preprocessor improvements. Contributors are welcome!

## Installation

Add the package to your project via Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/Otosaku/KokoroTTS-iOS", from: "1.1.0")
]
```

Or in Xcode: **File → Add Package Dependencies...** and enter the repository URL.

## Required Resources

Before using the library, you need to download the following resources:

| Resource | Description | URL |
|----------|-------------|-----|
| TTS Models | Core ML models for speech synthesis | [Download](https://firebasestorage.googleapis.com/v0/b/my-project-1494707780868.firebasestorage.app/o/converted.zip?alt=media&token=c27a1359-37c6-4b26-bd7d-8471d409a841) |
| G2P Vocab | Grapheme-to-phoneme vocabulary files | [Download](https://firebasestorage.googleapis.com/v0/b/my-project-1494707780868.firebasestorage.app/o/v6%2Fg2p.zip?alt=media&token=c42ca3e3-c743-40a0-9f72-9afa5e8007f9) |
| POS Model | Part-of-speech tagging model | [Download](https://firebasestorage.googleapis.com/v0/b/my-project-1494707780868.firebasestorage.app/o/quantized-bert-pos-tag.zip?alt=media&token=cd8b9030-8abd-4385-9fb9-9ec27ae5cad7) |
| Espeak Data | Espeak-ng phoneme data | [Download](https://firebasestorage.googleapis.com/v0/b/my-project-1494707780868.firebasestorage.app/o/v6%2Fespeak-ng-data-complete.zip?alt=media&token=a3f64856-c99f-4104-a04f-a34cde286648) |

## Usage

### Initialize the Pipeline

```swift
import iOS_TTS
import CoreML

// Configure ML compute units (optional)
let configuration = MLModelConfiguration()
configuration.computeUnits = .all // or .cpuAndNeuralEngine, .cpuOnly

// Initialize the pipeline
let pipeline = try TTSPipeline(
    modelPath: modelsDirectory,           // URL to extracted TTS models
    vocabURL: vocabDirectory,             // URL to extracted G2P vocab
    postaggerModelURL: posModelsDirectory, // URL to extracted POS model
    language: .englishUS,                  // Language selection
    espeakDataPath: espeakDataDirectory.path, // Path to espeak-ng data
    configuration: configuration
)
```

### Generate Speech

```swift
// Basic generation
let audioSamples = try await pipeline.generate(text: "Hello, world!")

// With options
let options = GenerationOptions(
    style: .afHeart,  // Voice style
    speed: 1.0        // Speech speed
)
let audioSamples = try await pipeline.generate(text: "Hello, world!", options: options)
```

### Available Languages

| Language | Code | Status |
|----------|------|--------|
| English (US) | `.englishUS` | Supported |
| English (GB) | `.englishGB` | Supported |
| French | `.french` | Supported |
| Spanish | `.spanish` | Needs work |
| Italian | `.italian` | Needs work |
| Portuguese | `.portuguese` | Needs work |
| Hindi | `.hindi` | Needs work |
| Japanese | `.japanese` | Needs work |
| Chinese | `.chinese` | Needs work |

### Available Voices

**English (US):** afHeart, afAlloy, afAoede, afBella, afJessica, afKore, afNicole, afNova, afRiver, afSarah, afSky, amAdam, amEcho, amEric, amFenrir, amLiam, amMichael, amOnyx, amPuck, amSanta

**English (GB):** bfAlice, bfEmma, bfIsabella, bfLily, bmDaniel, bmFable, bmGeorge, bmLewis

**French:** ffSiwis

## Example Project

The repository includes an Example app that demonstrates:
- Downloading and extracting required resources
- Initializing the TTS pipeline
- Generating and playing speech
- Voice and language selection

To run the example:
1. Open `Example/Example.xcodeproj` in Xcode
2. Build and run on an iOS device

## Requirements

- iOS 16.0+
- Swift 6.0+
- Xcode 16.0+

## License

MIT License
