# Kokoro TTS iOS Benchmarks

## Test Parameters
- **Text**: "Hello, this is a test of text to speech technology." (или аналогичный)
- **Audio output**: ~48620 samples (~2 sec @ 24kHz)
- **Voice**: af_heart (American English, female)

---

## Benchmark Results

### Run 1 - Baseline (2025-02-25)
**Device**: [указать устройство]
**iOS Version**: [указать версию]
**Compute Units**: Generator=cpuOnly, остальные=default

| Module | Time (ms) | % of Pipeline |
|--------|-----------|---------------|
| BERT | 398.32 | 7.8% |
| BERT Encoder | 6.90 | 0.1% |
| Duration Encoder | 47.19 | 0.9% |
| Prosody Predictor | 35.25 | 0.7% |
| Alignment | 1.57 | 0.03% |
| F0 Predictor | 85.54 | 1.7% |
| Text Encoder | 49.28 | 1.0% |
| Decoder | 2544.87 | **50.1%** |
| **Generator (total)** | **1913.34** | **37.6%** |
| - F0 Upsample | 0.85 | - |
| - Sine Generator | 437.18 | - |
| - Source Module | 3.18 | - |
| - STFT | 141.62 | - |
| - Generator Core | 615.78 | - |
| - Inverse STFT | 302.83 | - |
| **Total Pipeline** | **5084.02** | 100% |

**Key Issues**:
1. Decoder - 50% времени (2.5 sec)
2. Generator Core на cpuOnly - 616ms
3. Sine Generator на Swift - 437ms
4. STFT/iSTFT через RosaKit - 444ms суммарно

---

### Run 2 - [описание изменений]
**Device**:
**Changes**:

| Module | Time (ms) | Delta |
|--------|-----------|-------|
| ... | ... | ... |

---

## Optimization Ideas

### High Priority
- [ ] Generator Core: убрать cpuOnly, попробовать Neural Engine
- [ ] Decoder: проверить compute units, возможно dynamic shapes мешают ANE
- [ ] SineGen: оптимизировать с Accelerate/vDSP или переделать в CoreML модель

### Medium Priority
- [ ] STFT/iSTFT: заменить RosaKit на Accelerate vDSP
- [ ] Уменьшить копирование данных между MLMultiArray

### Low Priority
- [ ] BERT: уже достаточно быстрый
- [ ] Мелкие модули (< 50ms): оптимизация даст мало
