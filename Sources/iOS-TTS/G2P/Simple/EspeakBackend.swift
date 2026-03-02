import Foundation

/// Swift реализация EspeakBackend из Python phonemizer
/// Самодостаточный класс, объединяющий функциональность BaseBackend и BaseEspeakBackend
/// Основано на phonemizer/backend/espeak/espeak.py
public class EspeakBackend {
    public let language: String
    public let preservePunctuation: Bool
    private let withStress: Bool
    private let tie: String?
    private let languageSwitch: String
    private let espeakDataPath: String?
    private let punctuationMarks: CharacterSet
    
    // Regular expression для поиска стрессов в espeak output
    private static let stressRegex = try! NSRegularExpression(pattern: "[ˈˌ'-]+", options: [])
    
    public init(
        language: String,
        preservePunctuation: Bool = false,
        withStress: Bool = false,
        tie: TieOption = .none,
        languageSwitch: String = "keep-flags",
        espeakDataPath: String? = nil
    ) {
        self.language = language
        self.preservePunctuation = preservePunctuation
        self.withStress = withStress
        self.tie = Self.initTie(tie)
        self.languageSwitch = languageSwitch
        self.espeakDataPath = espeakDataPath
        // Используем тот же набор пунктуации что и Python phonemizer
        // _DEFAULT_MARKS = ';:,.!?¡¿—…"«»""(){}[]'
        // НЕ включаем дефис и апостроф!
        // Python использует regex r'(\s*[{marks}]+\s*)+'  - включает пробелы!
        let punctuationString = ";:,.!?¡¿—…\"«»\u{201C}\u{201D}(){}[]"
        self.punctuationMarks = CharacterSet(charactersIn: punctuationString)
        
        // Проверяем что backend доступен
        guard Self.isAvailable() else {
            fatalError("\(Self.name()) not available on your system")
        }
        
        // Проверяем что язык поддерживается
        guard Self.supportedLanguages().keys.contains(language) else {
            fatalError("language \"\(language)\" is not supported by the \(Self.name()) backend")
        }
        
        // Initialize espeak
        let initialized = if let dataPath = espeakDataPath {
            EspeakSwiftWrapper.initialize(withDataPath: dataPath)
        } else {
            EspeakSwiftWrapper.initialize()
        }
        
        if !initialized {
        }
    }
    
    /// Tie option как в Python
    public enum TieOption {
        case none           // false
        case `default`      // true (default U+361 tie character)
        case custom(String) // custom tie character
    }
    
    private static func initTie(_ tie: TieOption) -> String? {
        switch tie {
        case .none:
            return nil
        case .default:
            return "͡" // U+361 tie character
        case .custom(let char):
            if char.count != 1 {
                return "͡"
            }
            return char
        }
    }
    
    // MARK: - Static methods
    
    public static func name() -> String {
        return "espeak"
    }
    
    public static func isAvailable() -> Bool {
        // Просто проверяем что espeak доступен, не инициализируем
        return true // espeak-ng всегда доступен через SPM
    }
    
    public static func version() -> (Int, Int, Int) {
        // Для простоты возвращаем версию espeak-ng
        return (1, 51, 1) // Типичная версия espeak-ng
    }
    
    public static func supportedLanguages() -> [String: String] {
        // Упрощенный список поддерживаемых языков
        return [
            "en": "English",
            "en-us": "English (US)", 
            "en-gb": "English (GB)",
            "fr": "French",
            "fr-fr": "French (France)",
            "es": "Spanish",
            "it": "Italian", 
            "pt": "Portuguese",
            "pt-br": "Portuguese (Brazil)",
            "hi": "Hindi"
        ]
    }
    
    public static func isEspeakNG() -> Bool {
        // espeak-ng начинается с версии 1.49
        let version = Self.version()
        return version.0 > 1 || (version.0 == 1 && version.1 >= 49)
    }
    
    // MARK: - Main phonemization
    
    /// Основной метод фонемизации (эквивалент phonemize в Python)
    public func phonemize(_ text: [String], separator: Separator = Separator.default, strip: Bool = false) -> [String] {
        // ВАЖНО: Python phonemizer НЕ разбивает на chunks перед espeak!
        // Он передает весь текст БЕЗ пунктуации в espeak как одну строку
        
        if preservePunctuation {
            // Как в Python: фонемизируем каждый chunk отдельно!
            var allPhonemes: [String] = []
            var allMarks: [[PunctuationMark]] = []
            
            for (lineNum, line) in text.enumerated() {
                let (chunks, marks) = preserveLine(line, lineNum: lineNum)
                
                // Фонемизируем каждый chunk отдельно (как Python!)
                for chunk in chunks {
                    let phonemized = _phonemize_aux([chunk], offset: 0, separator: separator, strip: strip)
                    allPhonemes.append(contentsOf: phonemized)
                }
                
                allMarks.append(marks)
            }
            
            // НЕ объединяем! Передаём фонемы как отдельные chunks в алгоритм восстановления
            
            // Восстанавливаем пунктуацию - передаём каждый chunk отдельно
            let result = restorePunctuationSimple(allPhonemes, marks: allMarks.flatMap { $0 })
            
            // Объединяем результат в одну строку для финального вывода
            let finalResult = result.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            return [finalResult]
        } else {
            // Без пунктуации - просто удаляем её
            let cleanText = removePunctuationMarks(text)
            return _phonemize_aux(cleanText, offset: 0, separator: separator, strip: strip)
        }
    }
    
    /// Реализация _phonemize_aux из BaseBackend
    public func _phonemize_aux(_ text: [String], offset: Int, separator: Separator, strip: Bool) -> [String] {
        var output: [String] = []
        
        for (num, line) in text.enumerated() {
            // print("🔍 Processing sentence \(num): '\(line)'")
            // Вызываем espeak wrapper (эквивалент _espeak.text_to_phonemes в Python)  
            guard let rawPhonemes = EspeakSwiftWrapper.textToPhonemes(line, language: language, dataPath: espeakDataPath) else {
                output.append("")
                continue
            }
            // print("📌 Raw espeak output for '\(line)': '\(rawPhonemes)'")
            
            // Постобработка строки (эквивалент _postprocess_line в Python)
            let (processedLine, _) = _postprocess_line(rawPhonemes, num: num + offset, separator: separator, strip: strip)
            // print("🔧 After _postprocess_line: '\(processedLine)'")
            output.append(processedLine)
        }
        
        return output
    }
    
    /// Реализация _postprocess_line из BaseEspeakBackend
    public func _postprocess_line(_ line: String, num: Int, separator: Separator, strip: Bool) -> (String, Bool) {
        // espeak может разделить utterance на несколько строк из-за пунктуации,
        // здесь мы объединяем строки в одну
        var processedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
        
        // Исправление бага в espeak-ng: дополнительные separators в конце слов
        // См. https://github.com/espeak-ng/espeak-ng/issues/694
        processedLine = processedLine.replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
        processedLine = processedLine.replacingOccurrences(of: "_ ", with: " ")
        
        // TODO: Обработка language switches (пока пропускаем)
        
        if processedLine.isEmpty {
            return ("", false)
        }
        
        var outLine = ""
        let words = processedLine.split(separator: " ")
        
        for word in words {
            var processedWord = String(word).trimmingCharacters(in: .whitespaces)
            
            // Обработка стрессов
            processedWord = processStress(processedWord)
            
            // Обработка tie characters
            processedWord = processTie(processedWord)
            
            outLine += processedWord + " " // word separator
        }
        
        // Удаляем последний word separator
        if outLine.hasSuffix(" ") {
            outLine = String(outLine.dropLast())
        }
        
        return (outLine, false) // (processed_line, has_language_switch)
    }
    
    // MARK: - Sentence Splitting
    
    /// Разбиение текста на предложения как в Python phonemizer
    /// Эквивалент функции sentence splitting в phonemizer
    private func splitIntoSentences(_ text: [String]) -> [String] {
        var sentences: [String] = []
        
        for line in text {
            // Разбиваем по основным знакам препинания, которые завершают предложения
            let sentenceTerminators = CharacterSet(charactersIn: ".!?")
            let components = line.components(separatedBy: sentenceTerminators)
            
            var currentSentences: [String] = []
            var originalIndex = 0
            
            for (i, component) in components.enumerated() {
                let trimmed = component.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    // Найти соответствующий знак пунктуации из исходного текста
                    let componentEndIndex = originalIndex + component.count
                    
                    if i < components.count - 1 && componentEndIndex < line.count {
                        let punctuationIndex = line.index(line.startIndex, offsetBy: componentEndIndex)
                        let punctuation = line[punctuationIndex]
                        currentSentences.append(trimmed + String(punctuation))
                    } else {
                        // Последний компонент без знака пунктуации
                        currentSentences.append(trimmed)
                    }
                }
                originalIndex += component.count + 1 // +1 для разделителя
            }
            
            sentences.append(contentsOf: currentSentences)
        }
        
        return sentences.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
    
    // MARK: - Preprocessing/Postprocessing
    
    /// Предобработка текста (удаление/сохранение пунктуации)
    private func _phonemize_preprocess(_ text: [String]) -> ([String], [[PunctuationMark]]) {
        // ВАЖНО: Python phonemizer НЕ разбивает текст на предложения в preprocess!
        // Он передает весь текст целиком и пусть espeak сам разбирается
        
        if preservePunctuation {
            // Сохраняем пунктуацию для последующего восстановления
            return preservePunctuationMarks(text)
        } else {
            // Удаляем всю пунктуацию
            return (removePunctuationMarks(text), [])
        }
    }
    
    /// Постобработка фонемизированного текста
    private func _phonemize_postprocess(_ phonemized: [String], punctuationMarks: [[PunctuationMark]], separator: Separator, strip: Bool) -> [String] {
        if preservePunctuation && !punctuationMarks.isEmpty {
            // Восстанавливаем пунктуацию
            return restorePunctuationMarks(phonemized, punctuationMarks: punctuationMarks, separator: separator, strip: strip)
        }
        return phonemized
    }
    
    /// Удаление пунктуации из текста
    private func removePunctuationMarks(_ text: [String]) -> [String] {
        return text.map { line in
            String(line.unicodeScalars.filter { !punctuationMarks.contains($0) })
        }
    }
    
    /// Тип позиции пунктуации (как в Python phonemizer)
    private enum PunctuationPosition {
        case begin    // B - в начале 
        case end      // E - в конце
        case intermediate // I - в середине
        case alone    // A - только пунктуация
    }
    
    private struct PunctuationMark {
        let index: Int          // К какому chunk'у относится (как в Python)
        let mark: String        // Сама пунктуация (может быть несколько символов)
        let position: PunctuationPosition
        
        init(index: Int, mark: String, position: PunctuationPosition) {
            self.index = index
            self.mark = mark
            self.position = position
        }
        
        // Backward compatibility
        var character: Character { Character(mark) }
        var chunkIndex: Int { index }
    }
    
    /// Сохранение пунктуации с определением позиций (как в Python)
    private func preservePunctuationMarks(_ text: [String]) -> ([String], [[PunctuationMark]]) {
        var allProcessedChunks: [String] = []
        var allPunctuationMarks: [[PunctuationMark]] = []
        
        for (lineNum, line) in text.enumerated() {
            // Разбиваем строку на chunks по пунктуации как в Python
            let (chunks, marks) = preserveLine(line, lineNum: lineNum)
            
            // DEBUG: Отключено
            // print("DEBUG: Original line: '\(line)'")
            // print("DEBUG: Chunks: \(chunks)")
            // print("DEBUG: Punctuation: \(marks.map { "(\($0.character), \($0.position))" })")
            
            allProcessedChunks.append(contentsOf: chunks)
            allPunctuationMarks.append(marks)
        }
        
        return (allProcessedChunks, allPunctuationMarks)
    }
    
    /// Разбивка одной строки на chunks с сохранением пунктуации (как в Python _preserve_line)
    private func preserveLine(_ line: String, lineNum: Int) -> ([String], [PunctuationMark]) {
        var chunks: [String] = []
        var marks: [PunctuationMark] = []
        var currentChunk = ""
        var inPunctuation = false
        var punctuationBuffer = ""
        var chunkIndex = 0  // Для отслеживания к какому chunk'у относится пунктуация
        
        // Сначала найдем все последовательности пунктуации
        for char in line {
            if punctuationMarks.contains(char.unicodeScalars.first!) {
                if !inPunctuation && !currentChunk.isEmpty {
                    // Закончился текстовый chunk, начинается пунктуация
                    chunks.append(currentChunk)
                    currentChunk = ""
                    chunkIndex = chunks.count - 1  // Пунктуация относится к последнему chunk'у
                }
                punctuationBuffer.append(char)
                inPunctuation = true
            } else {
                if inPunctuation {
                    // Закончилась пунктуация, сохраняем её
                    if !punctuationBuffer.isEmpty {
                        // Определяем позицию
                        let position: PunctuationPosition
                        if chunks.isEmpty {
                            position = .begin  // В начале строки
                        } else {
                            position = .intermediate  // В середине
                        }
                        // Добавляем пробел после intermediate пунктуации (как в Python)
                        var markWithSpaces = punctuationBuffer
                        if position == .intermediate {
                            markWithSpaces = punctuationBuffer + " "
                        }
                        // Сохраняем с индексом строки (как в Python) - все marks одной строки имеют один индекс!
                        marks.append(PunctuationMark(index: lineNum, mark: markWithSpaces, position: position))
                        punctuationBuffer = ""
                    }
                    inPunctuation = false
                }
                currentChunk.append(char)
            }
        }
        
        // Обработка остатков
        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }
        if !punctuationBuffer.isEmpty {
            // Пунктуация в конце
            let position: PunctuationPosition = chunks.isEmpty ? .alone : .end
            marks.append(PunctuationMark(index: lineNum, mark: punctuationBuffer, position: position))
        }
        
        // Фильтруем пустые chunks
        chunks = chunks.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        
        return (chunks, marks)
    }
    
    /// Полноценное восстановление пунктуации как в Python phonemizer
    private func restorePunctuationSimple(_ phonemized: [String], marks: [PunctuationMark]) -> [String] {
        // Реализация алгоритма phonemizer.punctuation.Punctuation.restore()
        var text = phonemized
        var marks = marks
        var punctuatedText: [String] = []
        var pos = 0
        let separator = Separator() // Используем дефолтный разделитель
        let strip = false
        
        while !text.isEmpty || !marks.isEmpty {
            
            if marks.isEmpty {
                // Нет больше пунктуации - добавляем оставшийся текст
                for line in text {
                    var finalLine = line
                    if !strip && !separator.word.isEmpty && !line.hasSuffix(separator.word) {
                        finalLine = line + separator.word
                    }
                    punctuatedText.append(finalLine)
                }
                text = []
                
            } else if text.isEmpty {
                // Текст закончился, остались только marks
                let remainingMarks = marks.map { $0.mark }.joined()
                let processedMarks = remainingMarks.replacingOccurrences(of: " ", with: separator.word)
                punctuatedText.append(processedMarks)
                marks = []
                
            } else {
                let currentMark = marks[0]
                
                if currentMark.index == pos {
                    // Пора вставить текущую пунктуацию
                    let mark = marks.removeFirst()
                    let markText = mark.mark.replacingOccurrences(of: " ", with: separator.word)
                    
                    // Убираем word separator в конце текущего слова если есть
                    if !separator.word.isEmpty && text[0].hasSuffix(separator.word) {
                        text[0] = String(text[0].dropLast(separator.word.count))
                    }
                    
                    switch mark.position {
                    case .begin:
                        // B: пунктуация в начале
                        text[0] = markText + text[0]
                        
                    case .end:
                        // E: пунктуация в конце - добавляем в результат
                        let finalText = text[0] + markText + (strip || markText.hasSuffix(separator.word) ? "" : separator.word)
                        punctuatedText.append(finalText)
                        text.removeFirst()
                        pos += 1
                        
                    case .alone:
                        // A: только пунктуация
                        let finalText = markText + (strip || markText.hasSuffix(separator.word) ? "" : separator.word)
                        punctuatedText.append(finalText)
                        pos += 1
                        
                    case .intermediate:
                        // I: пунктуация в середине - объединяем с следующим словом
                        // ВАЖНО: pos НЕ увеличиваем! (как в Python)
                        if text.count == 1 {
                            // Corner case: последняя часть intermediate mark не была фонемизирована
                            text[0] = text[0] + markText
                        } else {
                            let firstWord = text[0]
                            text.removeFirst()
                            text[0] = firstWord + markText + text[0]
                        }
                    }
                    
                } else {
                    // Индекс не совпадает - добавляем текущее слово в результат
                    punctuatedText.append(text[0])
                    text.removeFirst()
                    pos += 1
                }
            }
        }
        
        return punctuatedText
    }
    
    /// Восстановление пунктуации в фонемизированном тексте (как в Python)
    private func restorePunctuationMarks(_ phonemized: [String], punctuationMarks: [[PunctuationMark]], separator: Separator, strip: Bool) -> [String] {
        var result: [String] = []
        
        for (index, line) in phonemized.enumerated() {
            guard index < punctuationMarks.count else {
                result.append(line)
                continue
            }
            
            let linePunctuation = punctuationMarks[index]
            
            if linePunctuation.isEmpty {
                result.append(line)
                continue
            }
            
            // DEBUG: Commented out punctuation debug
            // print("DEBUG: Phonemized line: '\(line)'")
            // print("DEBUG: Punctuation to restore: \(linePunctuation.map { "(\($0.character), \($0.position))" })")
            
            var restoredLine = line
            
            // Восстанавливаем пунктуацию по позициям как в Python
            for mark in linePunctuation {
                switch mark.position {
                case .begin:
                    // B: добавить в начало
                    restoredLine = String(mark.character) + restoredLine
                    
                case .end:
                    // E: добавить в конец
                    restoredLine = restoredLine + String(mark.character)
                    
                case .intermediate:
                    // I: добавить после первого слова (упрощенная логика)
                    let words = restoredLine.split(separator: " ", maxSplits: 1)
                    if words.count >= 1 {
                        let firstWord = String(words[0]) + String(mark.character)
                        let rest = words.count > 1 ? " " + String(words[1]) : ""
                        restoredLine = firstWord + rest
                    } else {
                        restoredLine = restoredLine + String(mark.character)
                    }
                    
                case .alone:
                    // A: вся строка это пунктуация
                    restoredLine = String(mark.character)
                }
            }
            
            // print("DEBUG: Restored line: '\(restoredLine)'")
            result.append(restoredLine)
        }
        
        return result
    }
    
    // MARK: - Word processing
    
    /// Обработка стрессов (эквивалент _process_stress в Python)
    private func processStress(_ word: String) -> String {
        if withStress {
            return word
        }
        
        // Удаляем стрессы на фонемах
        let range = NSRange(location: 0, length: word.utf16.count)
        return Self.stressRegex.stringByReplacingMatches(
            in: word,
            options: [],
            range: range,
            withTemplate: ""
        )
    }
    
    /// Обработка tie characters (эквивалент _process_tie в Python)
    private func processTie(_ word: String) -> String {
        var result = word
        
        // ПРИМЕЧАНИЕ: баг в espeak добавляет ties к (en) флагам как (͡e͡n).
        // Мы не исправляем это здесь, как и в Python версии.
        
        if let tie = self.tie, tie != "͡" {
            // Заменяем дефолтный '͡' на запрошенный
            result = result.replacingOccurrences(of: "͡", with: tie)
        }
        
        // Заменяем '_' на phone separator (для простоты используем пробел)
        result = result.replacingOccurrences(of: "_", with: " ")
        
        // E2M постобработка как в Python misaki/espeak.py строка 22:
        // 'ʲo':'jo', 'ʲə':'jə', 'ʲ':''
        result = result.replacingOccurrences(of: "ʲo", with: "jo")
        result = result.replacingOccurrences(of: "ʲə", with: "jə")
        result = result.replacingOccurrences(of: "ʲ", with: "")  // Remove palatalization
        
        return result
    }
}

/// Структура для разделителей фонем, слогов и слов
public struct Separator: Sendable {
    public let phone: String
    public let syllable: String  
    public let word: String
    
    public init(phone: String = " ", syllable: String = "", word: String = " ") {
        self.phone = phone
        self.syllable = syllable
        self.word = word
    }
    
    public static let `default` = Separator()
}
