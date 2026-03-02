import Foundation

/// Простая G2P реализация для языков, использующих базовую фонемизацию
/// Поддерживает: французский, испанский, итальянский, португальский, хинди
/// Основано на Python EspeakG2P с собственным espeak wrapper
public class G2PSimple: G2P {
    private let language: Language
    private let backend: EspeakBackend
    
    /// Базовый маппинг фонем как в Python EspeakG2P.e2m
    private var phonemeMapping: [(String, String)] = [
        ("a^ɪ", "I"), ("a^ʊ", "W"),
        ("d^z", "ʣ"), ("d^ʒ", "ʤ"),
        ("e^ɪ", "A"),
        ("o^ʊ", "O"), ("ə^ʊ", "Q"),
        ("s^s", "S"),
        ("t^s", "ʦ"), ("t^ʃ", "ʧ"),
        ("ɔ^ɪ", "Y")
    ]
    
    public init(language: Language, espeakDataPath: String? = nil) throws {
        guard [.french, .spanish, .italian, .portuguese, .hindi].contains(language) else {
            throw TTSError.invalidInput("G2PSimple doesn't support language: \(language.rawValue)")
        }
        self.language = language
        
        // В Python version=None по умолчанию, поэтому version 2.0 маппинги НЕ добавляются
        // Используем только базовые маппинги
        
        // Сортируем маппинг как в Python
        phonemeMapping.sort { $0.0.count > $1.0.count || ($0.0.count == $1.0.count && $0.0 < $1.0) }
        
        // Создаем EspeakBackend с параметрами как в Python EspeakG2P
        let languageCode = Self.getEspeakLanguageCode(for: language)
        self.backend = EspeakBackend(
            language: languageCode,
            preservePunctuation: true,      // как в Python
            withStress: true,              // как в Python  
            tie: .custom("^"),             // как в Python tie='^'
            languageSwitch: "remove-flags", // как в Python
            espeakDataPath: espeakDataPath
        )
    }
    
    public func convert(_ text: String) throws -> G2PResult {
        // Эквивалент Python EspeakG2P.__call__(text)
        // DEBUG: Отключено
        // print("🔤 Input text: '\(text)'")
        var processedText = text
        
        // Angles to curly quotes
        processedText = processedText
            .replacingOccurrences(of: "«", with: "\u{201C}")
            .replacingOccurrences(of: "»", with: "\u{201D}")
        
        // Parentheses to angles (для сохранения)
        processedText = processedText
            .replacingOccurrences(of: "(", with: "«")
            .replacingOccurrences(of: ")", with: "»")
        
        // DEBUG: Отключено  
        // print("🔄 Preprocessed text: '\(processedText)'")
        
        // В Python здесь вызывается phonemizer.backend.phonemize([text])
        // Используем наш EspeakBackend
        let phonemizedResults = backend.phonemize([processedText])
        guard !phonemizedResults.isEmpty else {
            throw TTSError.invalidInput("Failed to phonemize text for language: \(language.rawValue)")
        }
        
        // Берем единственный результат (как в Python)
        guard let phonemeString = phonemizedResults.first, !phonemeString.isEmpty else {
            throw TTSError.invalidInput("Failed to phonemize text for language: \(language.rawValue)")
        }
        
        // DEBUG: Включаем для проверки финального результата
        
        // Применяем пост-обработку как в Python EspeakG2P
        let finalPhonemes = postprocessPhonemes(phonemeString)
        
        
        return G2PResult(
            phonemeString: finalPhonemes,
            tokens: [] // Пустой массив токенов для простой реализации
        )
    }
    
    // MARK: - Helper Methods
    
    /// Получить код языка для espeak  
    private static func getEspeakLanguageCode(for language: Language) -> String {
        switch language {
        case .french:
            return "fr"  // Откат к рабочему варианту
        case .spanish:
            return "es"
        case .italian:
            return "it"
        case .portuguese:
            return "pt-br"
        case .hindi:
            return "hi"
        default:
            return language.rawValue
        }
    }
    
    /// Пост-обработка фонем как в Python EspeakG2P
    private func postprocessPhonemes(_ phonemes: String) -> String {
        var result = phonemes
        
        // Применяем маппинг фонем
        for (old, new) in phonemeMapping {
            result = result.replacingOccurrences(of: old, with: new)
        }
        
        // ВАЖНО: Vocab содержит носовые гласные как базовая_гласная + диакритик ̃ 
        // Swift обрабатывает носовые гласные как составные символы, а Python как отдельные
        // Нормализуем в NFD (Normalization Form Decomposed) для разделения base + combining
        result = result.decomposedStringWithCanonicalMapping
        
        // print("🔍 After NFD normalization: '\(result)'")
        
        // Delete any remaining tie characters
        result = result.replacingOccurrences(of: "^", with: "")
        
        // Normalize apostrophes (replace curly apostrophes with straight ones)
        result = result.replacingOccurrences(of: "'", with: "'")  // U+2019 -> U+0027
        result = result.replacingOccurrences(of: "'", with: "'")  // U+2018 -> U+0027
        
        // В Python version=None по умолчанию, поэтому используем else ветку
        // Удаляем дефисы (логика для версий != "2.0")
        result = result.replacingOccurrences(of: "-", with: "")
        
        // Angles back to parentheses
        result = result
            .replacingOccurrences(of: "«", with: "(")
            .replacingOccurrences(of: "»", with: ")")
        
        return result
    }
}
