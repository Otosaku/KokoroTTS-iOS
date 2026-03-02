import Foundation

/// Контекст для обработки токенов (как TokenContext в Python)
public class TokenContext {
    var futureVowel: Bool? = nil
    var futureTo: Bool = false
    
    public init(futureVowel: Bool? = nil, futureTo: Bool = false) {
        self.futureVowel = futureVowel
        self.futureTo = futureTo
    }
}

/// Класс для работы со словарями фонем (эквивалент Python Lexicon)
public class Lexicon {
    private let british: Bool
    private let capStresses: (Double, Double) = (0.5, 2.0)
    private var golds: [String: Any] = [:]
    private var silvers: [String: Any] = [:]
    
    // Константы из Python
    private static let diphthongs = Set("AIOQWYʤʧ")
    private static let consonants = Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
    private static let usTaus = Set("AIOWYiuæɑəɛɪɹʊʌ")
    private static let vowels = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ".map { String($0) })
    private static let stresses = "ˌˈ"
    private static let primaryStress = "ˈ"
    private static let secondaryStress = "ˌ"
    
    private static let currencies: [String: (String, String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent")
    ]
    
    private static let ordinals = Set(["st", "nd", "rd", "th"])
    private static let addSymbols = [".": "dot", "/": "slash"]
    private static let symbols = ["%": "percent", "&": "and", "+": "plus", "@": "at"]
    
    private static let usVocab = Set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ")
    private static let gbVocab = Set("AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ")
    
    private static let lexiconOrds: Set<Int> = Set([39, 45] + Array(65...90) + Array(97...123))
    
    public init(british: Bool, vocabURL: URL) throws {
        self.british = british
        try loadVocabularies(from: vocabURL)
    }
    
    private func loadVocabularies(from url: URL) throws {
        // Загружаем gold словарь
        let goldFilename = british ? "en_gb_gold.json" : "en_us_gold.json"
        let goldURL = url.appendingPathComponent(goldFilename)
        
        // Проверяем существование файла
        guard FileManager.default.fileExists(atPath: goldURL.path) else {
            throw TTSError.vocabLoadFailed("Gold vocabulary file not found: \(goldFilename)")
        }
        
        let goldData = try Data(contentsOf: goldURL)
        let goldDict = try JSONSerialization.jsonObject(with: goldData) as? [String: Any] ?? [:]
        self.golds = Lexicon.growDictionary(goldDict)
        
        // Загружаем silver словарь
        let silverFilename = british ? "en_gb_silver.json" : "en_us_silver.json"
        let silverURL = url.appendingPathComponent(silverFilename)
        
        // Проверяем существование файла
        guard FileManager.default.fileExists(atPath: silverURL.path) else {
            throw TTSError.vocabLoadFailed("Silver vocabulary file not found: \(silverFilename)")
        }
        
        let silverData = try Data(contentsOf: silverURL)
        let silverDict = try JSONSerialization.jsonObject(with: silverData) as? [String: Any] ?? [:]
        self.silvers = Lexicon.growDictionary(silverDict)
        
    }
    
    /// Расширение словаря (как grow_dictionary в Python)
    private static func growDictionary(_ dict: [String: Any]) -> [String: Any] {
        var extended: [String: Any] = [:]
        
        for (key, value) in dict {
            if key.count < 2 {
                continue
            }
            
            if key == key.lowercased() {
                if key != key.capitalized {
                    extended[key.capitalized] = value
                }
            } else if key == key.lowercased().capitalized {
                extended[key.lowercased()] = value
            }
        }
        
        return extended.merging(dict) { (_, new) in new }
    }
    
    /// Получение NNP фонем (как get_NNP в Python)
    private func getNNP(_ word: String) -> (String?, Int?) {
        let phonemes = word.compactMap { char in
            char.isLetter ? golds[String(char.uppercased())] as? String : nil
        }
        
        if phonemes.isEmpty {
            return (nil, nil)
        }
        
        let joined = phonemes.compactMap { $0 }.joined()
        let stressed = applyStress(joined, stress: 0)
        let parts = stressed.components(separatedBy: Lexicon.secondaryStress)
        let result = parts.joined(separator: Lexicon.primaryStress)
        
        return (result, 3)
    }
    
    /// Применение ударения (как apply_stress в Python)
    private func applyStress(_ phonemes: String, stress: Double?) -> String {
        guard let stress = stress else { return phonemes }
        
        if stress < -1 {
            return phonemes.replacingOccurrences(of: Lexicon.primaryStress, with: "")
                           .replacingOccurrences(of: Lexicon.secondaryStress, with: "")
        } else if stress == -1 || (stress == 0 || stress == -0.5) && phonemes.contains(Lexicon.primaryStress) {
            return phonemes.replacingOccurrences(of: Lexicon.secondaryStress, with: "")
                           .replacingOccurrences(of: Lexicon.primaryStress, with: Lexicon.secondaryStress)
        } else if (stress == 0 || stress == 0.5 || stress == 1) && !phonemes.contains(where: { Lexicon.stresses.contains($0) }) {
            if !phonemes.contains(where: { Lexicon.vowels.contains(String($0)) }) {
                return phonemes
            }
            return restress(Lexicon.secondaryStress + phonemes)
        } else if stress >= 1 && !phonemes.contains(Lexicon.primaryStress) && phonemes.contains(Lexicon.secondaryStress) {
            return phonemes.replacingOccurrences(of: Lexicon.secondaryStress, with: Lexicon.primaryStress)
        } else if stress > 1 && !phonemes.contains(where: { Lexicon.stresses.contains($0) }) {
            if !phonemes.contains(where: { Lexicon.vowels.contains(String($0)) }) {
                return phonemes
            }
            return restress(Lexicon.primaryStress + phonemes)
        }
        
        return phonemes
    }
    
    /// Перестановка ударений (как restress в Python)
    private func restress(_ phonemes: String) -> String {
        let chars = Array(phonemes)
        var indexed: [(Int, Character)] = []
        
        for (i, char) in chars.enumerated() {
            indexed.append((i, char))
        }
        
        var stresses: [Int: Int] = [:]
        for (i, char) in indexed {
            if Lexicon.stresses.contains(char) {
                if let vowelIndex = indexed[i...].first(where: { Lexicon.vowels.contains(String($0.1)) })?.0 {
                    stresses[i] = vowelIndex
                }
            }
        }
        
        for (i, j) in stresses {
            indexed[i] = (j - 1, indexed[i].1) // j - 0.5 approximated as j - 1
        }
        
        return String(indexed.sorted { $0.0 < $1.0 }.map { $0.1 })
    }
    
    /// Проверка является ли строка числом
    private static func isDigit(_ text: String) -> Bool {
        return text.range(of: "^[0-9]+$", options: .regularExpression) != nil
    }
    
    /// Получение специальных случаев (как get_special_case в Python)
    public func getSpecialCase(_ word: String, tag: String?, stress: Double?, context: TokenContext) -> (String?, Int?) {
        if tag == "ADD" && Lexicon.addSymbols[word] != nil {
            return lookup(Lexicon.addSymbols[word]!, tag: nil, stress: -0.5, context: context)
        } else if let symbol = Lexicon.symbols[word] {
            return lookup(symbol, tag: nil, stress: nil, context: context)
        } else if word.contains(".") && word.trimmingCharacters(in: CharacterSet(charactersIn: ".")).allSatisfy({ $0.isLetter }) {
            let parts = word.components(separatedBy: ".")
            if parts.map({ $0.count }).max() ?? 0 < 3 {
                return getNNP(word)
            }
        } else if word == "a" || word == "A" {
            return tag == "DT" ? ("ɐ", 4) : ("ˈA", 4)
        } else if word == "am" || word == "Am" || word == "AM" {
            if tag?.hasPrefix("NN") == true {
                return getNNP(word)
            } else if context.futureVowel == nil || word != "am" || (stress != nil && stress! > 0) {
                return (golds["am"] as? String, 4)
            }
            return ("ɐm", 4)
        } else if word == "an" || word == "An" || word == "AN" {
            if word == "AN" && tag?.hasPrefix("NN") == true {
                return getNNP(word)
            }
            return ("ɐn", 4)
        } else if word == "I" && tag == "PRP" {
            return ("\(Lexicon.secondaryStress)I", 4)
        } else if (word == "by" || word == "By" || word == "BY") && Lexicon.getParentTag(tag) == "ADV" {
            return ("bˈI", 4)
        } else if word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")) {
            let futureVowel = context.futureVowel
            if futureVowel == nil {
                return (golds["to"] as? String, 4)
            } else if futureVowel == false {
                return ("tə", 4)
            } else {
                return ("tʊ", 4)
            }
        } else if word == "in" || word == "In" || (word == "IN" && tag != "NNP") {
            let stressStr = (context.futureVowel == nil || tag != "IN") ? Lexicon.primaryStress : ""
            return ("\(stressStr)ɪn", 4)
        } else if word == "the" || word == "The" || (word == "THE" && tag == "DT") {
            return context.futureVowel == true ? ("ði", 4) : ("ðə", 4)
        } else if tag == "IN" && word.range(of: "(?i)vs\\.?$", options: .regularExpression) != nil {
            return lookup("versus", tag: nil, stress: nil, context: context)
        } else if word == "used" || word == "Used" || word == "USED" {
            if (tag == "VBD" || tag == "JJ") && context.futureTo {
                if let usedDict = golds["used"] as? [String: String] {
                    return (usedDict["VBD"], 4)
                }
            }
            if let usedDict = golds["used"] as? [String: String] {
                return (usedDict["DEFAULT"], 4)
            }
        }
        
        return (nil, nil)
    }
    
    /// Получение родительского тега (как get_parent_tag в Python)
    public static func getParentTag(_ tag: String?) -> String? {
        guard let tag = tag else { return nil }
        
        if tag.hasPrefix("VB") {
            return "VERB"
        } else if tag.hasPrefix("NN") {
            return "NOUN"
        } else if tag.hasPrefix("ADV") || tag.hasPrefix("RB") {
            return "ADV"
        } else if tag.hasPrefix("ADJ") || tag.hasPrefix("JJ") {
            return "ADJ"
        }
        
        return tag
    }
    
    /// Проверка известности слова (как is_known в Python)
    public func isKnown(_ word: String, tag: String?) -> Bool {
        if golds[word] != nil || Lexicon.symbols[word] != nil || silvers[word] != nil {
            return true
        } else if !word.allSatisfy({ $0.isLetter }) || !word.allSatisfy({ char in Lexicon.lexiconOrds.contains(Int(char.asciiValue ?? 0)) }) {
            return false // TODO: café
        } else if word.count == 1 {
            return true
        } else if word == word.uppercased() && golds[word.lowercased()] != nil {
            return true
        }
        
        return word.dropFirst().allSatisfy({ $0.isUppercase })
    }
    
    /// Основной метод поиска (как lookup в Python)
    public func lookup(_ word: String, tag: String?, stress: Double?, context: TokenContext) -> (String?, Int?) {
        var searchWord = word
        var isNNP: Bool? = nil
        
        if word == word.uppercased() && golds[word] == nil {
            searchWord = word.lowercased()
            isNNP = (tag == "NNP")
        }
        
        var phonemes = golds[searchWord]
        var rating = 4
        
        if phonemes == nil && isNNP != true {
            phonemes = silvers[searchWord]
            rating = 3
        }
        
        if let dict = phonemes as? [String: Any] {
            var lookupTag = tag
            if context.futureVowel == nil && dict["None"] != nil {
                lookupTag = "None"
            } else if dict[tag ?? ""] == nil {
                lookupTag = Lexicon.getParentTag(tag)
            }
            phonemes = dict[lookupTag ?? ""] ?? dict["DEFAULT"]
        }
        
        if let phonemeStr = phonemes as? String {
            if phonemeStr.isEmpty || (isNNP == true && !phonemeStr.contains(Lexicon.primaryStress)) {
                return getNNP(searchWord)
            }
            return (applyStress(phonemeStr, stress: stress), rating)
        } else if isNNP == true {
            return getNNP(searchWord)
        }
        
        return (nil, nil)
    }
    
    /// Метод для обработки токена (основной интерфейс как __call__ в Python)
    public func processToken(_ token: MToken, context: TokenContext) -> (String?, Int?) {
        var word = token.underscore.alias ?? token.text
        word = word.replacingOccurrences(of: "'", with: "'").replacingOccurrences(of: "'", with: "'")
        
        // Нормализация Unicode
        word = word.precomposedStringWithCompatibilityMapping
        
        // Определение ударения по регистру
        let stress: Double? = word == word.lowercased() ? nil : (word == word.uppercased() ? capStresses.1 : capStresses.0)
        
        // Попробуем найти слово
        let (phonemes, rating) = getWord(word, tag: token.tag, stress: stress, context: context)
        
        if let phonemes = phonemes {
            let finalPhonemes = applyStress(appendCurrency(phonemes, currency: token.underscore.currency), 
                                          stress: token.underscore.stress)
            return (finalPhonemes, rating)
        }
        
        // Проверяем числа
        if Lexicon.isNumber(word, isHead: token.underscore.isHead) {
            let (numberPhonemes, numberRating) = getNumber(word, currency: token.underscore.currency, 
                                                         isHead: token.underscore.isHead, numFlags: token.underscore.numFlags)
            if let numberPhonemes = numberPhonemes {
                return (applyStress(numberPhonemes, stress: token.underscore.stress), numberRating)
            }
        }
        
        // Проверяем что все символы в допустимом диапазоне
        if !word.allSatisfy({ char in Lexicon.lexiconOrds.contains(Int(char.asciiValue ?? 0)) }) {
            return (nil, nil)
        }
        
        return (nil, nil)
    }
    
    // MARK: - Stemming methods
    
    /// Добавление окончания -s (как _s в Python)
    private func appendS(_ stem: String) -> String? {
        guard !stem.isEmpty else { return nil }
        
        let lastChar = stem.last!
        if "ptkfθ".contains(lastChar) {
            return stem + "s"
        } else if "szʃʒʧʤ".contains(lastChar) {
            return stem + (british ? "ɪ" : "ᵻ") + "z"
        }
        return stem + "z"
    }
    
    /// Обработка окончания -s (как stem_s в Python)
    public func stemS(_ word: String, tag: String?, stress: Double?, context: TokenContext) -> (String?, Int?) {
        if word.count < 3 || !word.hasSuffix("s") {
            return (nil, nil)
        }
        
        let stem: String
        if !word.hasSuffix("ss") && isKnown(String(word.dropLast()), tag: tag) {
            stem = String(word.dropLast())
        } else if (word.hasSuffix("'s") || (word.count > 4 && word.hasSuffix("es") && !word.hasSuffix("ies"))) && 
                  isKnown(String(word.dropLast(2)), tag: tag) {
            stem = String(word.dropLast(2))
        } else if word.count > 4 && word.hasSuffix("ies") && isKnown(String(word.dropLast(3)) + "y", tag: tag) {
            stem = String(word.dropLast(3)) + "y"
        } else {
            return (nil, nil)
        }
        
        let (stemPhonemes, rating) = lookup(stem, tag: tag, stress: stress, context: context)
        guard let stemPhonemes = stemPhonemes else { return (nil, nil) }
        
        return (appendS(stemPhonemes), rating)
    }
    
    /// Добавление окончания -ed (как _ed в Python)
    private func appendEd(_ stem: String) -> String? {
        guard !stem.isEmpty else { return nil }
        
        let lastChar = stem.last!
        if "pkfθʃsʧ".contains(lastChar) {
            return stem + "t"
        } else if lastChar == "d" {
            return stem + (british ? "ɪ" : "ᵻ") + "d"
        } else if lastChar != "t" {
            return stem + "d"
        } else if british || stem.count < 2 {
            return stem + "ɪd"
        } else if stem.count >= 2 && Lexicon.usTaus.contains(stem.suffix(2).first!) {
            return String(stem.dropLast()) + "ɾᵻd"
        }
        return stem + "ᵻd"
    }
    
    /// Обработка окончания -ed (как stem_ed в Python)
    public func stemEd(_ word: String, tag: String?, stress: Double?, context: TokenContext) -> (String?, Int?) {
        if word.count < 4 || !word.hasSuffix("d") {
            return (nil, nil)
        }
        
        let stem: String
        if !word.hasSuffix("dd") && isKnown(String(word.dropLast()), tag: tag) {
            stem = String(word.dropLast())
        } else if word.count > 4 && word.hasSuffix("ed") && !word.hasSuffix("eed") && 
                  isKnown(String(word.dropLast(2)), tag: tag) {
            stem = String(word.dropLast(2))
        } else {
            return (nil, nil)
        }
        
        let (stemPhonemes, rating) = lookup(stem, tag: tag, stress: stress, context: context)
        guard let stemPhonemes = stemPhonemes else { return (nil, nil) }
        
        return (appendEd(stemPhonemes), rating)
    }
    
    /// Добавление окончания -ing (как _ing в Python)
    private func appendIng(_ stem: String) -> String? {
        guard !stem.isEmpty else { return nil }
        
        if british {
            let lastChar = stem.last!
            if "əː".contains(lastChar) {
                return nil
            }
        } else if stem.count > 1 && stem.suffix(2).first! == "t" && 
                  Lexicon.usTaus.contains(stem.suffix(2).last!) {
            return String(stem.dropLast()) + "ɾɪŋ"
        }
        return stem + "ɪŋ"
    }
    
    /// Обработка окончания -ing (как stem_ing в Python)
    public func stemIng(_ word: String, tag: String?, stress: Double?, context: TokenContext) -> (String?, Int?) {
        if word.count < 5 || !word.hasSuffix("ing") {
            return (nil, nil)
        }
        
        let stem: String
        if word.count > 5 && isKnown(String(word.dropLast(3)), tag: tag) {
            stem = String(word.dropLast(3))
        } else if isKnown(String(word.dropLast(3)) + "e", tag: tag) {
            stem = String(word.dropLast(3)) + "e"
        } else if word.count > 5 && 
                  word.range(of: "([bcdgklmnprstvxz])\\1ing$|cking$", options: .regularExpression) != nil &&
                  isKnown(String(word.dropLast(4)), tag: tag) {
            stem = String(word.dropLast(4))
        } else {
            return (nil, nil)
        }
        
        let useStress = stress == nil ? 0.5 : stress!
        let (stemPhonemes, rating) = lookup(stem, tag: tag, stress: useStress, context: context)
        guard let stemPhonemes = stemPhonemes else { return (nil, nil) }
        
        return (appendIng(stemPhonemes), rating)
    }
    
    // MARK: - Вспомогательные методы
    
    private func getWord(_ word: String, tag: String?, stress: Double?, context: TokenContext) -> (String?, Int?) {
        // Сначала проверяем специальные случаи
        let (specialPhonemes, specialRating) = getSpecialCase(word, tag: tag, stress: stress, context: context)
        if specialPhonemes != nil {
            return (specialPhonemes, specialRating)
        }
        
        // Логика из Python get_word
        let lowercased = word.lowercased()
        var searchWord = word
        
        if word.count > 1 && word.replacingOccurrences(of: "'", with: "").allSatisfy({ $0.isLetter }) && 
           word != word.lowercased() && 
           (tag != "NNP" || word.count > 7) &&
           golds[word] == nil && silvers[word] == nil &&
           (word == word.uppercased() || word.dropFirst().allSatisfy({ $0.isLowercase })) &&
           (golds[lowercased] != nil || silvers[lowercased] != nil || 
            stemS(lowercased, tag: tag, stress: stress, context: context).0 != nil ||
            stemEd(lowercased, tag: tag, stress: stress, context: context).0 != nil ||
            stemIng(lowercased, tag: tag, stress: stress, context: context).0 != nil) {
            searchWord = lowercased
        }
        
        if isKnown(searchWord, tag: tag) {
            return lookup(searchWord, tag: tag, stress: stress, context: context)
        } else if searchWord.hasSuffix("s'") && isKnown(String(searchWord.dropLast(2)) + "'s", tag: tag) {
            return lookup(String(searchWord.dropLast(2)) + "'s", tag: tag, stress: stress, context: context)
        } else if searchWord.hasSuffix("'") && isKnown(String(searchWord.dropLast()), tag: tag) {
            return lookup(String(searchWord.dropLast()), tag: tag, stress: stress, context: context)
        }
        
        // Проверяем stemming
        let (sPhonemes, sRating) = stemS(searchWord, tag: tag, stress: stress, context: context)
        if sPhonemes != nil {
            return (sPhonemes, sRating)
        }
        
        let (edPhonemes, edRating) = stemEd(searchWord, tag: tag, stress: stress, context: context)
        if edPhonemes != nil {
            return (edPhonemes, edRating)
        }
        
        let ingStress = stress ?? 0.5
        let (ingPhonemes, ingRating) = stemIng(searchWord, tag: tag, stress: ingStress, context: context)
        if ingPhonemes != nil {
            return (ingPhonemes, ingRating)
        }
        
        return (nil, nil)
    }
    
    private func appendCurrency(_ phonemes: String, currency: String?) -> String {
        guard let currency = currency,
              let currencyInfo = Lexicon.currencies[currency] else {
            return phonemes
        }
        
        let currencyPhonemes = appendS(currencyInfo.0 + "s")
        return currencyPhonemes != nil ? "\(phonemes) \(currencyPhonemes!)" : phonemes
    }
    
    /// Обработка чисел (как get_number в Python)
    private func getNumber(_ word: String, currency: String?, isHead: Bool, numFlags: String) -> (String?, Int?) {
        // suffix = re.search(r"[a-z']+$", word)
        let suffixRegex = try! NSRegularExpression(pattern: "[a-z']+$")
        let suffixMatch = suffixRegex.firstMatch(in: word, range: NSRange(word.startIndex..., in: word))
        let suffix: String? = suffixMatch.flatMap { Range($0.range, in: word) }.map { String(word[$0]) }
        
        // word = word[:-len(suffix)] if suffix else word
        var processWord = word
        if let suffix = suffix {
            processWord = String(word.dropLast(suffix.count))
        }
        
        var result: [(String?, Int?)] = []
        
        // if word.startswith('-'): result.append(self.lookup('minus', None, None, None)); word = word[1:]
        if processWord.hasPrefix("-") {
            result.append(lookup("minus", tag: nil, stress: nil, context: TokenContext()))
            processWord = String(processWord.dropFirst())
        }
        
        // def extend_num(num, first=True, escape=False):
        func extendNum(_ num: String, first: Bool = true, escape: Bool = false) {
            let numStr: String
            if escape {
                numStr = num
            } else if let intNum = Int(num) {
                numStr = Num2Words.convert(intNum, to: "cardinal")
            } else {
                numStr = num
            }
            
            // splits = re.split(r'[^a-z]+', num if escape else num2words(int(num)))
            let splits = numStr.components(separatedBy: CharacterSet.letters.inverted).filter { !$0.isEmpty }
            
            for (i, w) in splits.enumerated() {
                // if w != 'and' or '&' in num_flags:
                if w != "and" || numFlags.contains("&") {
                    // if first and i == 0 and len(splits) > 1 and w == 'one' and 'a' in num_flags:
                    if first && i == 0 && splits.count > 1 && w == "one" && numFlags.contains("a") {
                        result.append(("ə", 4))
                    } else {
                        let stress: Double? = (w == "point") ? -2 : nil
                        result.append(lookup(w, tag: nil, stress: stress, context: TokenContext()))
                    }
                } else if w == "and" && numFlags.contains("n") && !result.isEmpty {
                    // elif w == 'and' and 'n' in num_flags and result:
                    //     result[-1] = (result[-1][0] + 'ən', result[-1][1])
                    if let last = result.last, let phonemes = last.0 {
                        result[result.count - 1] = (phonemes + "ən", last.1)
                    }
                }
            }
        }
        
        // if is_digit(word) and suffix in ORDINALS:
        if Lexicon.isDigit(processWord) && suffix != nil && Lexicon.ordinals.contains(suffix!) {
            if let intWord = Int(processWord) {
                let ordinalStr = Num2Words.convert(intWord, to: "ordinal")
                extendNum(ordinalStr, escape: true)
            }
        }
        // elif not result and len(word) == 4 and currency not in CURRENCIES and is_digit(word):
        else if result.isEmpty && processWord.count == 4 && 
                (currency == nil || !Lexicon.currencies.keys.contains(currency!)) && 
                Lexicon.isDigit(processWord) {
            if let intWord = Int(processWord) {
                let yearStr = Num2Words.convert(intWord, to: "year")
                extendNum(yearStr, escape: true)
            }
        }
        // elif not is_head and '.' not in word:
        else if !isHead && !processWord.contains(".") {
            let num = processWord.replacingOccurrences(of: ",", with: "")
            if num.first == "0" || num.count > 3 {
                // [extend_num(n, first=False) for n in num]
                for char in num {
                    extendNum(String(char), first: false)
                }
            } else if num.count == 3 && !num.hasSuffix("00") {
                extendNum(String(num.first!))
                if num.dropFirst().first == "0" {
                    result.append(lookup("O", tag: nil, stress: -2, context: TokenContext()))
                    extendNum(String(num.suffix(1)), first: false)
                } else {
                    extendNum(String(num.suffix(2)), first: false)
                }
            } else {
                extendNum(num)
            }
        }
        // elif word.count('.') > 1 or not is_head:
        else if processWord.components(separatedBy: ".").count - 1 > 1 || !isHead {
            var first = true
            for num in processWord.replacingOccurrences(of: ",", with: "").components(separatedBy: ".") {
                if num.isEmpty {
                    continue
                } else if num.first == "0" || (num.count != 2 && num.dropFirst().contains { $0 != "0" }) {
                    for char in num {
                        extendNum(String(char), first: false)
                    }
                } else {
                    extendNum(num, first: first)
                }
                first = false
            }
        }
        // elif currency in CURRENCIES and Lexicon.is_currency(word):
        else if let currency = currency, 
                let currencyInfo = Lexicon.currencies[currency], 
                Lexicon.isCurrency(processWord) {
            
            let parts = processWord.replacingOccurrences(of: ",", with: "").components(separatedBy: ".")
            var pairs: [(Int, String)] = []
            
            for (i, part) in parts.enumerated() {
                let num = Int(part.isEmpty ? "0" : part) ?? 0
                let unit = i == 0 ? currencyInfo.0 : currencyInfo.1
                pairs.append((num, unit))
            }
            
            // Убираем нулевые значения
            if pairs.count > 1 {
                if pairs[1].0 == 0 {
                    pairs = Array(pairs.prefix(1))
                } else if pairs[0].0 == 0 {
                    pairs = Array(pairs.suffix(1))
                }
            }
            
            for (i, (num, unit)) in pairs.enumerated() {
                if i > 0 {
                    result.append(lookup("and", tag: nil, stress: nil, context: TokenContext()))
                }
                extendNum(String(num), first: i == 0)
                
                let finalUnit: String
                if abs(num) != 1 && unit != "pence" {
                    finalUnit = unit + "s"
                    if let stemmed = appendS(unit) {
                        result.append((stemmed, 4))  // Используем stemmed версию
                    } else {
                        result.append(lookup(finalUnit, tag: nil, stress: nil, context: TokenContext()))
                    }
                } else {
                    result.append(lookup(unit, tag: nil, stress: nil, context: TokenContext()))
                }
            }
        }
        // else: (обычные числа)
        else {
            if Lexicon.isDigit(processWord) {
                if let intWord = Int(processWord) {
                    let cardinalStr = Num2Words.convert(intWord, to: "cardinal")
                    extendNum(cardinalStr, escape: true)
                }
            } else if !processWord.contains(".") {
                let cleanNum = processWord.replacingOccurrences(of: ",", with: "")
                if let intNum = Int(cleanNum) {
                    let type = (suffix != nil && Lexicon.ordinals.contains(suffix!)) ? "ordinal" : "cardinal"
                    let numStr = Num2Words.convert(intNum, to: type)
                    extendNum(numStr, escape: true)
                }
            } else {
                let cleanNum = processWord.replacingOccurrences(of: ",", with: "")
                if cleanNum.hasPrefix(".") {
                    var pointStr = "point"
                    for char in cleanNum.dropFirst() {
                        if let digit = Int(String(char)) {
                            pointStr += " " + Num2Words.convert(digit)
                        }
                    }
                    extendNum(pointStr, escape: true)
                } else if let floatNum = Double(cleanNum) {
                    let floatStr = Num2Words.convert(floatNum)
                    extendNum(floatStr, escape: true)
                }
            }
        }
        
        if result.isEmpty {
            return (nil, nil)
        }
        
        // result, rating = ' '.join(p for p, _ in result), min(r for _, r in result)
        let phonemes = result.compactMap { $0.0 }.joined(separator: " ")
        let rating = result.compactMap { $0.1 }.min() ?? 1
        
        // Обработка суффиксов
        if let suffix = suffix {
            if suffix == "s" || suffix == "'s" {
                return (appendS(phonemes), rating)
            } else if suffix == "ed" || suffix == "'d" {
                return (appendEd(phonemes), rating)
            } else if suffix == "ing" {
                return (appendIng(phonemes), rating)
            }
        }
        
        return (phonemes, rating)
    }
    
    /// Проверка является ли слово валютой (как is_currency в Python)
    private static func isCurrency(_ word: String) -> Bool {
        if !word.contains(".") {
            return true
        } else if word.components(separatedBy: ".").count > 2 {
            return false
        }
        
        let cents = word.components(separatedBy: ".")[1]
        return cents.count < 3 || Set(cents).isSubset(of: Set("0"))
    }
    
    private static func isNumber(_ word: String, isHead: Bool) -> Bool {
        if word.allSatisfy({ !$0.isNumber }) {
            return false
        }
        
        let suffixes = ["ing", "'d", "ed", "'s"] + Array(Lexicon.ordinals) + ["s"]
        var checkWord = word
        
        for suffix in suffixes {
            if word.hasSuffix(suffix) {
                checkWord = String(word.dropLast(suffix.count))
                break
            }
        }
        
        return checkWord.enumerated().allSatisfy { (index, char) in
            char.isNumber || ",.".contains(char) || (isHead && index == 0 && char == "-")
        }
    }
}