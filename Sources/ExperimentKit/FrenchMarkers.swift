import Foundation

/// Crude keyword metric for the toy concept only: counts French function
/// words and accented characters. Concept-specific scoring lives in the
/// experiment layer, never in SteeringKit. Real experiments use the
/// validated rubrics from Phase 2, not this.
enum FrenchMarkers {

    private static let functionWords: Set<String> = [
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "est",
        "dans", "que", "qui", "je", "tu", "il", "elle", "nous", "vous",
        "ne", "pas", "pour", "avec", "sur", "mais", "plus", "très", "être",
        "avoir", "fait", "comme", "tout", "bien", "aussi", "votre", "mon",
    ]

    private static let accentedCharacters = Set("àâäéèêëîïôöùûüçœÀÂÉÈÊËÎÏÔÙÛÇŒ")

    static func count(in text: String) -> Int {
        let words = text.lowercased()
            .components(separatedBy: CharacterSet.letters.inverted)
            .filter { !$0.isEmpty }
        let wordHits = words.count(where: { functionWords.contains($0) })
        let accentHits = text.count(where: { accentedCharacters.contains($0) })
        return wordHits + accentHits
    }
}
