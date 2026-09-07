import Foundation

extension RepEReader {
    public static let evidenceRoleNote =
        "heldOutAccuracy is a MODEL-SELECTION statistic: the same held-out rows "
        + "chose the direction's sign and ranked the layers, so it is not an "
        + "untouched final-test estimate. Final generalization evidence is "
        + "finalTestAccuracy, scored on rows marked split 'finalTest' that no "
        + "fitting or selection step read. When finalTestAccuracy is absent the "
        + "dataset reserved no such rows, and the result must be reported as "
        + "selection/validation evidence, not as a final test."

    public struct EvidenceRoles: Codable, Sendable, Equatable {
        public var trainAccuracy: String = "fit"
        public var heldOutAccuracy: String?
        public var finalTestAccuracy: String?
        public var signSelectedBy: String
        public var layerRecommendedBy: String?
        public var splitCounts: [String: Int]

        enum CodingKeys: String, CodingKey {
            case trainAccuracy, heldOutAccuracy, finalTestAccuracy
            case signSelectedBy, layerRecommendedBy, splitCounts
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(trainAccuracy, forKey: .trainAccuracy)
            try c.encodeIfPresent(heldOutAccuracy, forKey: .heldOutAccuracy)
            try c.encodeIfPresent(finalTestAccuracy, forKey: .finalTestAccuracy)
            try c.encode(signSelectedBy, forKey: .signSelectedBy)
            try c.encode(layerRecommendedBy, forKey: .layerRecommendedBy)
            try c.encode(splitCounts, forKey: .splitCounts)
        }

        public func validate() throws {
            if let heldOutAccuracy, !["selection", "validation"].contains(heldOutAccuracy) {
                throw ReaderError(reason: "heldOutAccuracy describes selection/validation; use finalTestAccuracy for final evaluation.")
            }
            if let finalTestAccuracy, finalTestAccuracy != "finalEvaluation" {
                throw ReaderError(reason: "finalTestAccuracy must carry the finalEvaluation role.")
            }
        }
    }

    /// Match the Python exact-text check; neither near-duplicates nor held-out
    /// versus final-test overlap are assessed by that contract.
    public static func checkSplitOverlap(_ dataset: Dataset, source: String = "reader dataset") throws {
        func normalized(_ text: String) -> String {
            // Python str.split includes the information separators as well
            // as Unicode White_Space. Preserve normalization form: Python's
            // casefold does not equate composed and decomposed spellings.
            text.unicodeScalars.split {
                $0.properties.isWhitespace || (0x1c ... 0x1f).contains($0.value)
            }.map { String(String.UnicodeScalarView($0)) }.joined(separator: " ")
                .folding(options: .caseInsensitive, locale: Locale(identifier: "en_US_POSIX"))
        }
        func key(_ pair: Pair) -> Data {
            if let stimulus = pair.stimulus { return Data(normalized(stimulus).utf8) }
            return Data((normalized(pair.positiveStimulus) + "\0" + normalized(pair.negativeStimulus)).utf8)
        }
        var train: [Data: String] = [:]
        for (index, pair) in dataset.pairs.enumerated() where pair.split.lowercased() == "train" {
            train[key(pair)] = train[key(pair)] ?? (pair.id ?? "#\(index + 1)")
        }
        for (index, pair) in dataset.pairs.enumerated() where pair.split.lowercased() != "train" {
            if let original = train[key(pair)] {
                throw ReaderError(reason: "\(source): row \(pair.id ?? "#\(index + 1)") (split \(pair.split)) has the same stimulus text as train row \(original). Repair: rewrite or drop the duplicate so held-out/final-test evidence uses distinct stimuli.")
            }
        }
    }
}

extension RepEReader.Artifact {
    /// A reading of legacy stamps, not a retroactively written check/measurement.
    public var resolvedEvidenceRoles: RepEReader.EvidenceRoles {
        if let evidenceRoles { return evidenceRoles }
        let sign = signConvention == .heldOutPairAgreement ? "heldOut" : "train"
        let layer: String? = layerRecommendationBasis == "heldOutAccuracy" ? "heldOut"
            : (layerRecommendationBasis == "trainAccuracy" ? "train" : nil)
        return .init(
            heldOutAccuracy: heldOutAccuracy == nil ? nil : (sign == "heldOut" || layer == "heldOut" ? "selection" : "validation"),
            finalTestAccuracy: finalTestAccuracy == nil ? nil : "finalEvaluation",
            signSelectedBy: sign, layerRecommendedBy: layer,
            splitCounts: ["train": trainPairCount, "heldOut": heldOutPairCount, "finalTest": finalTestPairCount ?? 0])
    }
}
