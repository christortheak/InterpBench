import Testing
@testable import ExperimentKit

@Suite struct FrenchMarkersTests {

    @Test func countsFunctionWordsAndAccents() {
        // "le", "est", "très" (word + accent), "à" (accent only)
        let french = "Le café est très bon à Paris."
        let english = "The coffee shop on Main Street is very good."
        #expect(FrenchMarkers.count(in: french) > FrenchMarkers.count(in: english))
    }

    @Test func englishProseScoresNearZero() {
        let english = "I wake up at seven, make coffee, and read the news before work."
        #expect(FrenchMarkers.count(in: english) == 0)
    }
}
