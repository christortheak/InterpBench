import Foundation
import Testing

@testable import ExperimentKit

struct CredentialObservationTests {
    @Test func missingEnvironmentIsUnknownRatherThanAnAbsentStoredCredential() throws {
        #expect(CredentialObservation.anthropic(environment: [:]) == .notChecked)
        #expect(CredentialObservation.openRouter(environment: ["OPENROUTER_API_KEY": ""]) == .notChecked)
        #expect(CredentialObservation.openRouter(environment: ["ANTHROPIC_API_KEY": "fixture-only"]) == .notChecked)
        let state = CredentialObservation.openRouter(environment: ["OPENROUTER_API_KEY": "fixture-only"])
        #expect(state == .available)
        #expect(CredentialObservation.State.notChecked.knownPresence == nil)
        #expect(CredentialObservation.State.absent.knownPresence == false)
        #expect(String(decoding: try JSONEncoder().encode(state), as: UTF8.self) == "\"available\"")
    }

    @Test func uncheckedCredentialsDoNotHideAuthoringChoicesOrClaimReadiness() {
        let selected = JudgeModelSpelling.spellOpenRouter(model: "example/model", provider: "provider")
        let offers = JudgeModelOffers.compose(selected: selected, candidates: [],
            openRouterCredentialState: .notChecked, installed: { _ in false })
        #expect(offers.openRouter.contains { $0.id == JudgeModelOffers.openRouterSentinel })
        #expect(offers.openRouter.contains { $0.id == selected })
        #expect(offers.openRouterHint == CredentialObservation.deferredCheckMessage)
        #expect(offers.selectionCaption != nil)
        #expect(!offers.openRouterHint!.contains("No OpenRouter key"))
        let absent = JudgeModelOffers.compose(selected: "", candidates: [],
            openRouterCredentialState: .absent, installed: { _ in false })
        #expect(absent.openRouter.isEmpty)
        #expect(absent.openRouterHint == JudgeModelOffers.openRouterKeyHint)
    }

    @Test func structuralRefusalsStillWinOverDeferredCredentialChecks() {
        let selected = "example/local-model"
        let offers = JudgeModelOffers.compose(selected: selected, candidates: [],
            openRouterCredentialState: .notChecked, installed: { _ in false })
        let explicit = JudgeModelOffers.compose(selected: selected, candidates: [],
            openRouterKeyPresent: true, installed: { _ in false })
        #expect(offers.selectionCaption == explicit.selectionCaption)
        #expect(offers.selectionCaption != CredentialObservation.deferredCheckMessage)
        #expect(offers.models.first { $0.id == selected }?.caption == explicit.models.first { $0.id == selected }?.caption)
    }
}
