import Foundation
import Testing
@testable import ExperimentKit

@MainActor struct ResearcherVectorControlsTests {
    @Test func enablingOneMutedVectorDoesNotReactivateTheOldMix() throws {
        let suite = "experience-controls-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let service = ChatService(cluster: ClusterConnectionStore(defaults: defaults))
        service.slots = [.init(), .init()]
        let first = service.slots[0].id, second = service.slots[1].id
        service.steeringEnabled = false
        service.setVectorEnabled(id: second, enabled: true)
        #expect(service.steeringEnabled)
        #expect(!service.slots[0].enabled && service.slots[1].enabled)
        service.setVectorEnabled(id: first, enabled: true)
        #expect(service.slots[0].enabled && service.slots[1].enabled)
        service.setVectorEnabled(id: second, enabled: false)
        #expect(service.slots[0].enabled && !service.slots[1].enabled)
    }
}
