import Foundation
import Testing

@testable import ExperimentKit

/// The fresh-machine model story on Local (MLX): what the model button DOES
/// for a selection, why a load refuses weights that are not here, and the
/// install state machine that replaces the silent download the Load button
/// used to start. Pure/headless — the fetch itself is injected, so nothing
/// here touches the network or the Hugging Face cache.
@MainActor
struct LocalModelInstallTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.local-model-install.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    // MARK: The button's decision

    @Test func uninstalledSelectionOffersDownloadNotLoad() {
        let action = ChatService.localModelAction(
            selectedModelID: "vendor-a/model-small-4bit",
            isInstalled: false, isLoading: false, isInstalling: false)
        #expect(action == .download("vendor-a/model-small-4bit"))
    }

    @Test func installedSelectionLoads() {
        let action = ChatService.localModelAction(
            selectedModelID: "vendor-a/model-small-4bit",
            isInstalled: true, isLoading: false, isInstalling: false)
        #expect(action == .load("vendor-a/model-small-4bit"))
    }

    /// Neither verb may start while the other is running — the overlap that
    /// made repeated presses race each other.
    @Test func inFlightWorkMakesTheButtonBusy() {
        let loading = ChatService.localModelAction(
            selectedModelID: "vendor-a/model-small-4bit",
            isInstalled: true, isLoading: true, isInstalling: false)
        #expect(loading == .busy(reason: "a model load is already in progress"))

        let installing = ChatService.localModelAction(
            selectedModelID: "vendor-a/model-small-4bit",
            isInstalled: false, isLoading: false, isInstalling: true)
        #expect(installing == .busy(reason: "a model download is already in progress"))
    }

    // MARK: The load refusal

    @Test func loadRefusesUninstalledWeightsAndNamesTheModel() throws {
        let refusal = try #require(
            ChatService.localLoadRefusal(
                modelID: "vendor-a/model-small-4bit", isInstalled: false))
        #expect(refusal.contains("vendor-a/model-small-4bit"))
        #expect(refusal.contains("not installed"))
        #expect(refusal.contains("Load never downloads"))
        #expect(
            ChatService.localLoadRefusal(
                modelID: "vendor-a/model-small-4bit", isInstalled: true) == nil)
    }

    /// The engine-level gate, not just the button's: a `loadModel()` on a
    /// model that is not in the cache reports instead of starting a download,
    /// and leaves the load state alone.
    @Test func loadModelRefusesInsteadOfDownloading() async throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("load-gate"))
        let catalog = SubstrateCatalog(store: store, localCacheScan: { [] })
        let service = ChatService(cluster: store, catalog: catalog)
        service.selectedModelID = "vendor-a/model-small-4bit"

        await service.loadModel()

        #expect(service.state == .unloaded)
        #expect(service.loadedModelID == nil)
        #expect(service.errorMessage?.contains("not installed") == true)
    }

    // MARK: Slug identity

    /// A slug-installed model must keep its own identity. Falling back to the
    /// first pinned tier would load a DIFFERENT model than the selected one,
    /// silently.
    @Test func unknownSlugKeepsItsIdentityInTheSpec() {
        let spec = ChatService.modelSpec(for: "vendor-b/model-large-8bit")
        #expect(spec.id == "vendor-b/model-large-8bit")
        #expect(spec.contextWindow == nil)

        let pinned = ChatService.modelSpec(for: ChatService.availableModels[0].id)
        #expect(pinned.id == ChatService.availableModels[0].id)
        #expect(pinned.contextWindow == ChatService.availableModels[0].contextWindow)
    }

    // MARK: Install refusals (slug shape)

    @Test func installRefusesMalformedSlugs() {
        #expect(
            LocalModelInstaller.installRefusal(slug: "  ", isInstalling: false)?
                .contains("enter a Hugging Face repo id") == true)
        #expect(
            LocalModelInstaller.installRefusal(slug: "model small", isInstalling: false)?
                .contains("no spaces") == true)
        #expect(
            LocalModelInstaller.installRefusal(slug: "model-small-4bit", isInstalling: false)?
                .contains("owner/repo") == true)
        #expect(
            LocalModelInstaller.installRefusal(slug: "vendor-a/", isInstalling: false)?
                .contains("owner/repo") == true)
        #expect(
            LocalModelInstaller.installRefusal(
                slug: "vendor-a/model-small-4bit", isInstalling: false) == nil)
        #expect(
            LocalModelInstaller.installRefusal(
                slug: "vendor-a/model-small-4bit", isInstalling: true)?
                .contains("already running") == true)
    }

    // MARK: The install state machine

    @Test func successfulInstallReportsProgressAndNotifiesTheRegistry() async throws {
        let finished = Mailbox()
        let installer = LocalModelInstaller { _, report in
            report(40)
            report(100)
        }
        installer.onFinished = { modelID in finished.record(modelID) }

        #expect(installer.install("vendor-a/model-small-4bit") == nil)
        try await settle(installer)

        #expect(installer.phase == .finished(modelID: "vendor-a/model-small-4bit"))
        #expect(installer.statusLine?.contains("installed and ready") == true)
        #expect(finished.value == "vendor-a/model-small-4bit")
    }

    @Test func failedInstallKeepsItsReasonOnScreen() async throws {
        struct Refused: LocalizedError {
            var errorDescription: String? { "repository not found" }
        }
        let installer = LocalModelInstaller { _, _ in throw Refused() }

        installer.install("vendor-a/model-small-4bit")
        try await settle(installer)

        #expect(installer.isFailed)
        #expect(installer.statusLine?.contains("repository not found") == true)
        #expect(installer.statusLine?.contains("vendor-a/model-small-4bit") == true)
    }

    /// A malformed slug is reported in the same place a fetch failure is —
    /// a refusal the researcher never sees is a hang with extra steps.
    @Test func malformedSlugSurfacesAsAVisibleFailure() {
        let installer = LocalModelInstaller { _, _ in }
        #expect(installer.install("not-a-slug") != nil)
        #expect(installer.isFailed)
        #expect(installer.statusLine?.contains("owner/repo") == true)
    }

    @Test func cancelStopsTheInstallAndSaysSo() {
        let installer = LocalModelInstaller { _, _ in
            try await Task.sleep(for: .seconds(60))
        }
        installer.install("vendor-a/model-small-4bit")
        #expect(installer.isInstalling)
        #expect(installer.installingModelID == "vendor-a/model-small-4bit")

        installer.cancel()

        #expect(installer.phase == .cancelled(modelID: "vendor-a/model-small-4bit"))
        #expect(installer.statusLine?.contains("cancelled") == true)
        #expect(!installer.isInstalling)
    }

    // MARK: Helpers

    /// Spin the main actor until the install task has left `.installing`.
    private func settle(_ installer: LocalModelInstaller) async throws {
        for _ in 0..<200 {
            if !installer.isInstalling { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("install never settled")
    }

    /// Minimal main-actor recorder for the completion callback.
    @MainActor
    private final class Mailbox {
        private(set) var value: String?
        func record(_ text: String) { value = text }
    }
}
