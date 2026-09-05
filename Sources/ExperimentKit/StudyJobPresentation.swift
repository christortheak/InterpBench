import Foundation
import Observation
import SteeringKit

/// UI effects for job controllers. No controller needs an ExperimentPanel.
@MainActor
struct StudyJobPresentation {
    var note: (String, PanelNotice.Severity) -> Void = { _, _ in }
    var status: (String?) -> Void = { _ in }
    var refresh: () -> Void = {}
    var selectResult: (String, String) -> Void = { _, _ in }
    var startLog: (String, String) -> UUID? = { _, _ in nil }
    var updateLog: (UUID, String, [String]) -> Void = { _, _, _ in }
}

/// A bounded display log with an explicit presentation sink.
@MainActor
final class StudyDisplayLog {
    private var id: UUID?
    private var title = ""
    private var lines: [String] = []
    var presentation = StudyJobPresentation()

    func begin(title: String, initialLine: String) {
        self.title = title
        lines = [initialLine]
        id = presentation.startLog(title, initialLine)
    }
    func append(_ line: String) {
        guard let id else { return }
        lines.append(line)
        if lines.count > 400 { lines.removeFirst(lines.count - 400) }
        presentation.updateLog(id, title, lines)
    }
    func end(_ line: String? = nil) {
        if let line { append(line) }
        id = nil
        lines = []
    }
}
