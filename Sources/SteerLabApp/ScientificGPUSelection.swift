import ExperimentKit
import SwiftUI

struct ScientificGPUSelection: View {
    let options: ScientificGPUPlacement
    @Binding var selection: String
    var title = "GPU for this submission"
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker(title, selection: $selection) {
                Text(options.defaultLabel).tag("")
                ForEach(options.gpuTypes, id: \.self) { Text($0).tag($0) }
            }
            .help("Choose from this controller’s declared GPU types. Changing the selection requires a new plan; it does not edit the study or fitting request.")
            Text(options.capacityLabel(selection)).font(.caption)
        }
    }
}
