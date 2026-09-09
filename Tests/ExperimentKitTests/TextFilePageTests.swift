import Foundation
import Testing
@testable import ExperimentKit

struct TextFilePageTests {
    @Test func pagesPreserveEveryByteAcrossLongUnicodeRecords() throws {
        let url = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let original = String(repeating: "{\"text\":\"a😀é中\"}\n", count: 200) + "malformed final row"
        try original.write(to: url, atomically: true, encoding: .utf8)
        var offset: UInt64 = 0
        var joined = ""
        repeat {
            let page = try TextFilePage.read(url: url, offset: offset, byteLimit: 17)
            #expect(page.nextOffset > offset)
            joined += page.text; offset = page.nextOffset
            if !page.hasMore { break }
        } while true
        #expect(joined == original)
    }

    @Test func emptyFileAndMissingFileAreDistinct() throws {
        let url = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) { try TextFilePage.read(url: url) }
        try Data().write(to: url)
        let page = try TextFilePage.read(url: url)
        #expect(page.text.isEmpty && !page.hasMore && page.nextOffset == 0)
    }
}
