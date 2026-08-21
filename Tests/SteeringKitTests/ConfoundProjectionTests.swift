import Testing
@testable import SteeringKit

@Suite struct ConfoundProjectionTests {

    /// Rows varying along axes 0 and 1 (axis 0 dominant): the top-2 PCs must
    /// span those axes, be unit-norm, and be mutually orthogonal.
    @Test func deflationFindsOrthogonalComponents() throws {
        let rows: [[Float]] = (0 ..< 12).map { index in
            [
                Float(index) - 5.5,                  // dominant variance
                Float(index % 3) - 1,                // secondary variance
                0, 0,
            ]
        }
        let components = try SteeringVectorMath.principalComponents(of: rows, count: 2)
        #expect(components.count == 2)
        for component in components {
            #expect(abs(SteeringVectorMath.l2Norm(component) - 1) < 1e-4)
        }
        #expect(abs(SteeringVectorMath.dot(components[0], components[1])) < 1e-3)
        #expect(abs(components[0][0]) > 0.99)  // PC1 ≈ axis 0
        #expect(abs(components[1][1]) > 0.99)  // PC2 ≈ axis 1
    }

    @Test func projectingOutRemovesComponent() throws {
        let component: [Float] = [1, 0, 0]
        let vector: [Float] = [3, 4, 5]
        let projected = SteeringVectorMath.projectingOut(vector, components: [component])
        #expect(abs(projected[0]) < 1e-5)
        #expect(projected[1] == 4)
        #expect(projected[2] == 5)
        // Idempotent.
        let again = SteeringVectorMath.projectingOut(projected, components: [component])
        #expect(abs(again[0]) < 1e-5)
    }
}
