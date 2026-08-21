import MLX
import Testing
@testable import SteeringKit

@Suite struct VectorInjectorTests {

    /// Prefill-shaped input: the injection lands on the last position only.
    @Test func injectsAtLastPositionOnly() {
        let injector = VectorInjector(layer: 0, vector: [1, 1, 1, 1], alpha: 2)
        let h = MLXArray.zeros([1, 3, 4])

        let out = injector.apply(h, layer: 0, offset: 0)
        eval(out)

        #expect(out.shape == [1, 3, 4])
        #expect(allClose(out[0, 0, 0...], MLXArray.zeros([4])).item(Bool.self))
        #expect(allClose(out[0, 1, 0...], MLXArray.zeros([4])).item(Bool.self))
        #expect(allClose(out[0, 2, 0...], MLXArray.full([4], values: MLXArray(2.0))).item(Bool.self))
    }

    /// Decode-shaped input (seq length 1): the single position is steered.
    @Test func injectsOnDecodeStep() {
        let injector = VectorInjector(layer: 5, vector: [1, 0, -1, 0], alpha: 3)
        let h = MLXArray.ones([1, 1, 4])

        let out = injector.apply(h, layer: 5, offset: 17)
        eval(out)

        let expected = MLXArray([4.0, 1.0, -2.0, 1.0] as [Float]).reshaped([1, 1, 4])
        #expect(allClose(out, expected).item(Bool.self))
    }

    /// Layers without a configured injection pass through untouched.
    @Test func otherLayersPassThrough() {
        let injector = VectorInjector(layer: 0, vector: [1, 1, 1, 1], alpha: 2)
        let h = MLXArray.ones([1, 2, 4])
        let out = injector.apply(h, layer: 1, offset: 0)
        eval(out)
        #expect(allClose(out, h).item(Bool.self))
    }
}

@Suite struct ActivationRecorderTests {

    @Test func recordsConfiguredLayersAtLastPosition() {
        let recorder = ActivationRecorder(layers: [1])

        let pass1 = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]).reshaped([1, 2, 2])
        _ = recorder.apply(pass1, layer: 0, offset: 0)  // not configured
        _ = recorder.apply(pass1, layer: 1, offset: 0)

        let pass2 = MLXArray([5.0, 6.0] as [Float]).reshaped([1, 1, 2])
        _ = recorder.apply(pass2, layer: 1, offset: 2)

        let captures = recorder.captures
        #expect(captures.count == 2)
        #expect(captures[0].layer == 1)
        #expect(captures[0].offset == 0)
        #expect(captures[0].values == [3.0, 4.0])  // last position of pass 1
        #expect(captures[1].values == [5.0, 6.0])
        #expect(captures[1].offset == 2)

        recorder.reset()
        #expect(recorder.captures.isEmpty)
    }

    /// The recorder must not perturb the forward pass.
    @Test func returnsInputUnchanged() {
        let recorder = ActivationRecorder(layers: [0])
        let h = MLXArray.ones([1, 2, 3])
        let out = recorder.apply(h, layer: 0, offset: 0)
        eval(out)
        #expect(allClose(out, h).item(Bool.self))
    }
}
