import Testing
@testable import SteeringKit

struct GenerativeInventoryTests {
    @Test func featureDictionariesAndEncodersAreNotLanguageGenerators() {
        #expect(!SteeredContainerLoader.isGenerativeModelConfiguration(["d_sae": 16384, "model_name": "a-model"]))
        #expect(!SteeredContainerLoader.isGenerativeModelConfiguration(["model_type": "bert", "architectures": ["BertForSequenceClassification"]]))
        #expect(!SteeredContainerLoader.isGenerativeModelConfiguration(["peft_type": "LORA"]))
        #expect(SteeredContainerLoader.isGenerativeModelConfiguration(["architectures": ["Qwen3ForCausalLM"]]))
        #expect(SteeredContainerLoader.isGenerativeModelConfiguration(["model_type": "gemma3", "text_config": ["vocab_size": 100]]))
    }
}
