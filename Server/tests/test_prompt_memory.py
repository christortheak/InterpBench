"""The submission's memory model (issue #6), against hand-written arithmetic.

Every expected value below is worked out by hand in its comment, from the
model's geometry, independently of ``prompt_memory``. The 27B hybrid is the
handoff's: 64 layers, 16 of them full attention with 4 key-value heads of 256
dimensions, 48 linear-attention (gated delta rule) layers with 48 value heads
and 16 key heads of 128 dimensions, a 5,120-wide residual, and 51.75 GiB of
bfloat16 weights. Its query heads (24) and MLP width (17,408) are the
Qwen3.5-27B pattern's; the handoff's "one pass of the MLP over 217,000
tokens holds about 21 GiB" is 3 × 217,000 × 17,408 × 2 bytes = 21.1 GiB.
"""

import pytest

from steerlab_server.api import prompt_memory as pm
from steerlab_server.experiment import prompt_reading as pr

GiB = 1024 ** 3

HYBRID = {"num_hidden_layers": 64, "hidden_size": 5120,
          "intermediate_size": 17408, "num_attention_heads": 24,
          "num_key_value_heads": 4, "head_dim": 256,
          "linear_num_value_heads": 48, "linear_num_key_heads": 16,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4, "full_attention_interval": 4}
WEIGHTS = 55_566_139_392          # 51.75 × 1024³
USABLE_A100 = 85_094_039_552      # 79.25 × 1024³, what CUDA reports for an 80 GB A100
# One linear layer's fixed state: 48 × 128 × 128 float32 recurrent state
# (3,145,728 bytes) plus a (2 × 16 × 128 + 48 × 128) = 10,240-wide bfloat16
# convolution window of kernel 4 − 1 = 3 steps (61,440 bytes) = 3,207,168.
LINEAR_STATE = 48 * 3_207_168     # 153,944,064 bytes = 0.143 GiB ("0.14 GiB")


# --- the cache, by layer type ------------------------------------------------------

def test_the_hybrid_keeps_64_kib_per_token_on_16_of_64_layers():
    # 16 layers × 2 (K and V) × 4 heads × 256 dims × 2 bytes = 65,536 bytes.
    # At 217,230 + 8,192 = 225,422 tokens: 225,422 × 65,536 =
    # 14,773,256,192 bytes = 13.76 GiB, the handoff's "13.8 GiB".
    priced = pm.cache_bytes(HYBRID, 225_422)
    assert priced["perTokenBytes"] == 65_536
    assert priced["byLayerType"]["fullAttention"] == {
        "layers": 16, "bytes": 14_773_256_192}
    assert priced["byLayerType"]["linearAttention"]["layers"] == 48
    assert priced["byLayerType"]["linearAttention"]["bytes"] == LINEAR_STATE
    assert priced["bytes"] == 14_773_256_192 + 153_944_064
    assert priced["layerTypesBasis"] == "fullAttentionInterval"


def test_declared_layer_types_win_and_the_nested_text_config_is_read():
    types = (["linear_attention"] * 3 + ["full_attention"]) * 16
    config = {"text_config": {**HYBRID, "layer_types": types,
                              "full_attention_interval": 2}}
    priced = pm.cache_bytes(config, 225_422)
    assert priced["layerTypesBasis"] == "declared"
    assert priced["byLayerType"]["fullAttention"]["layers"] == 16
    assert priced["bytes"] == 14_773_256_192 + 153_944_064


def test_sliding_layers_are_charged_for_at_most_their_window():
    # Gemma 3's pattern: every 6th layer full, the rest sliding. 12 layers →
    # layers 6 and 12 full, 10 sliding with a 1,024-token window. Per layer
    # per token: 2 × 8 heads × 128 dims × 2 bytes = 4,096 bytes.
    # Full: 2 × 4,096 × 10,000 = 81,920,000. Sliding: 10 × 4,096 × 1,024 =
    # 41,943,040. Total 123,863,040.
    config = {"num_hidden_layers": 12, "num_key_value_heads": 8,
              "num_attention_heads": 16, "head_dim": 128,
              "sliding_window": 1024, "sliding_window_pattern": 6}
    priced = pm.cache_bytes(config, 10_000)
    assert priced["byLayerType"]["fullAttention"] == {"layers": 2,
                                                     "bytes": 81_920_000}
    assert priced["byLayerType"]["slidingAttention"]["bytes"] == 41_943_040
    assert priced["bytes"] == 123_863_040
    # A context shorter than the window charges sliding layers its length:
    # 12 × 4,096 × 500 = 24,576,000.
    assert pm.cache_bytes(config, 500)["bytes"] == 24_576_000


def test_an_all_full_model_is_priced_as_before():
    # 4 layers × 2 × 2 heads × 64 dims × 2 bytes × 1,000 tokens = 2,048,000,
    # the arithmetic the preflight always used.
    config = {"num_hidden_layers": 4, "num_key_value_heads": 2,
              "num_attention_heads": 4, "hidden_size": 256}
    priced = pm.cache_bytes(config, 1_000)
    assert priced["bytes"] == 2_048_000
    assert priced["layerTypesBasis"] == "assumedFull"


def test_an_unknown_layer_type_is_priced_as_full_and_named():
    config = {"num_hidden_layers": 2, "num_key_value_heads": 2,
              "head_dim": 64, "layer_types": ["full_attention", "hybrid"]}
    priced = pm.cache_bytes(config, 1_000)
    assert priced["byLayerType"]["fullAttention"]["layers"] == 2
    assert priced["pricedAsFull"] == ["hybrid"]


def test_a_linear_state_without_geometry_takes_the_stated_constant():
    config = {"num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 64,
              "layer_types": ["linear_attention", "full_attention"]}
    priced = pm.cache_bytes(config, 1_000)
    linear = priced["byLayerType"]["linearAttention"]
    assert linear["stateBasis"] == "statedConstant"
    assert linear["bytes"] == 8 * 1024 ** 2


# --- the handoff's two measured cases --------------------------------------------

def test_the_82398_token_single_pass_is_predicted_not_to_fit_an_a100():
    # Measured: OOM at 72.39 GiB allocated, asking 1.89 GiB more, on an
    # A100-SXM4-80GB reporting 79.25 GiB. The model's prompt pass:
    #   residual  4 × 82,398 × 5,120 × 2          =  3,375,022,080
    #   linear   10 × 82,398 × 48 × 128 × 4       = 20,250,132,480
    #   (MLP 3 × 82,398 × 17,408 × 2 = 8,606,306,304 is smaller;
    #    no mask in a causal single pass)
    #   working memory                             = 23,625,154,560
    #   prompt cache 82,398 × 65,536 + 153,944,064 =  5,553,979,392
    #   peak = 55,566,139,392 + 5,553,979,392 + 23,625,154,560
    #        = 84,745,273,344, + 0.5 GiB context = 85,282,144,256
    # against 79.25 GiB = 85,094,039,552 usable: over by 188,104,704
    # (0.18 GiB), so the prediction is right but narrow.
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=82_398,
                         generation_tokens=8_192, rule={"mode": "singlePass"})
    assert priced["workingMemory"]["residualBytes"] == 3_375_022_080
    assert priced["workingMemory"]["linearAttentionFallbackBytes"] == \
        20_250_132_480
    assert priced["workingMemory"]["mlpBytes"] == 8_606_306_304
    assert priced["workingMemory"]["bytes"] == 23_625_154_560
    assert priced["workingMemory"]["maskBytes"] == 0
    assert priced["promptPassPeakBytes"] == 84_745_273_344
    assert priced["estimatedBytes"] == 85_282_144_256
    verdict = pm.fit(priced, gpu_type="A100", vram_table={"A100": 80})
    assert verdict["usableBytes"] == USABLE_A100
    assert verdict["memoryFit"] == "mayNotFit"


def test_the_217230_token_chunked_read_is_predicted_to_fit_an_a100():
    # Measured: completed, peaking at about 71 GiB allocated (69.35 GiB plus a
    # 1.66 GiB mask request) of 79.25. The model at chunk 4,096:
    #   mask      4,096 × 217,230 × 3                     =  2,669,322,240
    #   residual  4 × 4,096 × 5,120 × 2                   =    167,772,160
    #   attention projections 4,096 × (48 + 8) × 256 × 2  =    117,440,512
    #             repeated K, V 2 × 24 × 256 × 217,230 × 2 =  5,338,644,480
    #             cache growth  2 × 4 × 256 × 217,230 × 2 =    889,774,080
    #   working memory = 2,669,322,240 + 167,772,160 + 6,345,859,072
    #                  = 9,182,953,472
    #   prompt cache 217,230 × 65,536 + 153,944,064 = 14,390,329,344
    #   peak = 55,566,139,392 + 14,390,329,344 + 9,182,953,472
    #        = 79,139,422,208 (73.70 GiB), + 0.5 GiB = 79,676,293,120,
    # which leaves 5,417,746,432 bytes (5.0 GiB) of the 79.25 GiB.
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=217_230,
                         generation_tokens=8_192, rule=pr.CUDA_DEFAULT)
    assert priced["chunkTokens"] == 4096
    assert priced["workingMemory"]["maskBytes"] == 2_669_322_240
    assert priced["workingMemory"]["attentionBytes"] == 6_345_859_072
    assert priced["workingMemory"]["bytes"] == 9_182_953_472
    assert priced["promptPassPeakBytes"] == 79_139_422_208
    assert priced["estimatedBytes"] == 79_676_293_120
    # Certain: 55,566,139,392 + 14,773,256,192 + 153,944,064 =
    # 70,493,339,648 = 65.65 GiB, the handoff's "65.6 GiB".
    assert priced["certainBytes"] == 70_493_339_648
    verdict = pm.fit(priced, gpu_type="A100", vram_table={"A100": 80})
    assert verdict["memoryFit"] == "fits"
    assert USABLE_A100 - priced["estimatedBytes"] == 5_417_746_432


def test_a_reasoning_budget_grows_the_certain_part():
    # 217,230 + 8,192 + 32,768 = 258,190 tokens × 65,536 = 16,920,739,840;
    # + 55,566,139,392 weights + 153,944,064 linear state = 72,640,823,296
    # = 67.65 GiB, the handoff's "67.6 GiB".
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=217_230,
                         generation_tokens=8_192 + 32_768,
                         rule=pr.CUDA_DEFAULT)
    assert priced["certainBytes"] == 72_640_823_296


# --- certain refuses, the estimate advises --------------------------------------

def test_the_certain_part_alone_over_the_card_exceeds():
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=217_230,
                         generation_tokens=8_192, rule=pr.CUDA_DEFAULT)
    verdict = pm.fit(priced, gpu_type="G64", vram_table={"G64": 64, "A100": 80})
    assert verdict["memoryFit"] == "exceeds"
    assert verdict["certainFittingGPUTypes"] == ["A100"]


def test_without_a_declared_size_the_capacity_is_unknown():
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=1000,
                         generation_tokens=100, rule=None)
    assert pm.fit(priced, gpu_type="Z9", vram_table={})["memoryFit"] == \
        "capacityUnknown"


def test_a_config_without_the_working_geometry_keeps_the_flat_allowance():
    # The old preflight's rule: (weights + cache) × 1.2.
    config = {"num_hidden_layers": 2, "num_key_value_heads": 2,
              "num_attention_heads": 4, "head_dim": 8}
    priced = pm.estimate(config, weights_bytes=4096, prompt_tokens=2048,
                         generation_tokens=512, rule=None)
    assert priced["estimateBasis"] == "flatHeadroom"
    assert priced["certainBytes"] == 4096 + 327_680
    assert priced["estimatedBytes"] == int((4096 + 327_680) * 1.2)


# --- choosing the reading ---------------------------------------------------------

def _choose(vram, declared=None, prompt=217_230):
    return pm.choose_reading(HYBRID, weights_bytes=WEIGHTS,
                             prompt_tokens=prompt, generation_tokens=8_192,
                             declared=declared, gpu_type="G",
                             vram_table={"G": vram})


def test_the_default_is_kept_where_it_fits():
    choice = _choose(80)
    assert choice["rule"] == pr.CUDA_DEFAULT
    assert not choice["steppedDown"]


def test_the_largest_smaller_chunk_that_fits_is_chosen():
    # A declared 74 GB card leaves 73.31 GiB usable: the default's 74.20 GiB
    # estimate does not fit, and chunk 2,048 (72.83 GiB) does.
    choice = _choose(74)
    assert choice["rule"] == {"mode": "chunked", "chunkTokens": 2048,
                              "aboveTokens": 8192}
    assert choice["steppedDown"]
    assert [t["memoryFit"] for t in choice["tried"]] == ["mayNotFit", "fits"]


def test_when_nothing_fits_the_default_is_kept_and_only_advised():
    choice = _choose(71)
    assert choice["rule"] == pr.CUDA_DEFAULT
    assert not choice["steppedDown"]
    assert choice["fit"]["memoryFit"] == "mayNotFit"
    assert len(choice["tried"]) == 4


def test_a_declaration_is_checked_and_never_changed():
    declared = {"mode": "singlePass"}
    choice = _choose(80, declared=declared)
    assert choice["rule"] == declared
    assert choice["fit"]["memoryFit"] == "mayNotFit"
    assert not choice["steppedDown"]
    assert len(choice["tried"]) == 1


def test_a_short_prompt_never_steps_down():
    # 8,000 tokens is read in one pass under the default and every step-down
    # alike (all keep aboveTokens 8,192), so there is nothing to step to.
    choice = _choose(57, prompt=8_000)
    assert choice["rule"] == pr.CUDA_DEFAULT
    assert len(choice["tried"]) == 1


@pytest.mark.parametrize("tokens,expected", [
    (8_192, None), (8_193, 4096), (4_096, None)])
def test_the_rule_chunks_above_the_larger_of_its_two_numbers(tokens, expected):
    assert pr.chunk_size(pr.CUDA_DEFAULT, tokens) == expected
    assert pr.chunk_size({"mode": "chunked", "chunkTokens": 9000,
                          "aboveTokens": 0}, 8_999) is None


# --- a shorter prompt read in one pass can be the peak (review, item 2) ----------

def test_the_longest_single_pass_the_rule_allows_is_priced_too():
    # Declared: chunks of 4,096 above 100,000 tokens; the longest prompt is
    # 150,000 tokens (chunked), but a 100,000-token prompt would be read in
    # ONE pass. Hand arithmetic:
    #   single pass of 100,000: 4 × 100,000 × 5,120 × 2 + 10 × 100,000 × 48
    #     × 128 × 4 = 28,672,000,000 of working memory; peak 55,566,139,392
    #     + 100,000 × 65,536 + 153,944,064 + 28,672,000,000 = 90,945,683,456,
    #     + 0.5 GiB = 91,482,554,368 (85.2 GiB)
    #   the chunked 150,000-token pass alone: 72,516,567,040 (67.5 GiB)
    rule = {"mode": "chunked", "chunkTokens": 4096, "aboveTokens": 100_000}
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=150_000,
                         generation_tokens=8_192, rule=rule)
    assert priced["singlePassBelowThreshold"]["promptTokens"] == 100_000
    assert priced["singlePassBelowThreshold"]["peakBytes"] == 90_945_683_456
    assert priced["peakFrom"] == "singlePassBelowThreshold"
    assert priced["estimatedBytes"] == 91_482_554_368
    assert priced["certainBytes"] == 66_087_354_368
    assert pm.fit(priced, gpu_type="A100",
                  vram_table={"A100": 80})["memoryFit"] == "mayNotFit"


def test_the_default_readings_short_pass_does_not_move_the_handoff_case():
    # Under the CUDA default the longest single pass is 8,192 tokens, far
    # below the chunked 217,230-token pass: the estimate is unchanged.
    priced = pm.estimate(HYBRID, weights_bytes=WEIGHTS, prompt_tokens=217_230,
                         generation_tokens=8_192, rule=pr.CUDA_DEFAULT)
    assert "peakFrom" not in priced
    assert priced["estimatedBytes"] == 79_676_293_120


# --- refuse only on what the configuration states (review, item 4) ---------------

DENSE = {"num_hidden_layers": 4, "num_key_value_heads": 2,
         "num_attention_heads": 4, "hidden_size": 256}


def test_a_dense_model_without_layer_types_still_refuses_over_the_card():
    priced = pm.estimate(DENSE, weights_bytes=2 * GiB, prompt_tokens=1_000,
                         generation_tokens=100, rule=None)
    assert priced["cache"]["assumptions"] == []
    assert pm.fit(priced, gpu_type="S", vram_table={"S": 1})["memoryFit"] == "exceeds"


@pytest.mark.parametrize("hint,assumption", [
    ({"hybrid_override_pattern": "M-M*-M"}, "carries hybrid settings"),
    ({"layer_types": ["full_attention", "hybrid", "full_attention",
                      "full_attention"]}, "are not ones this check knows"),
    ({"layer_types": ["full_attention", "linear_attention", "full_attention",
                      "full_attention"]}, "stated 8 MiB"),
])
def test_an_over_price_resting_on_an_assumption_is_advice(hint, assumption):
    priced = pm.estimate({**DENSE, **hint}, weights_bytes=2 * GiB,
                         prompt_tokens=1_000, generation_tokens=100, rule=None)
    assert any(assumption in a for a in priced["cache"]["assumptions"])
    verdict = pm.fit(priced, gpu_type="S", vram_table={"S": 1})
    assert verdict["memoryFit"] == "mayNotFit"
    assert verdict["certainByAssumption"] is True


def test_a_jamba_period_and_offset_are_priced_layer_by_layer():
    # 8 layers, an attention layer where i % 4 == 1 (layers 1 and 5), the
    # other 6 Mamba layers at the stated 8 MiB. At 1,000 tokens: 2 × 2 × 8
    # heads × 128 dims × 2 bytes × 1,000 + 6 × 8 MiB = 58,523,648.
    config = {"num_hidden_layers": 8, "num_key_value_heads": 8,
              "num_attention_heads": 32, "head_dim": 128,
              "attn_layer_period": 4, "attn_layer_offset": 1}
    priced = pm.cache_bytes(config, 1_000)
    assert priced["layerTypesBasis"] == "attentionLayerPeriod"
    assert priced["byLayerType"]["fullAttention"]["layers"] == 2
    assert priced["bytes"] == 58_523_648


def test_a_layers_block_type_list_is_read():
    config = {**DENSE, "layers_block_type": ["mamba", "attention", "mamba",
                                             "attention"],
              "linear_num_value_heads": 8, "linear_key_head_dim": 64,
              "linear_value_head_dim": 64}
    priced = pm.cache_bytes(config, 1_000)
    assert priced["layerTypesBasis"] == "layersBlockType"
    assert priced["byLayerType"]["fullAttention"]["layers"] == 2
    assert priced["byLayerType"]["linearAttention"]["layers"] == 2
    assert priced["assumptions"] == []
