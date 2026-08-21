"""Request/response schemas (versioned DTOs).

The Swift ``WebServer`` used inline ``struct Body: Decodable`` per route and a
``StateDTO`` snapshot. Here those are promoted to named pydantic models so the
contract is explicit and the later native-Mac remote client has a stable schema
to target. Field names follow the Swift JSON (camelCase) where a route already
exists, so the web UI can be pointed here with minimal change.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    model: str
    revision: str | None = None
    dtype: str = "auto"          # auto = bf16 (CUDA) / fp16 (MPS) / fp32 (CPU)
    device: str | None = None    # auto = CUDA → MPS (Apple GPU) → CPU


class InjectionDTO(BaseModel):
    """One steering cell. Either supply an explicit ``vector`` (engine-level), or
    reference a saved artifact by ``vectorPath`` + ``name`` to load it."""

    layer: int
    #: α when steering, λ when ablating.
    alpha: float
    #: "add" (steer, the default) or "ablate". Absent means add — the same
    #: absent-means-default convention the manifest and variant artifacts use,
    #: so a steering request's body is unchanged.
    mode: str = "add"
    #: Names the concept; orders the ablation basis when several directions are
    #: ablated at one layer (Gram-Schmidt is order-dependent).
    concept: str = ""
    #: Ablation-direction centering, artifact-referenced ablations only:
    #: "none" (default) or "neutralMean" — project the artifact's stored
    #: neutral residual mean out of the direction before ablating (extracted
    #: vectors share a large component with that mean; ablating it raw at λ=1
    #: collapses generation). Explicit-vector cells must center client-side —
    #: the server has no mean to apply, so a non-"none" value there refuses.
    centering: str = "none"
    vector: list[float] | None = None
    vectorPath: str | None = None      # run directory holding <name>.safetensors
    name: str | None = None            # artifact base name within vectorPath


class GenerateRequest(BaseModel):
    # Either a single `text` prompt, or a multi-turn `messages` transcript
    # ([{role, content, seeded?, edited?}, ...]). When messages are present
    # they take precedence, so remote chat sees the full conversation, not
    # just the last turn. `seeded: true` marks a researcher-authored
    # assistant turn (send-as-assistant); `edited: true` marks a generated
    # turn the researcher altered afterwards. Both are provenance for the
    # request record only and never change rendering — a seeded/edited turn
    # is byte-identical to a real one in the prompt the model sees. Messages
    # are plain dicts, so unknown provenance keys from newer clients are
    # tolerated by construction.
    text: str = ""
    messages: list[dict] | None = None
    maxTokens: int = 512
    temperature: float = 0.0
    promptMode: str = "chatAssistant"
    systemPrompt: str | None = None
    qwenThinkingEnabled: bool = False
    # Assistant-prefix continuation ("prefill"): the final message must be an
    # assistant turn; generation continues it mid-turn (no end-of-turn marker,
    # no new generation prompt). Requires `messages`.
    continueFinalMessage: bool = False
    injections: list[InjectionDTO] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    output: str
    promptTokens: int
    modelID: str


class ExtractRequest(BaseModel):
    """Engine-level extraction of one concept directory (not the firewall path —
    that goes through a frozen manifest via the experiment job routes)."""

    conceptDirectory: str
    method: str = "meanDifference"
    readingPosition: str = "last token"
    neutralCorpusPath: str | None = None
    outputName: str | None = None


class JobDTO(BaseModel):
    id: str
    kind: str
    # pending | running | succeeded | failed | cancelled | cancelledResumable
    # | checkpointed | merging | prepared | parked. A plain `str` on purpose:
    # a client must tolerate a status it has never heard of rather than fail
    # to decode the record (see `RemoteJobStatusClass` on the Swift side).
    status: str
    createdAt: float
    startedAt: float | None = None
    finishedAt: float | None = None
    result: dict | None = None
    error: str | None = None
    logTail: list[str] = Field(default_factory=list)
    requestedResources: dict = Field(default_factory=dict)
    outputArtifacts: list[dict] = Field(default_factory=list)
    executor: str = "local"
    executorJobID: str | None = None
    cancellationRequested: bool = False
    capabilitySnapshot: dict = Field(default_factory=dict)


class StateDTO(BaseModel):
    """Trimmed state snapshot (subset of the Swift ``StateDTO``)."""

    models: list[str]
    loadedModel: str | None
    loadedRevision: str | None
    device: str | None
    # Actual parameter dtype of the active model (e.g. "bfloat16") — the
    # user-facing answer to "why is this model slow?" (a float32 fallback).
    dtype: str | None = None
    # Effective attention kernel ("eager" on MPS, "sdpa" on CUDA) — provenance
    # for the per-device choice in model_loader.attention_implementation.
    attnImplementation: str | None = None
    numLayers: int | None
    hiddenSize: int | None
    contextWindow: int | None
    isBusy: bool
    loadedModels: list[dict] = Field(default_factory=list)
    # Snapshot bytes per cached model (weights dominate = the device-memory
    # floor at matching dtype): the client grays out models that cannot fit
    # the live session's GPU (2026-07-18).
    modelSizesBytes: dict[str, int] = Field(default_factory=dict)
    jobs: list[JobDTO]
