"""Which layers an agent's ablating injection covers.

An ablating injection may list the layers it covers (``layers``); absent or
empty means every layer, because a removal at one layer is usually rewritten
by the layers above it. A list is a SET of blocks: ``[3, 7, 10]`` ablates
exactly those three and never the blocks between them, and neither its order
nor a repeat changes what is removed.

A listed layer the vector does not have is refused, never skipped. Skipping
runs a different study from the one declared and preregistered, and a list
naming only such layers used to ablate every layer on the Mac engine and
none here. Swift twin: ``ModelVariantArtifact.InjectionRef``
(``declaredAblationLayers``, ``ablationLayerProblem``), which writes the same
sentence.
"""

from __future__ import annotations


def covered(variant: str, concept, declared, layer_count: int) -> list[int]:
    """The layers, ascending and once each, that an ablating injection of
    ``variant`` covers on a vector of ``layer_count`` layers. Raises
    ``ValueError`` (which the routes map to a 400) when the list names a layer
    the vector does not have, or an entry that is not a layer index."""
    if not declared:
        return list(range(layer_count))
    if not isinstance(declared, list) or any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in declared):
        raise ValueError(
            f"variant '{variant}': the ablation of '{concept}' lists layers "
            f"{declared!r}, which are not all whole layer indices; list block "
            f"numbers such as [3, 7, 10], or delete the list to ablate every "
            f"layer; in a frozen study, save the corrected agent as a new agent, "
            f"duplicate the study, attach the new agent to the copy, then "
            f"validate and freeze it")
    layers = sorted(set(declared))
    outside = [layer for layer in layers if not 0 <= layer < layer_count]
    if outside:
        one = len(outside) == 1
        raise ValueError(
            f"variant '{variant}': the ablation of '{concept}' lists "
            f"{'layer' if one else 'layers'} "
            f"{', '.join(str(layer) for layer in outside)}, which its vector "
            f"does not have (its layers are 0 to {layer_count - 1}); remove "
            f"{'it' if one else 'them'} from the injection's layers, or delete "
            f"the list to ablate every layer; in a frozen study, save the "
            f"corrected agent as a new agent, duplicate the study, attach the new "
            f"agent to the copy, then validate and freeze it")
    return layers
