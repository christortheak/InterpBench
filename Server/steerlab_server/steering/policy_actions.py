"""Version-1 policy action arithmetic, separate from legacy steering owners."""
from dataclasses import dataclass
import torch
from .ablator import orthonormalized


@dataclass(frozen=True)
class Decision:
    """Provider output: a declared action ID and scalar or [tokens] strength."""
    action: str
    strength: object


@dataclass(frozen=True)
class Action:
    specification: dict
    strength: torch.Tensor  # [tokens], already bounded and schedule-masked


def residual(h, actions):
    if not actions: return h
    removals = [a for a in actions if a.specification['kind'] == 'ablate']
    result = h
    if removals:
        strength = removals[0].strength
        if any(not torch.equal(a.strength, strength) for a in removals[1:]):
            raise ValueError('Policies at one site requested different ablation strengths. Use one shared rule for their joint subspace, or different sites.')
        basis = orthonormalized([a.specification['vector'] for a in removals])
        q = torch.tensor(basis, dtype=torch.float32, device=h.device)
        x = h.float()
        result = (x - strength[None, :, None] * ((x @ q.T) @ q)).to(h.dtype)
    for action in actions:
        if action.specification['kind'] != 'add': continue
        v = torch.tensor(action.specification['vector'], dtype=torch.float32, device=h.device)
        result = (result.float() + action.strength[None, :, None] * v).to(h.dtype)
    if not torch.isfinite(result).all(): raise ValueError('Policy actions produced non-finite activations. Reduce their strength bounds.')
    return result


def logits(scores, actions):
    if not actions: return scores
    result = scores.clone()
    allowed = None
    for action in actions:
        spec = action.specification; tokens = spec['tokens']
        if max(tokens) >= scores.shape[-1]: raise ValueError('A policy token ID is outside this model vocabulary.')
        strength = action.strength.item()
        if spec['kind'] == 'logitBias': result[:, tokens] += strength
        elif strength > 0:
            selected = set(tokens)
            allowed = selected if allowed is None else allowed & selected
    if allowed is not None:
        if not allowed: raise ValueError('Policy token constraints conflict: their allowed tokens have no intersection.')
        mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        mask[list(sorted(allowed))] = False
        result[:, mask] = -torch.inf
    if torch.isnan(result).any() or torch.isposinf(result).any() or not torch.isfinite(result).any(dim=-1).all():
        raise ValueError('Policy token constraints leave no finite choice, or a bias overflowed. Adjust the policy; existing generation constraints remain in force.')
    return result
