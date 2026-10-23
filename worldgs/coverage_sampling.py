"""Preserve spatial coverage when initializing or imposing a Gaussian budget."""
import numpy as np


def component_stratified_seed_ids(xyz,components,limit,seed=0):
    """Reserve per-object detail, then resample the remaining spatial budget."""
    rng=np.random.default_rng(seed)
    groups=np.unique(components)
    reserved=[]
    for group in groups:
        candidates=np.flatnonzero(components==group)
        reserved.extend(rng.choice(candidates,min(len(candidates),limit//(2*len(groups))),replace=False))
    reserved=np.asarray(reserved,dtype=int)
    available=np.flatnonzero(~np.isin(np.arange(len(xyz)),reserved))
    remainder=available[stratified_seed_ids(xyz[available],limit-len(reserved),seed)]
    return np.r_[reserved,remainder]


def stratified_seed_ids(xyz, limit, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xyz))
    if len(xyz) <= limit:
        return order
    origin, cell = xyz.min(0), .025
    for _ in range(40):
        _, first = np.unique(np.floor((xyz[order]-origin)/cell), axis=0, return_index=True)
        representatives = order[first]
        if len(representatives) <= limit:
            keep = np.zeros(len(xyz), bool)
            keep[representatives] = True
            extra = order[~keep[order]][:limit-len(representatives)]
            return np.concatenate([representatives, extra])
        cell *= 1.35
    return order[:limit]


def coverage_drop_mask(means, scores, limit):
    import torch
    if len(means) <= limit:
        return torch.zeros(len(means), device=means.device, dtype=torch.bool)
    origin, cell = means.amin(0), .04
    for _ in range(40):
        _, inverse = torch.unique(torch.floor((means-origin)/cell).to(torch.int64), dim=0, return_inverse=True)
        count = int(inverse.max())+1
        if count <= limit:
            best = torch.full((count,), -torch.inf, device=scores.device, dtype=scores.dtype)
            best.scatter_reduce_(0, inverse, scores, reduce='amax', include_self=True)
            ids = torch.arange(len(scores), device=scores.device)
            winners = torch.full((count,), len(scores), device=scores.device, dtype=torch.int64)
            winners.scatter_reduce_(0, inverse, torch.where(scores == best[inverse], ids, len(scores)), reduce='amin', include_self=True)
            keep = torch.zeros(len(scores), device=scores.device, dtype=torch.bool)
            keep[winners] = True
            remaining = limit-count
            if remaining:
                keep[scores.masked_fill(keep, -torch.inf).topk(remaining).indices] = True
            return ~keep
        cell *= 1.4
    keep = scores.topk(limit).indices
    drop = torch.ones(len(scores), device=scores.device, dtype=torch.bool)
    drop[keep] = False
    return drop
