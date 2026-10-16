"""Robust inverse-depth alignment and edge-aware RGB-D seam treatment."""
import inspect
import numpy as np
from scipy.ndimage import median_filter, distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


def align_inverse_depth(predicted, reference, observed, bounds, preserve_observed=True):
    predicted = np.asarray(predicted, float)
    reference = np.asarray(reference, float)
    finite = np.isfinite(predicted) & (predicted > 0)
    good = observed & finite & np.isfinite(reference) & (reference > 0)
    safe = np.where(finite, predicted, 1.)
    report = {'anchors': int(good.sum()), 'method': 'room_scale_prior'}
    if good.sum() < 32:
        valid_bounds = bounds[np.isfinite(bounds) & (bounds > 0)]
        if not len(valid_bounds) or not finite.any():
            raise ValueError('No finite depth or scale bounds')
        scale = np.quantile(valid_bounds, .65)/max(np.quantile(safe[finite], .85), .01)
        result = safe*scale
    else:
        ids = np.flatnonzero(good)
        # Deterministic spatial holdout: evaluate calibration outside its fitting anchors.
        validation = (ids*13+ids//predicted.shape[1]) % 5 == 0
        train_ids = ids[~validation]
        if len(train_ids) < 24 or validation.sum() < 8:
            train_ids, validation = ids, np.zeros(len(ids), bool)
        p, r = safe.ravel()[train_ids], reference.ravel()[train_ids]
        scale = np.median(r/p)
        result = safe*scale
        report['method'] = 'shared_rgbd_overlap'
        # Relative monocular inverse-depth has scale AND shift ambiguity.
        x, y = 1/p, 1/r
        if np.std(x) > .005*max(np.mean(x), 1e-6):
            X = np.column_stack([x, np.ones_like(x)])
            weight = np.ones(len(x))
            solution = np.array([1/scale, 0.])
            for _ in range(8):
                solution = np.linalg.lstsq(X*weight[:, None]**.5, y*weight**.5, rcond=None)[0]
                residual = X@solution-y
                sigma = max(1.4826*np.median(np.abs(residual-np.median(residual))), .002)
                weight = np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1e-9))
            a, b = solution
            inverse = a/safe+b
            # Extrapolating an unconstrained affine fit must not create infinite/negative depths.
            admissible = a > 0 and (inverse[finite] > .025).mean() > .995
            candidate = 1/np.maximum(inverse, .025)
            check_ids = ids[validation] if validation.any() else train_ids
            ref = reference.ravel()[check_ids]
            old_error = np.median(np.abs(np.log(result.ravel()[check_ids]/ref)))
            new_error = np.median(np.abs(np.log(candidate.ravel()[check_ids]/ref)))
            if admissible and new_error <= old_error:
                result = candidate
                report.update(method='robust_inverse_affine_overlap', inverse_scale=float(a), inverse_shift=float(b))
            report.update(heldout_scale_log_error=float(old_error), heldout_selected_log_error=float(min(new_error, old_error) if admissible else old_error),
                          validation_anchors=int(len(check_ids)))
    upper = np.where(np.isfinite(bounds) & (bounds > .25), bounds, 40.)
    result = np.clip(result, .25, np.maximum(upper, .26))
    if preserve_observed:
        result[good] = reference[good]
    return result, report


def reject_depth_spikes(depth, rgb):
    """Remove isolated depth impulses only on locally colour-consistent surfaces."""
    safe = np.where(np.isfinite(depth) & (depth > 0), depth, 1.)
    med = median_filter(safe, size=3, mode='nearest')
    color = np.asarray(rgb, float)/255
    color_med = median_filter(color, size=(3, 3, 1), mode='nearest')
    isolated = (np.abs(safe-med) > .18*np.maximum(med, .25)) & (np.max(np.abs(color-color_med), axis=-1) < .06)
    cleaned = safe.copy()
    cleaned[isolated] = med[isolated]
    valid = np.isfinite(depth) & (depth > 0)
    return cleaned, valid, int(isolated.sum())


def match_depth_seam(depth, reference, observed, rgb, width=20):
    """Screened edge-aware log-depth correction in a narrow unobserved seam band.

    Observed pixels are Dirichlet anchors. The screened term makes correction decay
    away from the boundary. Large colour/depth edges disconnect neighbouring pixels.
    """
    depth = np.asarray(depth, float).copy()
    anchors = observed & np.isfinite(reference) & (reference > 0)
    if not anchors.any() or anchors.all():
        return depth
    distance = distance_transform_edt(~anchors)
    active = (~anchors) & (distance <= width)
    ids = np.full(depth.shape, -1, int)
    ids[active] = np.arange(active.sum())
    n = int(active.sum())
    if not n:
        return depth
    log = np.log(np.maximum(depth, .25))
    # Before the hard switch, estimate the boundary residual against calibrated prediction.
    residual = np.zeros_like(depth)
    residual[anchors] = np.clip(np.log(reference[anchors])-log[anchors], -.35, .35)
    color = np.asarray(rgb, float)/255
    rows, cols, values = [], [], []
    diagonal = np.full(n, .025)
    rhs = np.zeros(n)
    yy, xx = np.nonzero(active)
    current = ids[yy, xx]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = yy+dy, xx+dx
        inside = (ny >= 0) & (nx >= 0) & (ny < depth.shape[0]) & (nx < depth.shape[1])
        y, x, ny, nx, row = yy[inside], xx[inside], ny[inside], nx[inside], current[inside]
        diff_color = np.linalg.norm(color[y, x]-color[ny, nx], axis=1)
        diff_depth = np.abs(log[y, x]-log[ny, nx])
        weight = np.exp(-(diff_color/.12)**2-(diff_depth/.12)**2)
        diagonal[row] += weight
        neighbour = ids[ny, nx]
        unknown = neighbour >= 0
        rows.extend(row[unknown]); cols.extend(neighbour[unknown]); values.extend(-weight[unknown])
        rhs[row] += weight*residual[ny, nx]
    rows.extend(range(n)); cols.extend(range(n)); values.extend(diagonal)
    matrix = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    tolerance = {'rtol' if 'rtol' in inspect.signature(cg).parameters else 'tol': 1e-5}
    correction, info = cg(matrix, rhs, maxiter=160, **tolerance)
    if info == 0 and np.isfinite(correction).all():
        depth[active] *= np.exp(np.clip(correction, -.35, .35))
    depth[anchors] = reference[anchors]
    return depth


def feather_color_seam(rgb, reference_rgb, observed, depth, reference_depth, width=5):
    """Blend a few compatible pixels; never smear across a colour/depth discontinuity."""
    if not observed.any() or observed.all():
        return np.asarray(rgb).copy()
    distance, indices = distance_transform_edt(~observed, return_indices=True)
    nearest_rgb = reference_rgb[tuple(indices)]
    nearest_depth = reference_depth[tuple(indices)]
    similar = (np.max(np.abs(np.asarray(rgb, float)-nearest_rgb), axis=-1) < 35) & (np.abs(depth-nearest_depth) < .06*np.maximum(depth, .25))
    weight = np.where((~observed) & similar, np.maximum(0, 1-distance/width), 0.)
    result = np.asarray(rgb, float)*(1-weight[..., None])+nearest_rgb*weight[..., None]
    result[observed] = reference_rgb[observed]
    return np.uint8(np.clip(result, 0, 255))


def seam_jump(depth, observed, rgb):
    values = []
    rgb = np.asarray(rgb, float)/255
    for axis in (0, 1):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        compatible = (observed[a] != observed[b]) & (np.max(np.abs(rgb[a]-rgb[b]), axis=-1) < .12)
        jump = np.abs(np.log(np.maximum(depth[a], .25)/np.maximum(depth[b], .25)))
        values.extend(jump[compatible])
    return float(np.median(values)) if values else None


def refine_prediction(predicted, reference, observed, bounds, rgb):
    cleaned, valid, spikes = reject_depth_spikes(predicted, rgb)
    aligned, report = align_inverse_depth(cleaned, reference, observed & valid, bounds, preserve_observed=False)
    switched = aligned.copy()
    switched[observed] = reference[observed]
    result = match_depth_seam(aligned, reference, observed, rgb)
    upper = np.where(np.isfinite(bounds) & (bounds > .25), bounds, 40.)
    result[~observed] = np.minimum(result[~observed], upper[~observed])
    report.update(spikes_replaced=spikes, seam_log_jump_before=seam_jump(switched, observed, rgb),
                  seam_log_jump_after=seam_jump(result, observed, rgb))
    return result, valid, report
