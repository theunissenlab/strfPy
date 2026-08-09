# Analysis helpers for the seg+ID (segmentation + identification) models
# fitted by calcSegmentedModel.fit_seg.

import numpy as np


def reconstruct_dogs_rf(model, pca, feat_shape, nDOGS=5):
    """
    Reconstruct onset/offset spectro-temporal receptive fields from a fitted
    seg+ID DOGS model.

    fit_seg(..., kernel='DG', basis_args=dogs_args, nD=nDOGS) ridge-fits a
    weight for every (DOGS temporal order, event type, PCA component)
    combination into a single `weights` vector, ordered [DOGS order][event
    type (0=onset, 1=offset)][PCA component] (see
    calcSegmentedModel.generate_dogs_features). Summing those weights over
    DOGS temporal order for a given event type and projecting back through
    the PCA basis gives a single receptive-field image per event type, in
    the same axes as the raw event-triggered feature windows (e.g.
    spect_windows: frequency x time-relative-to-event, where sample index 0
    is `segmentBuffer` ms before the event and the window continues to
    `respChunkLen - segmentBuffer` ms after it -- see the `spect_inds`
    construction in calcSegmentedModel.preprocess_srData. The window is
    mostly post-event, not pre-event).

    Parameters
    ----------
    model : dict
        A fitted segmented model as returned by
        fit_seg(..., kernel='DG', basis_args=dogs_args, nD=nDOGS) -- e.g.
        segIDModelDG or segIDModelDGMPS from process_unit. Must contain a
        'weights' entry.
    pca : sklearn.decomposition.PCA
        The fitted PCA basis used to generate the feature `model` was fit
        on -- e.g. pca_spect or pca_mps from
        calcSegmentedModel.generate_event_pca_feature.
    feat_shape : tuple of int
        The (dim0, dim1) shape of one event's raw feature window before it
        was flattened for PCA -- e.g. (nFreq, respChunkLen) for
        spect_windows. dim0*dim1 must equal pca.components_.shape[1].
    nDOGS : int
        Number of DOGS temporal-basis orders `model` was fit with (must
        match the nD passed to fit_seg for this model). Default 5.

    Returns
    -------
    dict
        {'onset': ndarray of shape feat_shape,
         'offset': ndarray of shape feat_shape}
    """
    npcs = pca.n_components_
    nfeats = feat_shape[0] * feat_shape[1]
    if pca.components_.shape[1] != nfeats:
        raise ValueError(
            "feat_shape %s (%d features) does not match pca.components_'s "
            "%d features" % (feat_shape, nfeats, pca.components_.shape[1])
        )

    weights = np.asarray(model['weights'])
    nEventTypes = 2
    expected_size = nDOGS * nEventTypes * npcs
    if weights.size != expected_size:
        raise ValueError(
            "model['weights'] has %d entries, expected nDOGS*2*npcs = %d "
            "(nDOGS=%d, npcs=%d) -- check nDOGS matches the nD used to fit "
            "this model." % (weights.size, expected_size, nDOGS, npcs)
        )
    w = weights.reshape(nDOGS, nEventTypes, npcs)

    RFs = {}
    for iEventType, name in enumerate(["onset", "offset"]):
        comp_weight = w[:, iEventType, :].sum(axis=0)
        RFs[name] = (comp_weight @ pca.components_).reshape(feat_shape)
    return RFs
