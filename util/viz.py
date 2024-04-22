from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 12})
import ternary

from cortex.testing_utils import has_installed
import cortex

import numpy as np

BRIGHTNESS_MULTIPLIER = 3.5

# make sure fsaverage template is downloaded
# to pycortex data store
fsaverage = "fsaverage"
if not hasattr(cortex.db, fsaverage):
    cortex.utils.download_subject(
        subject_id = fsaverage,
        pycortex_store = cortex.db.filestore
    )
    cortex.db.reload_subjects()  # force filestore reload
    assert hasattr(cortex.db, fsaverage)

def plot_fsaverage(data, mask = None, sulci = True, positive = False,
                        vlim = None, cmap = None, ax = None, colorbar = True):
    '''
    Plots a (327684,) np.array `data` onto fsaverage surface
    using pycortex
    '''

    mask = np.ones_like(data).astype(bool) if mask is None else mask
    vlim = np.abs(data[np.isfinite(data)]).max() if vlim is None else vlim
    if cmap is None:
        cmap = 'plasma' if positive else 'coolwarm'
    color_kwargs = dict(
        vmax = vlim,
        vmin = 0 if positive else -vlim,
        cmap = cmap
    )
    _data = data.copy()
    _data[~mask] = np.nan
    vertex = cortex.Vertex(_data, 'fsaverage', **color_kwargs)
    fig = cortex.quickflat.make_figure(
        vertex,
        colorbar_location = 'center',
        with_rois = True,
        with_sulci = sulci,
        with_curvature = True,
        with_borders = True,
        with_colorbar = colorbar,
        fig = ax
    )
    return fig

def zoom_visual_cortex(ax):
    '''
    zooms into visual cortex for a pycortex plot
    '''
    xlim = ax.get_xlim()
    w = (xlim[1] - xlim[0])/2
    ax.set_xlim(-.2*w, .2*w) # zoom to visual cortex
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0])/2
    ax.set_ylim(-.35*h, .2*h)

def plot_fsaverage_rgb(data, ax = None):
    '''
    data : np.array of shape (3, )
    '''
    rgb = data / data.sum(0)
    rgb *= BRIGHTNESS_MULTIPLIER # increase brightness
    rgb[np.isfinite(rgb) & (rgb > 1)] = 1. # clip back to valid range

    red = cortex.Vertex(255 * rgb[0,:], 'fsaverage', vmin = 0, vmax = 255)
    green = cortex.Vertex(255 * rgb[1,:], 'fsaverage', vmin = 0, vmax = 255)
    blue = cortex.Vertex(255 * rgb[2,:], 'fsaverage', vmin = 0, vmax = 255)
    vertex_data = cortex.VertexRGB(
        red, green, blue,
        'fsaverage',
        vmin = 0, vmax = 255,
        vmin2 = 0, vmax2 = 255,
        vmin3 = 0, vmax3 = 255,
    )
    fig = cortex.quickflat.make_figure(
        vertex_data,
        with_rois = True,
        with_sulci = True,
        with_curvature = True,
        with_borders = True,
        with_colorbar = False,
        fig = ax
    )

def _color_point(x, y, z, scale):
    s = BRIGHTNESS_MULTIPLIER
    r = min(s*y/scale, 1.)
    g = min(s*z/scale, 1.)
    b = min(s*x/scale, 1.)
    return (r, g, b, 1.)

def _generate_heatmap_data(scale = 5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = _color_point(i, j, k, scale)
    return d

def plot_rgb_legend(ax, labels = None, label_offset = .2, scale = 100):
    if labels is None:
        labels = ['pose error', 'velocity', 'joint position']
    tax = ternary.TernaryAxesSubplot(ax = ax, scale = scale)
    data = _generate_heatmap_data(scale)
    tax.heatmap(data, style = "hexagonal", use_rgba = True, colorbar = False)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.boundary()
    tax.right_corner_label(labels[0], offset = -label_offset)
    tax.left_corner_label(labels[1], offset = -label_offset)
    tax.top_corner_label(labels[2], offset = label_offset)
    return tax

def plot_decoding_model(layout, sub, show_sulci = True, model = 'cortex',
                        weight_type = 'filters', ax = None, colorbar = True):
    '''
    plots the vertex-level weights of a decoding model in
    'derivatives/decoding' directory onto fsaverage surface
    '''
    f = layout.get(
        subject = sub,
        scope = 'decoding',
        desc = model,
        suffix = weight_type
        )[0]
    weights = np.load(f)
    weights /= np.nanmax(np.abs(weights)) # scale so all values between -1 and 1
    fig = plot_fsaverage(
        weights, sulci = show_sulci,
        cmap = 'seismic', ax = ax,
        colorbar = colorbar
        )
    return fig

def plot_decoding_roc(layout, sub, ax, legend = True):
    f = layout.get(
        subject = sub,
        scope = 'decoding',
        desc = 'theory',
        suffix = 'roc'
    )[0]
    roc = np.load(f)
    for i in range(roc.shape[0]):
        lab = 'permutations' if i == 0 else None
        fpr = roc[i, 0, :]
        tpr = roc[i, 1, :]
        ax.plot(fpr, tpr, alpha = .5, color = 'grey', label = lab)
    ax.plot((0, 1), (0, 1), color = 'black', linestyle = '--')
    ax.plot(
        roc[0, 0, :], roc[0, 1, :],
        color = 'red',
        label = '(a) theory mask'
        )
    f = layout.get(
        subject = sub,
        scope = 'decoding',
        desc = 'visuomotor',
        suffix = 'roc'
    )[0]
    roc = np.load(f)
    ax.plot(
        roc[0, 0, :], roc[0, 1, :],
        color = 'orange',
        label = '(b) visuomotor mask'
    )
    f = layout.get(
        subject = sub,
        scope = 'decoding',
        desc = 'cortex',
        suffix = 'roc'
    )[0]
    roc = np.load(f)
    ax.plot(
        roc[0, 0, :], roc[0, 1, :],
        color = 'blue',
        label = '(c) whole cortex'
    )
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    if legend:
        ax.legend()
