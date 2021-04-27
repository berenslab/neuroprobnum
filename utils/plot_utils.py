import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

from iteration_utils import auto_zip
import math_utils
import data_utils
import ode_solver


### Color palette. ###
DEFAULT_PALETTE = 'tab10'
TEXT_WIDTH = 5.2 # inch
FULLPAGE_WIDTH = 7.5 # inch
FIGURE_HEIGHT = 1.2 # inch
FIGURE_MAX_HEIGHT = 8.75 # inch

### Methods and method mapping. ###
method2colidx = {
    'FE': 1, 'HN': 2, 'RKBS': 9, 'RKCK': 2,
    'RKDP': 4, 'RK10': 6, 'EE': 5, 'EEMP': 3, 'IE': 7
}
method2marker_dict = {
    'FE': 'v', 'RKBS': '^', 'RKDP': 'P', 'RKCK': 'X',
    'RK10': 'P', 'EE': 'o', 'EEMP': '<', 'IE': 'd', 'HN': '>'
}
method2pos_dict = {
    'IE': 0, 'FE': 1, 'EE': 2, 'EEMP': 3, 'EEMPhs': 3.1, 'HN': 3.5,
    'RKBS': 4, 'RKCK': 6, 'RKDP': 7, 'RK10': 8
}

### Line parameters. ###
mean_kw_default = {"alpha": 0.8, "ms": 4., "markeredgewidth": 0.5, "zorder": 0, "zorder": 0}
line_kw_default = {"lw": 1.0, "ls": '-', "alpha": 1.0, "zorder": 1, "capsize": 0}
outl_kw_default = {"marker": '*', "lw": 0.5, "alpha": 1.0, "s": 4}
nans_kw_default = {"ls": ':', "zorder": -10, "alpha": 0.5}    

### Trace parameters. ###
tr_mean_kw_default = {"label": 'mean', "c": 'C0', "zorder": 0, "lw": 0.8, "alpha": 0.8}
tr_bnds_kw_default = {"color": 'red', "alpha": 0.5, "lw":0, "zorder": -20}
tr_smpl_kw_default = {"ls": (0, (5, 1)), "c":'k', "alpha": 0.8, "label": 'sample', "zorder": -10, "lw": 0.8}

### Labels for pertubation methods ###
pert_method2symbol  = {'conrad': 'x', 'abdulle': 't', 'abdulle_ln': 't'}
pert_param_symbol = "\sigma"

############################################################################
def set_rcParams():
    sns.set_context('paper')
    sns.set_style('ticks')
    plt.rcParams['axes.linewidth'] = .7
    plt.rcParams['xtick.major.width'] = .5
    plt.rcParams['ytick.major.width'] = .5
    plt.rcParams['xtick.minor.width'] = .5
    plt.rcParams['ytick.minor.width'] = .5
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['xtick.minor.size'] = 1
    plt.rcParams['ytick.minor.size'] = 1
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.major.pad']= 3.5
    plt.rcParams['ytick.major.pad']= 3.5
    plt.rcParams['figure.figsize'] = (FULLPAGE_WIDTH, FULLPAGE_WIDTH/3)
    plt.rcParams['savefig.facecolor']=(1,1,1,1)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.serif'] = 'Times New Roman'
    
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.default'] = 'cal'
    plt.rcParams['mathtext.cal'] = 'DejaVu Sans'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.sf'] = 'Arial'
    plt.rcParams['mathtext.tt'] = 'Arial'
    
    #plt.rcParams['axes.unicode_minus']=True
    plt.rcParams['figure.dpi'] = 120 # only affects the notebook
    plt.rcParams['figure.facecolor'] = (0.8,0.8,0.8,1) # only affects the notebook

    plt.rcParams['legend.borderaxespad'] = 0.2
    plt.rcParams['legend.borderpad'] = 0.0
    plt.rcParams['legend.columnspacing'] = 0.2
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.handlelength'] = 1.3
    plt.rcParams['legend.handletextpad'] = 0.3
    plt.rcParams['legend.labelspacing'] = 0.1
    plt.rcParams['legend.loc'] = 'upper right'

set_rcParams()

############################################################################
def savefig(filename, fig=None):
    """Save figure to file"""
    assert "figures" not in filename
    assert ".pdf" not in filename
    assert ".png" not in filename
    assert ".eps" not in filename
    
    data_utils.make_dir("_figures")
    data_utils.make_dir("../../_figures")
    
    if fig is None: fig = plt.gcf()
    
    fig.savefig(f"_figures/{filename}.pdf") # save in folder
    fig.savefig(f"../../_figures/{filename}.pdf") # save in general folder
    #fig.savefig(f"../../_figures/{filename}.tif") # save in general folder
    #fig.set_rasterized(True)
    #for ax in iterate_axes(fig.axes): ax.set_rasterized(True)
    #fig.savefig(f"../../_figures/{filename}.eps", rasterized=True) # save in general folder

############################################################################
def show_saved_figure(fig):
    fig.savefig('.temp.jpg', dpi=600)
    plt.figure(figsize=(10, 10), facecolor=(0.5,0.5,0.5,0.5))
    
    im = plt.imread('.temp.jpg')
    
    if (np.any(im[0,:]  < 255) or\
        np.any(im[-1,:] < 255) or\
        np.any(im[:,0]  < 255) or\
        np.any(im[:,-1] < 255)):
        print('Warning: Figure is probably clipped!')
    
    plt.imshow(im, aspect='equal')
    plt.axis('off')
    plt.show()
    
    from os import remove as removefile
    removefile('.temp.jpg')

def tight_layout(h_pad=1, w_pad=1, rect=(0,0,1,1), pad=None):
    """tigh layout with different default"""
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad, pad=pad or 2./plt.rcParams['font.size'], rect=rect)
    
############################################################################
def subplots(nx_sb=4, ny_sb=1, xsize='text', ysizerow='auto', yoffsize=0.0, **kwargs):
    """Like plt.subplots, but with auto size."""
    
    # Get auto fig size.
    if xsize=='text':
        xsize = TEXT_WIDTH
    elif xsize == 'fullwidth':
        xsize = FULLPAGE_WIDTH
    else:
        assert isinstance(xsize, (float, int)), "x not in {'text', 'fullwidth', float}"

    if ysizerow=='auto':
        ysizerow = FIGURE_HEIGHT
    
    ysize = ysizerow*ny_sb+yoffsize
    
    if ysize > FIGURE_MAX_HEIGHT:
        print(f"ysize: {ysize} > {FIGURE_MAX_HEIGHT}")
    
    fig, axs = plt.subplots(ny_sb, nx_sb, figsize=(xsize, ysize), **kwargs)
    return fig, axs
    
############################################################################
def auto_subplots(n_plots, max_nx_sb=4, max_ny_sb=20, xsize='text', ysizerow='auto', yoffsize=0.0,
                  allow_fewer=True, rm_unused=True, **kwargs):
    """Create subplots with auto infer n_rows and cols. Otherwise similar to subplots"""
    
    nx_sb = np.min([max_nx_sb, n_plots])
    ny_sb = np.min([max_ny_sb, int(np.ceil([n_plots/nx_sb]))])

    if not allow_fewer: assert nx_sb*ny_sb >= n_plots

    fig, axs = subplots(nx_sb=nx_sb, ny_sb=ny_sb, xsize=xsize, ysizerow=ysizerow, yoffsize=yoffsize, **kwargs)
    
    if n_plots > 1:
        if rm_unused and axs.size > n_plots:
            for i in np.arange(n_plots, axs.size):
                axs.flat[i].axis('off')
    
    return fig, axs

############################################################################
def iterate_axes(axs):
    """Make axes iterable, independent of type.
    axs (list of matplotlib axes or matplotlib axis) : Axes to apply function to.    
    """

    if isinstance(axs, list):
        return axs
    elif isinstance(axs, np.ndarray):
        return axs.flatten()
    else:
        return [axs]

def int_format_ticks(axs, which='both'):
    """Use integer ticks for integers."""
    from matplotlib.ticker import FuncFormatter
    def int_formatter(x, pos):
        if x.is_integer():
            return str(int(x))
        else:
            return f"{x:g}"
    formatter = FuncFormatter(int_formatter)
    if which in ['x', 'both']:
        for ax in iterate_axes(axs): ax.xaxis.set_major_formatter(formatter)
    if which in ['y', 'both']:
        for ax in iterate_axes(axs): ax.yaxis.set_major_formatter(formatter)
    
def scale_ticks(axs, scale, x=True, y=False):
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
    for ax in iterate_axes(axs):
        if x: ax.xaxis.set_major_formatter(ticks)
        if y: ax.yaxis.set_major_formatter(ticks)
    
############################################################################
def get_method_order(method):
    """Return order of method"""
    return ode_solver.get_solver(
        method=method, t0=0, y0=np.zeros(1), h0=1,
        odefun_yinf_and_yf=lambda t, y: (np.zeros(1), np.zeros(1)),
        odefun_ydot=lambda t, y: np.zeros(1)).order_B
    
############################################################################
def move_xaxis_outward(axs, scale=5):
    """Move xaxis outward.
    axs (array or list of matplotlib axes) : Axes to apply function to.
    scale (float) : How far xaxis will be moved.
    """
    for ax in iterate_axes(axs): ax.spines['bottom'].set_position(('outward', scale))
        
def move_yaxis_outward(axs, scale=5):
    """Move xaxis outward.
    axs (array or list of matplotlib axes) : Axes to apply function to.
    scale (float) : How far xaxis will be moved.
    """
    for ax in iterate_axes(axs): ax.spines['left'].set_position(('outward', scale))

############################################################################
def adjust_log_tick_padding(axs, pad=2.1):
    """ Change tick padding for all log scaled axes.    
    Parameters:
    axs (array or list of matplotlib axes) : Axes to apply function to.
    pad (float) : Size of padding.
    """

    for ax in iterate_axes(axs):
        if ax.xaxis.get_scale() == 'log':
            ax.tick_params(axis='x', which='major', pad=pad)
            ax.tick_params(axis='x', which='minor', pad=pad)
                
        if ax.yaxis.get_scale() == 'log':
            ax.tick_params(axis='y', which='major', pad=pad)
            ax.tick_params(axis='y', which='minor', pad=pad)
            
############################################################################
def set_labs(axs, xlabs=None, ylabs=None, titles=None, panel_nums=None, panel_num_space=0, panel_num_va='bottom'):
    """Set labels and titles for all given axes.
    Parameters:
    
    axs : array or list of matplotlib axes.
        Axes to apply function to.
        
    xlabs, ylabs, titles : str, list of str, or None
        Labels/Titles.
        If single str, will be same for all axes.
        Otherwise should have same length as axes.

    """
            
    for i, ax in enumerate(iterate_axes(axs)):
        if xlabs is not None:
            if isinstance(xlabs, str): xlab = xlabs
            else: xlab = xlabs[i]
            ax.set_xlabel(xlab)
            
        if ylabs is not None:
            if isinstance(ylabs, str): ylab = ylabs
            else: ylab = ylabs[i]
            ax.set_ylabel(ylab)
                        
        if titles is not None:
            if isinstance(titles, str): title = titles
            else: title = titles[i]
            ax.set_title(title)
            
        if panel_nums is not None:
            if panel_nums == 'auto': panel_num = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
            elif isinstance(panel_nums, str): panel_num = panel_nums
            else: panel_num = panel_nums[i]
            ax.set_title(panel_num+panel_num_space*' ', loc='left', fontweight='bold', ha='right', va=panel_num_va)
            
            
############################################################################
def left2right_ax(ax):
    """Create a twin axis, but remove all duplicate spines.
    Parameters:
    ax (Matplotlib axis) : Original axis to create twin from.
    Returns:
    ax (Matplotlib axis) : Twin axis with no duplicate spines.
    """
    
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax = ax.twinx()
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    return ax
            
    
def move_box(axs, dx=0, dy=0):
    """Change offset of box"""
    for ax in iterate_axes(axs):
        box = np.array(ax.get_position().bounds)
        if dx != 0: box[0] += dx
        if dy != 0: box[1] += dy
        ax.set_position(box)
        
def change_box(axs, dx=0, dy=0):
    """Change offset of box"""
    for ax in iterate_axes(axs):
        box = np.array(ax.get_position().bounds)
        if dx != 0: box[2] += dx
        if dy != 0: box[3] += dy
        ax.set_position(box)
    
############################################################################
def method2color(method, colorpalette='tab10', isscatter=False):
    method_idx = method2colidx.get(method, None)
    return idx2color(method_idx, colorpalette, isscatter)
    
############################################################################
def neuron2color(neuron_idx, colorpalette='Set1', isscatter=False):
     return idx2color(neuron_idx, colorpalette, isscatter)
    
############################################################################
def stim2color(stim_idx, colorpalette='Set2', isscatter=False):
    return idx2color(stim_idx, colorpalette, isscatter)
    
############################################################################
def idx2color(idx, colorpalette, isscatter):
    if idx is None:
        c = (0,0,0)
    else:
        c = ColorConverter.to_rgb(sns.color_palette(colorpalette).as_hex()[idx])
    if isscatter: c = np.atleast_2d(np.asarray(c))
    return c
    
############################################################################
def method2marker(method, default_marker='.'):
    """Return marker for method."""
    return method2marker_dict.get(method, default_marker)
    
############################################################################
def mode2ls(mode):
    """Return linestyle for mode. """
    if (mode == 'adaptive') or (mode == 1): return (1,(1,1))
    elif (mode == 2): return 'dashdot'
    else: return '-'
    
############################################################################
def mode2mfc(mode):
    """Return markerfacecolor for mode. """
    if (mode == 'adaptive') or (mode == 1): return 'w'
    elif (mode == 2): return 'k'
    else: return None

############################################################################
def sort_methods(methods):
    sortidxs = np.argsort([method2pos_dict.get(method.replace('$', '').split('_')[0], 100) for method in methods])
    return [methods[idx] for idx in sortidxs]

############################################################################
def text2mathtext(txt):
    txt = txt.replace('^', '}^\mathrm{')
    txt = txt.replace('_', '}_\mathrm{')
    txt = txt.replace(' ', '} \mathrm{')
    txt = txt.replace('-', '\mathrm{-}')
    return r"$\mathrm{" + txt + "}$"

def method2label(method, adaptive=None, step_param=None, pert_method=None, pert_param=None, time_unit='ms'):
    """Generate plot label for method."""
    label = method.replace('_', '')
    if adaptive is not None:
        if adaptive == 1:
            label += '_{a}'
        elif adaptive == 2:
            label += '_{pf}'
        else:
            label += '_{f}'
    if pert_method is not None:
        label += '^{' + pert_method2symbol[pert_method]
        if (pert_param is not None) and (pert_param != 'auto'): label += f"={pert_param!r}"
        label += '}'
    if step_param is not None:
        if adaptive == 1:
            label += "(κ=" + f"{step_param:.1e}".replace("e-0", "e-").replace(".0", "") + ")"
        else:
            label += "(Δt="
            if time_unit == 's':
                label += f"{step_param * 1e3!r}" + " ms)"
            elif time_unit == 'ms':
                label += f"{step_param!r}" + " ms)"
            else:
                label += f"{step_param!r} {time_unit}"  + " )"
    
    return text2mathtext(label)

def step_param2label(step_param, adaptive, time_unit='ms'):
    if adaptive == 1:
        label = f"κ={step_param:0.1e}".replace("e-0", "e-").replace(".0e", "e")
    else:
        dt = "Δt" if adaptive == 0 else "Δt^* "
        if time_unit == 's':
            label = f"{dt}={step_param*1e3!r} ms"
        elif time_unit == 'ms':
            label = f"{dt}={step_param!r} ms"
        else:
            label = f"{dt}={step_param!r} {time_unit}"
        
    return text2mathtext(label)

def step_param2tick(step_param, adaptive):
    if adaptive == 1:
        label = f"{step_param:0.1e}".replace("e-0", "e-").replace(".0e", "e")
    else:
        label = f"{step_param!r}"
    return label

        
def pert_param2label(pert_param, pert_method):
    label = f'{pert_method2symbol[pert_method]}={pert_param:.2g}'
    return text2mathtext(label)

def mode2label(mode):
    if (mode == 'adaptive') or (mode == 1): return 'Adaptive'
    elif (mode == 2): return 'Pseudo-fixed'
    else: return 'Fixed'
    
def mode2xlabel(mode, time_unit):
    if (mode == 'adaptive') or (mode == 1): return 'Tolerance'
    else: return f'Step-size ({time_unit})'

def stim2label(stim_name):
    if stim_name == 'Step':
        return 'Step'
    elif 'Noisy' in stim_name:
        return 'Noisy'
    else:
        return stim_name

############################################################################
def _get_method_legend_handles(methods, mean_kw=dict()):
    plot_mean_kw = mean_kw_default.copy()
    plot_mean_kw.update(mean_kw) 
    
    legend_handles = []
    for method in methods:
    
        method = method.replace('$', '').split('_')[0]
    
        plot_mean_kw_i = plot_mean_kw.copy()
    
        if 'color' not in plot_mean_kw_i: plot_mean_kw_i['color'] = method2color(method)
        if 'marker' not in plot_mean_kw_i: plot_mean_kw_i['marker'] = method2marker(method)
        
        mh = Line2D([0], [0], lw=0, **plot_mean_kw_i)
        
        legend_handles.append(mh)
    return legend_handles


def make_method_legend(ax, methods, mean_kw=dict(), legend_kw=dict(), pert_method=None):
    """Make method legend on given axis.
    methods (list of str) : Methods to add to legend.
    mean_kw (dict) : Parameters for mean
    kwargs : Parameters for plt.legend
    """
            
    legend_handles = _get_method_legend_handles(methods, mean_kw=mean_kw)
        
    if pert_method is not None:
        methods = [text2mathtext(f"{method}^{pert_method2symbol[pert_method]}") for method in methods]
            
    ax.legend(legend_handles, methods, **legend_kw)
    
    
def _get_mode_legend_handles(method, mean_kw=dict(), modes=[0,1]):
    """Get handles for mode legend.
    """
    plot_mean_kw = mean_kw_default.copy()   
    plot_mean_kw.update(mean_kw) 
    
    legend_handles = []
    for mode in modes:
    
        plot_mean_kw_i = plot_mean_kw.copy()
    
        if 'color' not in plot_mean_kw_i: plot_mean_kw_i['color'] = method2color(method)
        if 'marker' not in plot_mean_kw_i: plot_mean_kw_i['marker'] = method2marker(method)
        if 'mfc' not in plot_mean_kw_i: plot_mean_kw_i['mfc'] = mode2mfc(mode)
        if 'ls' not in plot_mean_kw_i: plot_mean_kw_i['ls'] = mode2ls(mode)
                
        mh = Line2D([0], [0], **plot_mean_kw_i)
       
        legend_handles.append(mh)
            
    return legend_handles
    
def make_mode_legend(ax, method, mean_kw=dict(), legend_kw=dict(handlelength=2.5), modes=[0,1]):
    """Make method legend on given axis."""
    legend_handles = _get_mode_legend_handles(method, mean_kw=mean_kw, modes=modes)
    
    modes_labels = []
    if 0 in modes: modes_labels.append('fixed')
    if 1 in modes: modes_labels.append('adaptive')
    if 2 in modes: modes_labels.append('pseudo-fixed')
    
    ax.legend(legend_handles, modes_labels, **legend_kw)

    
def grid(ax, axis='both', major=True, minor=False, **kwargs):
    """make grid on axis"""
    for axi in iterate_axes(ax):
        if major:
            axi.grid(True, axis=axis, which='major', alpha=.3, c='k',
                     lw=plt.rcParams['ytick.major.width'], zorder=-10000, **kwargs)
        if minor:
            axi.grid(True, axis=axis, which='minor', alpha=.3, c='gray',
                     lw=plt.rcParams['ytick.minor.width'], zorder=-20000, **kwargs)
    
    
def make_method_and_mode_legend(
        ax, methods, example_method=None, pert_method=None,
        mean_kw=dict(), mean_kw_example=dict(color='gray', marker='o'),
        mode_lines=True, mode_handlelength=2.5,
        legend1_kw=dict(loc='lower left'), legend2_kw=dict(loc='lower right'),
        modes=[0,1]
    ):
    """Make method legend on given axis."""
    
    legend_handles1 = _get_method_legend_handles(methods, mean_kw=mean_kw)
        
    if pert_method is not None:
        methods = [text2mathtext(f"{method}^{pert_method2symbol[pert_method]}") for method in methods]
    
    example_method = example_method or methods[-1]
    if not mode_lines: mean_kw_example['ls'] = 'None' 
    legend_handles2 = _get_mode_legend_handles(method=example_method, mean_kw=mean_kw_example, modes=modes)
    
    modes_labels = []
    if 0 in modes: modes_labels.append('fixed')
    if 1 in modes: modes_labels.append('adaptive')
    if 2 in modes: modes_labels.append('pseudo-fixed')
    
    l1 = ax.legend(legend_handles1, methods, **legend1_kw)    
    l2 = ax.legend(legend_handles2, modes_labels, **legend2_kw,
                   handlelength=mode_handlelength if mode_lines else plt.rcParams['legend.handlelength'])
    ax.add_artist(l1) # add l1 as a separate artist to the axes


def get_legend_handle(marker, color, ls, altcolor=None, **kwargs):
    """Make method legend on given axis."""
    return Line2D([0], [0], marker=marker, ls=ls, color=color, markerfacecoloralt=altcolor,
                  fillstyle='left' if altcolor is not None else 'full', **kwargs)
    
def get_legend_handles(markers=[], colors=[], lss=[], **kwargs):
    """Get handle for list of markers, colors and linestyles"""
    legend_handles = []
    for marker, color, ls in zip(markers, colors, lss):
        legend_handles.append(get_legend_handle(marker, color, ls, **kwargs))
    return legend_handles
    
    
def row_title(ax, title, pad=70, size='large', ha='left', va='center', **kwargs):
    """Create axis row title using annotation"""
    ax.annotate(title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=size, ha=ha, va=va, **kwargs)

############################################################################
def get_x_positions(n_positions, idx, n_idxs, offset=0.18):
    """Get positions on x-axis to plot for given parameter setting."""
    return np.arange(n_positions) + offset*(idx-((n_idxs-1)/2))

############################################################################
def make_share_xlims(axs, symmetric=False, xlim=None):
    """Use xlim lower and upper bounds for all axes."""
    if xlim is None:
        xlb = np.min([ax.get_xlim()[0] for ax in iterate_axes(axs)])
        xub = np.max([ax.get_xlim()[1] for ax in iterate_axes(axs)])

        if not symmetric:
            xlim = (xlb, xub)
        else:
            xlim = (-np.max(np.abs([xlb, xub])), np.max(np.abs([xlb, xub])))
    for ax in iterate_axes(axs): ax.set_xlim(xlim)

############################################################################
def make_share_ylims(axs, symmetric=False, ylim=None):
    """Use ylim lower and upper bounds for all axes."""
    if ylim is None:
        ylb = np.min([ax.get_ylim()[0] for ax in iterate_axes(axs)])
        yub = np.max([ax.get_ylim()[1] for ax in iterate_axes(axs)])

        if not symmetric:
            ylim = (ylb, yub)
        else:
            ylim = (-np.max(np.abs([ylb, yub])), np.max(np.abs([ylb, yub])))
    for ax in iterate_axes(axs): ax.set_ylim(ylim)

############################################################################
def get_bounds_and_fliers(arr, qs, n_nans_allowed):
    """Get mean, bounds, fliers and n_nans for array."""
    
    arr = np.asarray(arr)    
    nan_idx = np.isnan(arr)
    n_nans = np.sum(nan_idx)
    
    # Ensure not too many nans or infs.
    if (n_nans <= n_nans_allowed) and (np.sum(~nan_idx) > 0) and (np.sum(arr>1e200) < np.ceil(arr.size*(100-qs[1])/100.)):
        
        md = np.nanmedian(arr)
        lb = np.nanpercentile(arr, q=qs[0])
        ub = np.nanpercentile(arr, q=qs[1])
        fliers = arr[((arr < lb) | (arr > ub)) & ~np.isclose(arr, md)]            
        
    else:
        md = np.nan
        lb = np.nan
        ub = np.nan
        fliers = np.empty(0)
        
    return md, lb, ub, fliers, n_nans

def get_tlim_idxs(ts, tlim):
    """Get idx of ts array within given tlim"""
    idx0 = int(np.argwhere(ts >= tlim[0])[0])
    idx1 = ts.size-1 if tlim[1] > ts[-1] else int(np.argwhere(ts >= tlim[1])[0])
    return idx0, idx1


def interpolate_data(ts, ys, dt, t0=None, tmax=None):
    """Interpolate data to be able to plot means etc."""
    
    intpol_vs_list= []
    for ts_i, vs_i in auto_zip(ts, ys):
        
        if t0 is None: t0 = ts_i[0]
        if tmax is None: tmax = ts_i[-1]
        ts_intpol = np.arange(0, tmax, dt)
        
        intpol_vs_list.append(math_utils.intpol_ax0(ts_i, vs_i, ts_intpol=ts_intpol, kind='linear'))

    return ts_intpol, np.asarray(intpol_vs_list)


############################################################################    
def plot_mean_and_uncertainty(
        ax, ts, ys, qs=(10,90), intpol_dt=None, xlim=None, smpidxs=[],
        tr_mean_kw=dict(), tr_bnds_kw=dict(), tr_smp_kw=dict()
    ):
    # Plot parameters.
    plot_tr_mean_kw = tr_mean_kw_default.copy()
    plot_tr_bnds_kw = tr_bnds_kw_default.copy()
    plot_tr_smp_kw = dict(c='gray', alpha=0.8, lw=0.6, ls='-', clip_on=True)

    plot_tr_mean_kw.update(tr_mean_kw)
    plot_tr_bnds_kw.update(tr_bnds_kw)
    plot_tr_smp_kw.update(tr_smp_kw)
    
    if "label" not in plot_tr_bnds_kw:
        plot_tr_bnds_kw["label"] = f'q{qs[0]}-{qs[1]}'
        
    if intpol_dt is not None:
        ts, ys = interpolate_data(ts, ys, dt=intpol_dt)     
    
    assert isinstance(ys, np.ndarray), 'To plot list data, please set intpol_dt for interpolation'
                              
    # Compute data to plot.
    y_mu  = np.mean(ys, axis=0)
    y_lb = np.percentile(ys, q=qs[0], axis=0)
    y_ub = np.percentile(ys, q=qs[1], axis=0)
    
    if xlim is not None:
        idxs = (ts>=xlim[0]) & (ts<=xlim[1])
    else:
        idxs = np.ones_like(ts).astype(bool)
    
    # Plot data.
    for i, smpidx in enumerate(smpidxs):
        plot_sample_trace(ax=ax, ts=ts, ys=ys[smpidx], tr_kw=plot_tr_smp_kw, xlim=xlim, label='sample' if i == 0 else '_')
    
    ax.plot(ts[idxs], y_mu[idxs], **plot_tr_mean_kw)
    ax.fill_between(ts[idxs], y_lb[idxs], y_ub[idxs], **plot_tr_bnds_kw)
    
############################################################################
def plot_sample_trace(ax, ts, ys, tr_kw=dict(), label="", intpol_dt=None, xlim=None):
    """Plot a trace with smpl parameters"""
    
    ts = ts.copy()
    ys = ys.copy()
    
    if intpol_dt is not None:
        ts_intpol = np.arange(ts[0], ts[-1], intpol_dt)
        ys = math_utils.intpol_ax0(ts, ys, ts_intpol=ts_intpol, kind='linear')
        ts = ts_intpol
    
    if xlim is not None:
        idxs = (ts>=xlim[0]) & (ts<=xlim[1])
    else:
        idxs = np.ones_like(ts).astype(bool)
    
    plot_tr_smpl_kw = tr_smpl_kw_default.copy()
    plot_tr_smpl_kw.update(tr_kw)
    plot_tr_smpl_kw["label"] = label
    ax.plot(ts[idxs], ys[idxs], **plot_tr_smpl_kw)
    
############################################################################
def plot_stim_on_trace_plot(ax, ts, stim, stim_y0=-125, stim_y1=-111, fill=False):
    """Plot stimulus between ylimits"""
    stim = np.asarray(stim).copy()
    
    ax.set_ylim(stim_y0-5, None)
    stim_rng = (np.max(stim)-np.min(stim))
    
    if stim_rng > 0:
        stim = (stim-np.min(stim))/stim_rng
    else:
        stim = (stim-np.min(stim)) + 1
        
    stim = (stim_y1-stim_y0)*stim+stim_y0
    ax.plot(ts, stim, color='gray', lw=0.8, clip_on=False, zorder=0)
    if fill: ax.fill_between(ts, np.full(stim.size, stim_y0), stim,
                             color='gray', alpha=0.5, clip_on=False, zorder=-20)

############################################################################
def plot_percentiles(
        ax, data, positions=None, means=None, ticks=None,
        color='black', marker='o', showflier=True, qs=(10,90), zero2arrow=False, 
        mean_kw=dict(), line_kw=dict(), outl_kw=dict(), nans_kw=dict(),
        n_nans_allowed=0, nan_text=True, connect=True, connect_alpha=1.0,
    ):
    """ Make percentile plots.
    
    ax : Matplotlib axis
        Axis to plot on.
    
    positions : 1d-iterable or None
        x-positions of data
        
    data : 2d-iterable
        Data to plot at positions.
        First dim should match dim of positions.
        
    means : 1d-iterable or None
        Mean data to plot at positions.
        If means is set, will not plot the mean of data.
        First dim should match dim of positions.
        
    color, marker : matplotlib color / marker
    
    showflier : bool
        If True, will plot fliers.
        
    qs : tuple of ints
        Lower and ubber percentile to plot.
        
    """

    ax2 = ax.twinx()
    ax2.set_ylim(0,1)
    ax2.axis('off')

    if positions is None: positions = np.arange(len(data))
    if means is None: means = np.full(len(data), np.nan)
    
    plot_mean_kw = mean_kw_default.copy()
    plot_line_kw = line_kw_default.copy()
    plot_outl_kw = outl_kw_default.copy()
    plot_nans_kw = nans_kw_default.copy()
    
    plot_mean_kw.update(mean_kw) 
    plot_line_kw.update(line_kw)
    plot_outl_kw.update(outl_kw) 
    plot_nans_kw.update(nans_kw) 
    
    if "color" not in plot_mean_kw: plot_mean_kw['color'] = color
    if "color" not in plot_line_kw: plot_line_kw['color'] = color
    if "color" not in plot_outl_kw: plot_outl_kw['color'] = color
    if "marker" not in plot_mean_kw: plot_mean_kw['marker'] = marker
    
    for i, (position, data_i) in enumerate(zip(positions, data)):
    
        mu, lb, ub, fliers, n_nans = get_bounds_and_fliers(data_i, qs, n_nans_allowed)
        
        if np.isfinite(means[i]):
            mu = means[i] # Overwrite
        else:
            means[i] = mu # ... or save?
        
        label = plot_mean_kw.pop('label', None)
        
        if np.isfinite(mu):
            ax.plot(position, mu, **plot_mean_kw, label=label if i==0 else None)
            
        
        if np.isfinite(mu) and np.isfinite(lb) and np.isfinite(ub):
            ax.errorbar(x=position, y=mu, xerr=None, yerr=[[mu-lb], [ub-mu]], **plot_line_kw, label='_')
 
            if ((mu == 0) or (lb == 0)) and zero2arrow:
                ax2.plot(
                    position, 0.0, marker='$\downarrow$', ms=plot_mean_kw['ms']*1.5,
                    markeredgewidth=0.5, clip_on=False
                )
        
        if n_nans > 0:
            ax.axvline(position, color=color, **plot_nans_kw, label='_')
            if nan_text:
                ax2.text(
                    position, 0.5, f' #NaNs={100*n_nans/data_i.size:.0f}%',
                     ha='center', va='center', rotation=90,
                     bbox=dict(fc='lightgray', ec='None', alpha=0.7, pad=0.1)
                )

        if showflier:
            ax.scatter([position]*fliers.size, fliers, **plot_outl_kw)
            
    if connect:
        plot_mean_kw.pop('marker')
        plot_mean_kw.pop('alpha')
        ax.plot(positions, means, marker=None, **plot_mean_kw, label='_', alpha=connect_alpha)
            
############################################################################
def plot_xy_percentiles(
        ax, datax, datay, annotations=None, annot_kw=dict(textcoords='offset pixels', xytext=(10,-10)),
        color='black', marker='o', showflier=False, qs=(10,90),
        line_kw=dict(), mean_kw=dict(), outl_kw=dict(), nans_kw=dict(),
        n_nans_allowed=0, connect=True, connect_alpha=None,
    ):

    datax = np.asarray(datax)
    datay = np.asarray(datay)

    assert datax.ndim == 2, 'datax and datay must be 2d'
    assert datay.ndim == 2, 'datax and datay must be 2d'
    assert datax.shape[0] == datay.shape[0], 'First dimension of datax and datay must match'
    
    # Prepare plotting parameters.   
    plot_mean_kw = mean_kw_default.copy()
    plot_line_kw = line_kw_default.copy()
    plot_outl_kw = outl_kw_default.copy()
    
    plot_mean_kw.update(mean_kw)
    plot_line_kw.update(line_kw) 
    plot_outl_kw.update(outl_kw) 

    # Add colors and marker.
    if "color" not in plot_mean_kw: plot_mean_kw['color'] = color
    if "color" not in plot_line_kw: plot_line_kw['color'] = color
    if "color" not in plot_outl_kw: plot_outl_kw['color'] = color
    if "marker" not in plot_mean_kw: plot_mean_kw['marker'] = marker

    xmu_list, xlb_list, xub_list, xfl_list = [], [], [], []
    ymu_list, ylb_list, yub_list, yfl_list = [], [], [], []
    
    # Summarize data.
    for i, (datax_i, datay_i) in enumerate(zip(datax, datay)):
        xmu, xlb, xub, xfliers, xn_nans = get_bounds_and_fliers(datax_i, qs, n_nans_allowed)
        ymu, ylb, yub, yfliers, yn_nans = get_bounds_and_fliers(datay_i, qs, n_nans_allowed)

        xmu_list.append(xmu)
        xlb_list.append(xlb)
        xub_list.append(xub)
        xfl_list.append(xfliers)
        
        ymu_list.append(ymu)
        ylb_list.append(ylb)
        yub_list.append(yub)
        yfl_list.append(yfliers)
        
    # Plot means.
    label = plot_mean_kw.pop('label', '_')
    for i, (xmu, ymu) in enumerate(zip(xmu_list, ymu_list)):
        ax.plot(xmu, ymu, **plot_mean_kw, label=label if i == 0 else '_')
    
    if connect:
        plot_mean_kw.pop('marker')
        ls = plot_mean_kw.pop('ls', '-')
        alpha = plot_mean_kw.pop('alpha', 1.0)
        ax.plot(xmu_list, ymu_list, **plot_mean_kw, marker=None, ls=ls, alpha=connect_alpha or alpha, label='_')
        
    # Plot bounds and fliers.
    for idx, (xmu, xlb, xub, xfl, ymu, ylb, yub, yfl) in enumerate(zip(
            xmu_list, xlb_list, xub_list, xfl_list, ymu_list, ylb_list, yub_list, yfl_list
        )):

        if (xmu is not None) and (ymu is not None):
            plot_line_kw.pop('marker', None)
            ax.errorbar(x=xmu, y=ymu, xerr=[[xmu-xlb], [xub-xmu]], yerr=[[ymu-ylb], [yub-ymu]], **plot_line_kw)

        if showflier:
            ax.scatter(xfl, [ymu]*xfl.size, **outl_kw_default, marker='|')
            ax.scatter([xmu]*yfl.size, yfl, **outl_kw_default, marker='_')

        if annotations is not None:
            if annotations[idx] is not None:
                ax.annotate(annotations[idx], (xmu, ymu), **annot_kw)