import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import plot_utils as pltu
import frame_utils

def plot_MAE_hist(
        data_row=None, MAE_SS=None, MAE_SR=None, MAE_DR=None, avg_fun=np.median,
        ax=None, ax_boxes=None, legend=True, rng=None, log=False
    ):
    """Plot MAE histograms of SS and SR, also plot DR"""
    
    # Extract data.
    if data_row is not None:
        assert (MAE_SS is None) and (MAE_SR is None) and (MAE_DR is None)
        MAE_SS = data_row['MAE_SS']
        MAE_SR = data_row['MAE_SR']
        MAE_DR = data_row['MAE_DR']
        
    # Sanity checks.
    if data_row is not None:  
        if data_row.n_samples == 0: return
    else:
        if MAE_SS is None: return
        elif MAE_SS.size == 0: return
        
    
    # Create axis if None given.
    if ax is None: ax = plt.subplot(111)
    
    if MAE_SS.ndim == 2: # Flatten
        MAE_SS = MAE_SS[np.triu_indices(MAE_SS.shape[0], 1)]
    
    # Transform data?
    if log:
        MAE_SS = np.log10(MAE_SS)
        MAE_SR = np.log10(MAE_SR)
        MAE_DR = np.log10(MAE_DR)
        
    # Transform range?
    if log and rng is not None:
        assert rng[0] > 0, 'log of zero as xlim'
        rng = [np.log10(rng[0]), np.log10(rng[1])] 
        
    for i, MAE in enumerate([MAE_SS, MAE_SR]):
        
        label = ['SS', 'SR'][i]
        color = ['red', 'C0'][i]
        
        sns.histplot(
            MAE, ax=ax, label=label, color=color, bins=15, binrange=rng,
            stat="probability", element="step", fill=True, alpha=0.2
        )
        ax.plot(avg_fun(MAE), 0, color=color, ls='-', marker='d')

    ax.axvline(MAE_DR, color='grey', label='DR', ls='--', zorder=1000)
    if legend: ax.legend(loc='upper left')
        
        
def get_MAE_rng(df):
    """Get bounds from data"""
    MAE_SR_mins, MAE_SS_mins = [], []
    MAE_SR_maxs, MAE_SS_maxs = [], []
    
    found_any = False
    
    for (_, data_row) in df.iterrows():
        if data_row.n_samples > 0:
            found_any = True
            
            MAE_SR_mins.append(np.min(data_row.MAE_SR))
            MAE_SS_mins.append(np.nanmin(data_row.MAE_SS))
            
            MAE_SR_maxs.append(np.max(data_row.MAE_SR))
            MAE_SS_maxs.append(np.nanmax(data_row.MAE_SS))

    if found_any:
        lb = np.min(MAE_SR_mins+MAE_SS_mins)
        ub = np.max(MAE_SR_maxs+MAE_SS_maxs)
    else:
        lb = 0.99
        ub = 1.01
        
    return lb, ub


def plot_MAE_histograms(df, axs=None, titles=True, filename=None, log=False, sharex=True, row_title=True):
    """Plot traces and MAE histograms for different step params""" 
    
    cols = ['method', 'adaptive', 'step_param', 'pert_method']
    if 'stim' in df: cols += ['stim'] 
    for col in cols: assert np.unique(df[col]).size == 1, col
    
    df = df.sort_values(by=['pert_param'], ascending=True)

    if axs is None: fig, axs = pltu.subplots(df.shape[0], 1, squeeze=True, sharex='row')

    rng = get_MAE_rng(df)
        
    ## Plot ###
    for ax, (_, data_row) in zip(axs, df.iterrows()):      
        if titles: ax.set_title(pltu.pert_param2label(data_row.pert_param, data_row.pert_method))
        plot_MAE_hist(data_row=data_row, ax=ax, legend=False, rng=rng if sharex else None, log=log)

    ### Decorate ###
    axs[0].set_ylabel('norm.\n' + 'count')
    if row_title:
        pltu.row_title(ax=axs[0], title=pltu.method2label(
            method=data_row.method.replace('hs', ''), adaptive=data_row.adaptive,
            pert_method=data_row.pert_method), pad=35)
    for ax in axs: sns.despine(ax=ax)
    for ax in axs[1:]: ax.set_ylabel(None)
    for ax in axs: ax.set_yticks([0, ax.get_ylim()[1]*0.95])
    for ax in axs[1:]: ax.set_yticklabels([])
    axs[0].set_yticklabels(["0","1"])
    sns.despine(ax=axs[0])
    pltu.int_format_ticks(axs, which='x')
    if filename is not None: pltu.savefig(filename)
        
        
def plot_MAE_violins(ax, df, xparam='pert_param', log=True):
    """Plot MAE SS vs SR as a function of xparam."""
    
    assert df.shape[0] == np.unique(df[xparam]).size
    
    sns.violinplot(
        ax=ax, data=df2plot_df(df, xparam=xparam, log=log), scale="width",
        x='xparam', y='MAE', split=True, hue='type',
        inner='quartile', cut=0, linewidth=1
    )

    for l in ax.lines:
        l.set_alpha(0.0)
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)

    MAE_DR = np.log10(df.iloc[0].MAE_DR) if log else df.iloc[0].MAE_DR
        
    ax.set_xlabel(None)
    ax.axhline(MAE_DR, c='gray', ls='--', zorder=100, alpha=0.5)
    ax.legend().set_visible(False)
    

def plot_MAE_boxes(ax, df, xparam='pert_param', log=True, fliersize=1):
    """Plot MAE SS vs SR as a function of xparam."""
    
    assert df.shape[0] == np.unique(df[xparam]).size
    
    sns.boxplot(
        ax=ax, data=df2plot_df(df, xparam=xparam, log=log),
        x='xparam', y='MAE', hue='type', linewidth=1, fliersize=fliersize,
    )

    MAE_DR = np.log10(df.iloc[0].MAE_DR) if log else df.iloc[0].MAE_DR
        
    ax.set_xlabel(None)
    ax.axhline(MAE_DR, c='gray', ls='--', zorder=100, alpha=0.5)
    ax.legend().set_visible(False)
    
    
def plot_MAE_pointplot(ax, df, xparam='pert_param', log=True):
    """Plot MAE SS vs SR as a function of xparam."""
    
    assert df.shape[0] == np.unique(df[xparam]).size
    
    sns.pointplot(
        ax=ax, data=df2plot_df(df, xparam=xparam, log=log),
        x='xparam', y='MAE', hue='type', linewidth=1,
    )

    MAE_DR = np.log10(df.iloc[0].MAE_DR) if log else df.iloc[0].MAE_DR
        
    ax.set_xlabel(None)
    ax.axhline(MAE_DR, c='gray', ls='--', zorder=100, alpha=0.5)
    ax.legend().set_visible(False)
    
    
def plot_MAE_ratios_compare_step_params(df):

    assert np.unique(df.pert_method).size == 1
    pert_method = df['pert_method'].iloc[0]
    
    assert np.unique(df.adaptive).size == 1
    adaptive = df['adaptive'].iloc[0]

    g = sns.FacetGrid(
        df, col="step_param", hue="method",
        subplot_kws=dict(xscale='log'), gridspec_kws=dict(), margin_titles=True
    )

    def plot_MAE_ratio(pert_params, MAE_ratios, method, **kwargs):
        """Plot MAE as a function of the perturbation parameter"""
        ax = plt.gca()

        assert np.unique(method).size == 1
        method = method.iloc[0]

        pltu.plot_percentiles(
            ax=ax,
            data=MAE_ratios,
            positions=pert_params,
            showflier=False, connect=True, nan_text=False,
            color=pltu.method2color(method),
            marker=pltu.method2marker(method),
            n_nans_allowed=0,
        )

    
    g.fig.set_size_inches(pltu.FULLPAGE_WIDTH,2)
    g.map(plot_MAE_ratio, 'pert_param', 'MAE_ratios', 'method')
    plt.tight_layout()
    pltu.make_method_legend(
        ax=g.axes.flat[-1], methods=np.unique(df.method),
        legend_kw=dict(loc='lower right'), pert_method=pert_method
    )
    g.set_xlabels(fr"Perturbation parameter ${pltu.pert_method2symbol[pert_method]}$")
    g.set_ylabels(r"$\overline{\mathregular{MAE}}_\mathregular{SS}$ / $\overline{\mathregular{MAE}}_\mathregular{SR}$")
    g.set_titles(
        row_template=pltu.step_param2label(step_param=999, adaptive=adaptive, time_unit='ms').replace("999", "{row_name}"),
        col_template="{col_name}"
    )
    
    for ax in g.axes.flat:
        ax.axhline(1.41, c='gray', ls='--')
        ax.axhline(1, c='gray', ls=':')
        ax.axvline(1, c='gray', ls='--')
    
        ax.set_yticks([0,1,np.sqrt(2)])
        ax.set_yticklabels(["0","1",f"{np.sqrt(2):.2f}"])
    
    g.fig.suptitle(pltu.method2label(method='M', adaptive=adaptive, pert_method=pert_method), ha='left', x=0.0)
    plt.show()
    
    
def plot_MAE_ratios_compare_methods(df):

    assert np.unique(df.pert_method).size == 1
    pert_method = df['pert_method'].iloc[0]   
    
    g = sns.FacetGrid(
        df, col="method", hue="step_param_label", sharex=False, sharey=True,
        subplot_kws=dict(xscale='log'), gridspec_kws=dict(), margin_titles=True
    )
    
    def plot_MAE_ratio(pert_params, MAE_ratios, method, step_param_label, color, **kwargs):
        """Plot MAE as a function of the perturbation parameter"""
        ax = plt.gca()

        assert np.unique(method).size == 1

        method = method.iloc[0]

        pert_params = np.asarray(pert_params)

        pltu.plot_percentiles(
            ax=ax,
            data=MAE_ratios,
            positions=pert_params,
            showflier=False, connect=True, nan_text=False,
            color=color,
            marker='.',
            n_nans_allowed=0,
            mean_kw=dict(label=step_param_label.iloc[0]),
        )

    g.fig.set_size_inches(pltu.FULLPAGE_WIDTH,2)
    g.map(plot_MAE_ratio, 'pert_param', 'MAE_ratios', 'method', 'step_param_label')
    plt.tight_layout()
    g.set_xlabels(fr"${pltu.pert_method2symbol[pert_method]}$")
    g.set_ylabels(r"$\overline{\mathregular{MAE}}_\mathregular{SS}$ / $\overline{\mathregular{MAE}}_\mathregular{SR}$")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flat:
        ax.axhline(1.41, c='gray', ls='--')
        ax.axhline(1, c='gray', ls=':')
        ax.axvline(1, c='gray', ls='--')
        ax.legend(loc='upper left', fontsize=5, frameon=True, framealpha=0.6)

    g.fig.suptitle(pltu.method2label(method='M', pert_method=pert_method), ha='left', x=0.0)
    plt.show()
    

def plot_MAE_SS_SR_and_DR(df):
    """Plot MAE as a function of the pert param"""
    
    assert np.unique(df.pert_method).size == 1
    pert_method = df['pert_method'].iloc[0]
    
    df = df.sort_values(by=['adaptive', 'step_param'], ascending=[False, True], ignore_index=True)
    groups = df.groupby(['adaptive', 'step_param'])
    
    ncols = 1
    nrows = len(groups)
    
    fig, axs = pltu.subplots(
        ncols, nrows, sharex=True, sharey=False, squeeze=False, ysizerow=0.65, yoffsize=0.6,
        xsize='fullwidth' if ncols > 5 else 'text', subplot_kw=dict(xscale='log')
    )
    
    for ax, ((adaptive, step_param), group) in zip(axs.flat, groups):
                
        MAE_SSs = []
        for MAE_SS in group.MAE_SS:
            if isinstance(MAE_SS, np.ndarray):
                MAE_SSs.append(MAE_SS[np.triu_indices(MAE_SS.shape[0], 1)])
            else:
                MAE_SSs.append(np.full((1),np.nan))
        
        pltu.plot_percentiles(
            ax=ax,
            data=MAE_SSs,
            positions=list(group.pert_param),
            showflier=False, connect=True, nan_text=False,
            color='C0', marker='.', n_nans_allowed=0,
        )
        
        pltu.plot_percentiles(
            ax=ax,
            data=group.MAE_SR,
            positions=list(group.pert_param),
            showflier=False, connect=True, nan_text=False,
            color='C1', marker='.', n_nans_allowed=0,
        )
        
        
        ax.set_xlabel(None)
        ax.axhline(group.MAE_DR.iloc[0], c='gray', ls='--', zorder=-100, alpha=1, label='DR')
        ax.set_ylim(0, 5*group.MAE_DR.iloc[0])
        
        pltu.row_title(ax, pltu.step_param2label(adaptive=adaptive, step_param=step_param))
        
    axs.flat[-1].set_xlabel(fr"Perturbation parameter ${pltu.pert_method2symbol[pert_method]}$")
    axs.flat[0].legend(loc='lower right')
    plt.tight_layout()
    fig.align_ylabels()
    plt.show()