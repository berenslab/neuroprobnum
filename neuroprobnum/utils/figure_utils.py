from .metric_utils import compute_kde


def plot_stim(ax, ts, stim):
    """Plot a normalized stimulus"""
    ax.set_facecolor((1, 1, 1, 0))
    ax.plot(ts, stim, c='dimgray', clip_on=False, lw=1)
    ax.set(xlabel='Time (ms)', ylabel=None)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axis('off')


def plot_vs(ax, ts, vs, lw=1, **kwargs):
    """Plot a voltage trace"""
    ax.plot(ts, vs, lw=lw, **kwargs)
    ax.set_ylim(None, 50)
    ax.set(xlabel='Time (ms)', ylabel='v(t)', clip_on=False)
    ax.set_yticks([0, -60])


def plot_events(ax, events=None, event_idx=None, sort_idxs=None, event_traces=None, event_traces_space=2,
                lw=1.5, alpha=0.8, **kwargs):
    """Plot a events, e.g. spike times on a raster plot"""
    if events is not None:
        N = len(events)
        if len(events[0]) == 1:
            event_idx = 0
        else:
            assert (event_idx is not None) and (event_idx < len(events[0]))

        if sort_idxs is not None:
            sorted_events = [events[idx][event_idx] for idx in sort_idxs]
        else:
            sorted_events = [e_list[event_idx] for e_list in events]

        colors = ['C0'] * N
        ticks = [0, N - 1]
        ticklabels = ["1", f"{N}"]
    else:
        colors = []
        ticks = []
        ticklabels = []
        sorted_events = []
        N = 1

    if event_traces is not None:
        for i, (name, (event_trace, color)) in enumerate(event_traces.items()):
            if i > 0 or events is not None:
                sorted_events += [[]] * event_traces_space
                colors += ['w'] * event_traces_space

            sorted_events += [event_trace]
            colors += [color]
            ticks += [N - 1 + (1 + event_traces_space) * (i + int(events is not None))]
            ticklabels += [name]

    ax.eventplot(sorted_events, colors=colors, lw=lw, alpha=alpha, **kwargs)
    ax.set(xlabel='Time (ms)')

    if events is not None:
        ax.set_ylabel('Smp.', va='bottom', y=(N / len(sorted_events)) * 0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.spines['left'].set_bounds([0, N - 1])


def plot_kde(ax, kde_ts, kde):
    ax.fill_between(kde_ts, kde, facecolor='C0', edgecolor='C0', alpha=0.8, lw=0.3)
    ax.set(xlabel='Time (ms)', ylabel='SDF')
    ax.spines['right'].set_bounds([0, kde.max()])

    for side in ['top', 'bottom']:
        ax.spines[side].set_visible(False)

    return ax
