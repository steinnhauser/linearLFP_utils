import scipy.signal as ss
import numpy as np

def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.

    Parameters
    ----------
    x : ndarray
        Array to be downsampled along last axis.
    q : int
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    ndarray
        Downsampled signal.

    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]

def remove_axis_junk(ax, lines=['right', 'top'], ytickpos='left'):
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position(ytickpos)

def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        vlimpadding=None,
        unit='mV',
        ylabels=True,
        color='r',
        linetype='-',
        ztransform=True,
        filter_data=False,
        scalebar_size=10,
        scalebar_rot='vertical',
        on_top_of_eachother=False,
        custom_x_axis=None,
        xlabel = 't (ms)',
        yticklabel_size = 10,
        ylabel_size = 10,
        ylabel = 'channel',
        lw = 1,
        alpha = 1,
        include_x_reference = False,
        custom_yticks = None,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    """helper function to draw line plots"""
    tvec = np.arange(data.shape[1])*dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    vl_input = vlimround

    # apply temporal filter
    if filter_data:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    #subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        # print(data[i][tinds][0])
        if include_x_reference:
            ax.hlines(z,
                tvec[tinds][0] if custom_x_axis is None else custom_x_axis[0],
                tvec[tinds][-1] if custom_x_axis is None else custom_x_axis[-1],
                color="grey", linestyle="--", alpha=0.5, lw=0.8)
        if on_top_of_eachother:
            z=0
        if i == 0:
            ax.plot(tvec[tinds] if custom_x_axis is None else custom_x_axis,
                    data[i][tinds] / vlimround + z,
                    linetype if type(linetype) is str else linetype[i],
                    lw=lw,
                    alpha=alpha,
                    rasterized=False, label=label, clip_on=False,
                    color=color if type(color) is str else color[i])
        else:
            ax.plot(tvec[tinds] if custom_x_axis is None else custom_x_axis,
                    data[i][tinds] / vlimround + z,
                    linetype if type(linetype) is str else linetype[i],
                    lw=lw,
                    alpha=alpha,
                    rasterized=False, clip_on=False,
                    color=color if type(color) is str else color[i])

        if custom_yticks is not None:
            yticklabels.append('ch. %i' % (custom_yticks[i]))
        else:
            yticklabels.append('ch. %i' % (i+1))

        yticks.append(z)

    if scalebar: #  and not vl_input
        if custom_x_axis is None:
            ax.plot([tvec[-1], tvec[-1]], [0,-1] if on_top_of_eachother else [-1,-2],
                lw=2, color='k', clip_on=False)
            ax.text(tvec[-1]+ np.diff(T)*0.02 if vlimpadding is None else tvec[-1]+ np.diff(T)*vlimpadding,
                -0.2 if on_top_of_eachother else -1.2,
                '$2^{' + '{}'.format(np.log2(vlimround)) + '}$ ' + \
                '{0}'.format(unit) if unit=='mV' else '{}%'.format(vlimround*100),
                color='k', rotation=scalebar_rot, va='center', size=scalebar_size)
        else:
            ax.plot([custom_x_axis[-1], custom_x_axis[-1]], [0,-1] if on_top_of_eachother else [-1,-2],
                lw=2, color='k', clip_on=False)
            ax.text(custom_x_axis[-1]+ np.diff(T)*0.02 if vlimpadding is None else custom_x_axis[-1]+vlimpadding,
                -0.2 if on_top_of_eachother else -1.2,
                '$2^{' + '{}'.format(np.log2(vlimround)) + '}$ ' + \
                '{0}'.format(unit) if unit=='mV' else '{}%'.format(vlimround*100),
                color='k', rotation=scalebar_rot, va='center', size=scalebar_size)

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels, size=yticklabel_size)
        ax.set_ylabel(ylabel, labelpad=0.1, size=ylabel_size)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(xlabel, labelpad=0.1)

    return vlimround
