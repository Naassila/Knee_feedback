import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

def time_norm(data, time_n=np.linspace(0, 100, 101), axis=-1):
    own_time = np.linspace(time_n[0], time_n[-1], data.shape[0])
    a = data
    if np.isnan(a).any() and not (np.isnan(a).all()):
        first_non_nan = np.where(~np.isnan(a))[0][0]
        last_non_nan = np.where(~np.isnan(a))[0][-1]
        a_red = a[first_non_nan: last_non_nan + 1]

        ratio_first = int(np.ceil(first_non_nan / len(a) * len(time_n)))
        ratio_last = int(np.floor(last_non_nan / len(a) * len(time_n)))
        #         print(len())\n",
        a_red = fill_nan(a_red)
        f = interp1d(np.round(own_time[first_non_nan:last_non_nan + 1], 2), a_red, axis=axis, kind='cubic')
        try:
            y_red = f(time_n[ratio_first:ratio_last])
        except:
            print(f'Original x: {own_time[first_non_nan]} to {own_time[last_non_nan]} ')
            print(f'New x: {time_n[ratio_first]} to {time_n[ratio_last]} ')
            try:
                ratio_first += 1
                y_red = f(time_n[ratio_first:ratio_last])
            except:
                ratio_first += 1
                y_red = f(time_n[ratio_first:ratio_last])
        y = np.empty(len(time_n))
        y[:] = np.nan
        y[ratio_first:ratio_last] = y_red
    #         plt.plot(own_time,a)
    #         plt.plot(own_time[first_non_nan:last_non_nan+1], a_red)
    #         plt.figure()
    #         plt.plot(time_n, y)
    else:
        f = interp1d(own_time, a, axis=axis, kind='cubic')
        y = f(time_n)
    return y

def savgol(time, data, window_length, poly_order, deriv=0):
    """
    Apply a Savitzky-Golay filter on a TimeSeries.
    Notes
    -----
    - The sampling rate must be constant.
    """
    delta = time[1] - time[0]
    try:
        filtered_data = savgol_filter(data,window_length, poly_order, deriv,
                                  delta=delta, axis=0)
    except:
        filtered_data = savgol_filter(data, window_length, poly_order, deriv, mode='nearest',
                                      delta=delta, axis=0)

    return filtered_data

# from kinetics toolkits
def detect_cycles(array, time, event_name1, event_name2,
                  threshold1, threshold2, direction1='rising',
                  min_duration1=0, min_duration2=0, max_duration1=np.Inf, max_duration2=np.Inf,
                  min_peak_height1=-np.Inf, min_peak_height2=-np.Inf, max_peak_height1=np.Inf, max_peak_height2=np.Inf,
                  filter_input=False, range_to_center=[-1, 1],
                  ):
    # lowercase direction1 once
    direction1 = direction1.lower()
    if direction1 != 'rising' and direction1 != 'falling':
        raise ValueError("direction1 must be 'rising' or 'falling'")

    if filter_input:
        data = pd.Series(data=savgol(time, array, 21, 3))
    else:
        data = pd.Series(array)

    data = data - data[(data > range_to_center[0]) & (data < range_to_center[1])].mean()
    data.clip(upper=200, inplace=True)
    events = []

    is_phase1 = True

    for i in range(time.shape[0]):

        if direction1 == 'rising':
            crossing1 = data[i] >= threshold1
            crossing2 = data[i] <= threshold2
        else:
            crossing1 = data[i] <= threshold1
            crossing2 = data[i] >= threshold2

        if is_phase1 and crossing1:

            is_phase1 = False
            events.append({event_name1: time[i]})

        elif (not is_phase1) and crossing2:

            is_phase1 = True
            events.append({event_name2: time[i]})

    # Ensure that we start with event_name1 and that it's not on time0
    while (list(events[0].keys())[0] != event_name1) or (list(events[0].values())[0] == time[0]):
        events = events[1:]

    # Remove last cycle not finished
    if list(events[-1].keys())[0] == event_name1:
        events = events[:-1]

    # Remove cycles where criteria are not reached.
    valid_events = []
    i_event = 0
    while i_event < len(events) - 1:
        time1 = list(events[i_event].values())[0]
        time2 = list(events[i_event + 1].values())[0]
        try:
            time3 = list(events[i_event + 2].values())[0]
            sub_ts2 = data.iloc[np.where(time == time1)[0][0]: np.where(time == time3)[0][0] + 1]
        except IndexError:
            time3 = np.Inf
            sub_ts2 = data.iloc[np.where(time == time1)[0][0]:]

        sub_ts1 = data.iloc[np.where(time == time1)[0][0]: np.where(time == time2)[0][0] + 1]
        # sub_ts2 = data.iloc[np.where(time==time1)[0][0]: np.where(time==time3)[0][0]+1]

        if direction1 == 'rising':
            the_peak1 = sub_ts1.max()
            the_peak2 = sub_ts2.min()
        else:
            the_peak1 = sub_ts1.min()
            the_peak2 = sub_ts2.max()

        if (time2 - time1 >= min_duration1 and  # first phase long enough
                time2 - time1 <= max_duration1 and  # but not too long
                time3 - time2 >= min_duration2 and  # second phase long enough
                time3 - time2 <= max_duration2 and  # but not too long
                the_peak1 >= min_peak_height1 and  # Amplitude of first phase high enough, not just noise
                the_peak1 <= max_peak_height1 and
                the_peak2 >= min_peak_height2 and  # Amplitude of second phase high enough, not just noise
                the_peak2 <= max_peak_height2):
            # Save it.
            if valid_events != []:
                if list(events[i_event].values())[0] == list(valid_events[-1].values())[0]:
                    valid_events = valid_events[:-1]
            valid_events.append(events[i_event])
            valid_events.append(events[i_event + 1])
            if not np.isinf(time3):
                valid_events.append({'_': time3})
            i_event += 2
        else:
            next_event = events[i_event + 2:]
            if len(next_event) > 1:
                next_rec = next([i, x] for i, x in enumerate(next_event) if list(x.keys())[0] == 'rest')
                time2 = next_rec[1].get('rest')
                try:
                    time3 = list(events[i_event + 2 + next_rec[0] + 1].values())[0]
                    sub_ts2 = data.iloc[np.where(time == time1)[0][0]: np.where(time == time3)[0][0] + 1]
                except IndexError:
                    time3 = np.Inf
                    sub_ts2 = data.iloc[np.where(time == time1)[0][0]:]

                sub_ts1 = data.iloc[np.where(time == time1)[0][0]: np.where(time == time2)[0][0] + 1]

                if direction1 == 'rising':
                    the_peak1 = sub_ts1.max()
                    the_peak2 = sub_ts2.min()
                else:
                    the_peak1 = sub_ts1.min()
                    the_peak2 = sub_ts2.max()

                if (time2 - time1 >= min_duration1 and  # first phase long enough
                        time2 - time1 <= max_duration1 and  # but not too long
                        time3 - time2 >= min_duration2 and  # second phase long enough
                        time3 - time2 <= max_duration2 and  # but not too long
                        the_peak1 >= min_peak_height1 and  # Amplitude of first phase high enough, not just noise
                        the_peak1 <= max_peak_height1 and
                        the_peak2 >= min_peak_height2 and  # Amplitude of second phase high enough, not just noise
                        the_peak2 <= max_peak_height2):
                    # Save it.
                    if valid_events != []:
                        if list(events[i_event].values())[0] == list(valid_events[-1].values())[0]:
                            valid_events = valid_events[:-1]
                    valid_events.append(events[i_event])
                    valid_events.append(events[i_event + 2 + next_rec[0]])
                    if not np.isinf(time3):
                        valid_events.append({'_': time3})
                    i_event += 2 + next_rec[0] + 1
                else:
                    i_event += 4

            else:
                i_event += 2

    return valid_events


def on_pick(fig, event):
    event.artist.set_visible(not event.artist.get_visible())
    fig.canvas.draw()

def onpick(event, events_to_remove, f):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        thisline.set_color('r')
        xdata = thisline.get_xdata()[0]
        events_to_remove.append(xdata)
        print('onpick line:', xdata)
    elif isinstance(event.artist, Text):
        f.canvas.stop_event_loop()
        event.artist.set_backgroundcolor('green')


def plot_flexion_cycles(time, flexions, start_ends, itrial, event_name, text_output, path_fig_save):
    f, ax = plt.subplots(1, 1, figsize=(17, 7))
    ax.plot(time, flexions,)
    plt.suptitle(text_output)
    events_to_remove = []
    for i_event in start_ends:
        if list(i_event.keys())[0] == "start":
            t = ax.axvline(
                i_event.get("start"), color="k", linestyle="--", label="Start",
            )
            t.set_picker(True)
        elif list(i_event.keys())[0] == "rest":
            ax.axvline(i_event.get("rest"),color="b",linestyle="--",label="Rest",)
        else:
            ax.axvline(i_event.get("_"), color="r", linestyle="--", label="_")
    ax.set(title=f'{itrial} / {event_name}', ylim=[-10, 100])
    ax.legend(*[*zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    ax.set(xlabel="Time [s]", ylabel="Knee angle [°]")
    ax.text(0.5, -0.2, '✓ Correction done',
                  ha='left', va='center', transform=ax.transAxes,
                  fontsize=20, picker=10, bbox=dict(edgecolor='black', alpha=0.6, linewidth=0))
    plt.tight_layout(pad=1.5)
    f.canvas.mpl_connect('pick_event', lambda event: onpick(event, events_to_remove, f))
    f.canvas.start_event_loop(timeout=-1)
    plt.savefig(path_fig_save, dpi=600, transparent=True)
    plt.close()
    return events_to_remove

# fig, ax = plt.subplots()
# ax.set_title('custom picker for line data')
# line, = ax.plot(np.random. rand(100), rand(100), 'o', picker=line_picker)
# fig.canvas.mpl_connect('pick_event', onpick2)
# plt.show()