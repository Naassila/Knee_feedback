from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import CheckButtons, Button
from pyomeca import Analogs
import opensim as osim
import re
# import mplcursors
from matplotlib.widgets import MultiCursor
# import PyQt5.QtWidgets

# class FieldsAssignement()

class MVC:
    def __init__(self, directories, channels, params, plot_mva=False):
        self.trials_path = []
        for idir in directories:
            for ifile in Path(idir).glob('*.c3d'):
                if ifile.stem != '1st_session':
                    self.trials_path.append(ifile)
        self.channels = channels
        self.plot_mva = plot_mva
        self.rate = 0
        self.params = params
        self.trials = self.read_files()

    def read_files(self):
        trials=[]
        for itrial in self.trials_path:
            try:
                emg = Analogs.from_c3d(itrial)
                self.rate = emg.rate
                emg = emg.fillna(0)
                emg = (emg.meca.band_pass(order=self.params['order'], cutoff=self.params['band_pass_cutoff'], freq=emg.rate)
                 .meca.center()
                 .meca.abs()
                 .meca.low_pass(order=self.params['low_pass_order'], cutoff=self.params["low_pass_cutoff"], freq= emg.rate)
                 )
                trials.append(emg)
            except IndexError:
                emg = []

        return trials

    def get_mvc(self, time=1):
        mvc = {}
        seconds = int(time*self.rate)
        data = xarray.concat(self.trials, dim='time')
        fig, ax = plt.subplots(1, 5, sharey=False, sharex=True)
        for i,ichan in enumerate(data.channel):
            sorted_values = np.sort(data.sel(channel=ichan).data)
            mvc[str(ichan.data)]=np.nanmean(sorted_values[-seconds:])
            ax[i].plot(sorted_values[-int(10*seconds):], 'b-', label='Sorted activation')
            ax[i].axhline(y=mvc[str(ichan.data)], c='k', ls='--', label='MVA')
            ax[i].set_title(str(ichan.data))
        plt.title(f'Last {int(10*seconds/self.rate)} seconds')
        plt.legend()

        return mvc


def to_trc(data, filename):
    filename = Path(filename)
    # Make sure the directory exists, otherwise create it
    if not filename.parents[0].is_dir():
        filename.parents[0].mkdir()

    # Make sure the metadata are set
    if 'rate' not in data.attrs:
        raise ValueError('get_rate is empty. Please fill with `your_variable.get_rate = 100.0` for example')
    if 'units' not in data.attrs:
        raise ValueError('get_unit is empty. Please fill with `your_variable.get_unit = "mm"` for example')
    if len(data.channel.data) == 0:
        raise ValueError(
            'get_labels is empty. Please fill with `your_variable.get_labels = ["M1", "M2"]` for example')

    rate = data.attrs['rate']
    time_vector = data.time.values
    # time_vector = np.arange(start=0, stop=1 / rate * data.shape[2], step=1 / rate)
    # time_vector += data.attrs['first_frame'] / rate


    table = osim.TimeSeriesTableVec3()
    # set metadata
    table.setColumnLabels(data.channel.data)
    table.addTableMetaDataString('DataRate', str(rate))
    table.addTableMetaDataString('Units', data.attrs['units'])



    # for iframe in range(data.shape[-1]):
    #     a = np.round(data[:, :, iframe].data, decimals=4)
    #     row = osim.RowVectorVec3(
    #         [osim.Vec3(a[0, i], a[1, i], a[2, i]) for i in range(a.shape[-1])]
    #     )
    #     table.appendRow(time_vector[iframe], row)

    [table.appendRow(time_vector[iframe], osim.RowVectorVec3(
        list(map(lambda n: osim.Vec3(np.round(n, decimals=5)), data[:3, :, iframe].data.T)))) for iframe in
     range(data.shape[-1])]

    adapter = osim.TRCFileAdapter()
    adapter.write(table, str(filename))

def add_nan_markers_xarray(old_array, missing_markers, idx_markers):
    if len(missing_markers)!=len(idx_markers):
        raise ValueError('Mismatch in length between the name list of missing markers and their location list')

    coords = {}
    coords['axis'] =  ["x", "y", "z", "ones"]
    old_markers = old_array.channel.values
    old_data = old_array.data
    for i, iid in enumerate(idx_markers):
        old_markers = np.insert(old_markers, iid, missing_markers[i])
        old_data = np.insert(old_data, iid, 'nan', axis=1)
    coords['channel'] = old_markers
    coords["time"] = old_array.time.values

    new_data =  xr.DataArray(data=old_data,
                             dims=("axis", "channel", "time"),
                             coords=coords,
                             name="markers",)
    new_data.attrs = old_array.attrs
    return new_data

def define_blocks(events, blocks_name):
    blocks_bound = {}
    blocks_bound[blocks_name[0]] = [events['1st_session']['lowest'] - 20,
                                    events['1st_session']['retention_1_start'] + 5]
    #Retention blocks
    blocks_bound[blocks_name[1]] = [events['1st_session']['retention_1_start'] - 5,
                                    events['1st_session']['feedback_1_start'] + 5]
    blocks_bound[blocks_name[2]] = [events['1st_session']['retention_2_start'] - 5,
                                    events['1st_session']['feedback_2_start'] + 5]
    blocks_bound[blocks_name[3]] = [events['1st_session']['retention_3_start'] - 5,
                                    events['1st_session']['feedback_3_start'] + 5]
    blocks_bound[blocks_name[4]] = [events['1st_session']['retention_4_start'] - 5,
                                    'end']

    blocks_bound[blocks_name[5]] = [max(events['1st_session_15']['retention_15_start'], 0),
                                    'end']
    blocks_bound[blocks_name[6]] = [max(events['2nd_session']['pre-test_start'] - 5, 0),
                                    events['2nd_session']['feedback_0_start'] + 5]


    #Feedback_blocks
    blocks_bound[blocks_name[7]] = [events['1st_session']['feedback_1_start'] - 5,
                                    events['1st_session']['retention_2_start'] + 5]
    blocks_bound[blocks_name[8]] = [events['1st_session']['feedback_2_start'] - 5,
                                    events['1st_session']['retention_3_start'] + 5]
    blocks_bound[blocks_name[9]] = [events['1st_session']['feedback_3_start'] - 5,
                                    events['1st_session']['retention_4_start'] + 5]

    #Transfer blocks
    try:
        blocks_bound[blocks_name[10]] = [events['2nd_session']['feedback_0_start'] - 5,
                                     events['2nd_session']['setup_start'] + 5]
    except:
        blocks_bound[blocks_name[10]] = [events['2nd_session']['feedback_0_start'] - 5,
                                         events['2nd_session']['Setup_start'] + 5]
    blocks_bound[blocks_name[11]] = [events['2nd_session']['retention_2_start'] - 5,
                                     'end' ]

    cut_offs = np.array([v for k, v in events['1st_session'].items() if k.startswith('start_tape')])
    for iblock, ibound in blocks_bound.items():
        if iblock not in ['RT04', 'RT15', 'TR02', 'RT24', 'TR01']:
            mask = (cut_offs > ibound[0]) & (cut_offs < ibound[1])
            if mask.sum()!=0:
                ibound[1] = cut_offs[mask][0]


    return blocks_bound

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

def butt_low_filter(data, order, cutoff, freq):
    nq = freq/2
    corr_freq = cutoff / nq
    b, a = butter(N=order, Wn= corr_freq, btype='low')
    return filtfilt(b, a, data)

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

def falling_edge(data, thresh):
    sign = data <= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)
    return pos
def on_pick(fig, event):
    event.artist.set_visible(not event.artist.get_visible())
    fig.canvas.draw()

def on_keyboard(event, check_box):
    print('press', event.key)
    if event.key == 'z':
        check_box.eventson = False
        current_status = check_box.get_status()
        if not current_status[0]:
            check_box.set_active(0)
        if current_status[1]:
            check_box.set_active(1)
        if current_status[2]:
            check_box.set_active(2)
        check_box.eventson = True
    elif event.key == 'x':
        check_box.eventson = False
        current_status = check_box.get_status()
        if current_status[0]:
            check_box.set_active(0)
        if not current_status[1]:
            check_box.set_active(1)
        if current_status[2]:
            check_box.set_active(2)
        check_box.eventson = True

def on_keyboard_target(event, check_box):
    print('press', event.key)
    if event.key == 'z':
        check_box.eventson = False
        current_status = check_box.get_status()
        if not current_status[0]:
            check_box.set_active(0)
        if current_status[1]:
            check_box.set_active(1)
        if current_status[2]:
            check_box.set_active(2)
        check_box.eventson = True
    elif event.key == 'x':
        check_box.eventson = False
        current_status = check_box.get_status()
        if current_status[0]:
            check_box.set_active(0)
        if current_status[1]:
            check_box.set_active(1)
        if not  current_status[2]:
            check_box.set_active(2)
        check_box.eventson = True


def onpick(event, events_to_remove, f, check_box, t_remove):
    if isinstance(event.artist, Line2D) and check_box.get_status() == [0,0,1]:
        thisline = event.artist
        thisline.set_color('r')
        xdata = thisline.get_xdata()[0]
        if xdata not in events_to_remove:
            events_to_remove.append(xdata)
            print('onpick line:', xdata)
            t_remove.set_text(int(t_remove.get_text()) + 1)
    elif isinstance(event.artist, Text):
        f.canvas.stop_event_loop()
        event.artist.set_backgroundcolor('green')

def double_left_mouse(event, events_to_add, f, check_box, t_start, t_rest):
    if (event.button == plt.MouseButton.LEFT
            and event.dblclick == True
            and check_box.get_status()==[1,0,0]):
        check_box.eventson = False
        events_to_add.append(event.xdata)
        line = f.axes[0].axvline(event.xdata, color='#9467bd', linestyle="--",  label='Manually identified start')
        f.axes[0].draw_artist(line)
        check_box.set_active(0)
        check_box.set_active(1)
        check_box.eventson = True
        t_start.set_text(int(t_start.get_text())+1)
    elif (event.button == plt.MouseButton.LEFT
          and event.dblclick == True
          and check_box.get_status()==[0,1,0]):
        events_to_add.append(event.xdata)
        line = f.axes[0].axvline(event.xdata, color='#8c564b', linestyle="--", label='Manually identified end')
        f.axes[0].draw_artist(line)
        t_rest.set_text(int(t_rest.get_text()) + 1)
        check_box.eventson = False
        check_box.set_active(0)
        check_box.set_active(1)
        check_box.eventson = True

def double_left_mouse_end_only(event, events_to_add, f, check_box, t_start, t_rest):
    if event.button == plt.MouseButton.LEFT and event.dblclick == True and check_box.get_status()==[1,0,0]:
        # check_box.eventson = False
        if event.xdata not in events_to_add:
            events_to_add.append(event.xdata)
            line = f.axes[0].axvline(event.xdata, color='#9467bd', linestyle="--",  label='Manually identified start')
            f.axes[0].draw_artist(line)
            # check_box.set_active(0)
            # check_box.set_active(1)
            # check_box.eventson = True
            t_start.set_text(int(t_start.get_text())+1)


def plot_flexion_cycles(time, flexions, start_ends, itrial, text_output, path_fig_save):
    detected = [int(x) for x in re.findall('\d+', text_output)[-2:]]
    f, ax = plt.subplots(1, 1, figsize=(17, 7))
    ax.plot(time, flexions,)
    plt.suptitle(text_output)
    events_to_remove = []
    events_to_add = []
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
    ax.set(title=f'{itrial}', ylim=[-10, 100])
    ax.legend(*[*zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    ax.set(xlabel="Time [s]", ylabel="Knee angle [°]")
    ax.text(0.4, -0.2, '✓ Correction done',
                  ha='left', va='center', transform=ax.transAxes,
                  fontsize=20, picker=10, bbox=dict(edgecolor='black', alpha=0.6, linewidth=0))
    plt.subplots_adjust(left = 0.3, right=0.95, bottom=0.2)
    axcheck = plt.axes([0.03, 0.4, 0.2, 0.15])
    add_remove_check_box = CheckButtons(axcheck,
                                        labels=['Add start',
                                                'Add rest\n(immediately after add start)',
                                                'Select starts to remove cycle'],
                                        actives = [0,0,1])
    check_color = ['#9467bd', '#8c564b', 'r']
    [ilabel.set_color(check_color[icolor]) for icolor, ilabel in enumerate(add_remove_check_box.labels)]
    axcheck.text(0.03, -0.10, "Modification status", ha='left', va='center', transform=axcheck.transAxes, color='k')
    t_add_start = axcheck.text(0.03, -0.2, 0, ha='left', va='center', transform=axcheck.transAxes, color='#9467bd')
    t_add_rest = axcheck.text(0.03, -0.3, 0, ha='left', va='center', transform=axcheck.transAxes, color='#8c564b')
    t_remove = axcheck.text(0.03, -0.4, 0, ha='left', va='center', transform=axcheck.transAxes, color= 'r')
    # plt.tight_layout(pad=1.5)
    f.canvas.mpl_connect('pick_event', lambda event: onpick(event, events_to_remove, f, add_remove_check_box, t_remove))
    f.canvas.mpl_connect('button_press_event', lambda event: double_left_mouse(event, events_to_add, f, add_remove_check_box, t_add_start, t_add_rest))
    f.canvas.mpl_connect('key_press_event', lambda event: on_keyboard(event, add_remove_check_box))
    f.canvas.start_event_loop(timeout=-1)

    plt.savefig(path_fig_save, dpi=600, transparent=True)
    plt.close()
    return events_to_remove, events_to_add

def plot_start_5s(event, ax, targets, events_to_remove, events_to_add):
    if len(events_to_remove)>0:
        for iremove in events_to_remove:
            targets = np.delete(targets, np.argwhere(targets == iremove))

    if len(events_to_add)>0:
        for iadd in events_to_add:
            targets = np.append(targets, np.round(iadd, 2))

    targets_start = targets - 5
    for it in targets_start:
        ax.axvline(it, color="g", linestyle="--", label="Start", )

def plot_angle_targets(time, flexions, target_ends, itrial, text_output, path_fig_save):
    detected = [int(x) for x in re.findall('\d+', text_output)[-2:]]
    f, ax = plt.subplots(1, 1, figsize=(17, 7))
    plot1 = ax.plot(time, flexions,)
    axButn1 = plt.axes([0.1, 0.1, 0.1, 0.1])
    btn1 = Button(axButn1, 'Show 5s range', color='grey')

    cursor = MultiCursor(f.canvas, [ax], color='k', lw=0.5, horizOn=True, vertOn=True)
    # cursor = mplcursors.cursor(plot1, hover=True)
    final_plot = lambda x: plot_start_5s(x, ax, target_ends, events_to_remove, events_to_add)
    btn1.on_clicked(final_plot)
    plt.suptitle(text_output)
    events_to_remove = []
    events_to_add = []
    for i_event in target_ends:
        t = ax.axvline(
            i_event, color="k", linestyle="--", label="End",
        )
        t.set_picker(True)
        # if list(i_event.keys())[0] == "start":
        #     t = ax.axvline(
        #         i_event.get("start"), color="k", linestyle="--", label="Start",
        #     )
        #     t.set_picker(True)
        # elif list(i_event.keys())[0] == "rest":
        #     ax.axvline(i_event.get("rest"),color="b",linestyle="--",label="Rest",)
        # else:
        #     ax.axvline(i_event.get("_"), color="r", linestyle="--", label="_")
    ax.set(title=f'{itrial}', ylim=[-10, 100])
    ax.legend(*[*zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    ax.set(xlabel="Time [s]", ylabel="Knee angle [°]")
    ax.text(0.4, -0.2, '✓ Correction done',
                  ha='left', va='center', transform=ax.transAxes,
                  fontsize=20, picker=10, bbox=dict(edgecolor='black', alpha=0.6, linewidth=0))
    plt.subplots_adjust(left = 0.3, right=0.95, bottom=0.2)
    axcheck = plt.axes([0.03, 0.4, 0.2, 0.15])
    add_remove_check_box = CheckButtons(axcheck,
                                        labels=['Add end',
                                                '',
                                                'Select ends to remove'],
                                        actives = [0,0,1])
    check_color = ['#9467bd', '#ffffff', 'r']
    [ilabel.set_color(check_color[icolor]) for icolor, ilabel in enumerate(add_remove_check_box.labels)]
    axcheck.text(0.03, -0.10, "Modification status", ha='left', va='center', transform=axcheck.transAxes, color='k')
    t_add_end = axcheck.text(0.03, -0.2, 0, ha='left', va='center', transform=axcheck.transAxes, color='#9467bd')
    t_add_rest = axcheck.text(0.03, -0.3, 0, ha='left', va='center', transform=axcheck.transAxes, color='#8c564b')
    t_remove = axcheck.text(0.03, -0.4, 0, ha='left', va='center', transform=axcheck.transAxes, color= 'r')
    # plt.tight_layout(pad=1.5)
    f.canvas.mpl_connect('pick_event', lambda event: onpick(event, events_to_remove, f, add_remove_check_box, t_remove))
    f.canvas.mpl_connect('button_press_event', lambda event: double_left_mouse_end_only(event, events_to_add, f, add_remove_check_box, t_add_end, t_add_rest))
    f.canvas.mpl_connect('key_press_event', lambda event: on_keyboard_target(event, add_remove_check_box) )
    f.canvas.start_event_loop(timeout=-1)

    plt.savefig(path_fig_save, dpi=600, transparent=True)
    plt.close()
    return events_to_remove, events_to_add