from pyosim import Conf
import numpy as np
import pandas as pd
import seaborn as sns
import re
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from project_paths import *
from Analysis_tools import *

n = {"PR00": 6,
     "FB01": 24,
     "FB02": 24,
     "FB03": 24,
     "TR01": 6,
     }

conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()

for iparticipant in participants[-6:-5]:

    print(f"\nparticipant: {iparticipant}")

    target_dict = {'target_angles': {}}
    lat = conf.get_conf_field(participant=iparticipant, field=['leg']).lower()

    blocks = [
        ifile for ifile in (PROJECT_PATH / iparticipant / "1_inverse_kinematic").glob("*.mot")
        if ifile.stem in list(n.keys())
    ]

    try:
        json_path = str(PROJECT_PATH / iparticipant / "_conf.json")
        with open(json_path, 'r') as f:
            part_data = json.load(f)
            events = part_data['events']
            flexion_cycles = part_data['flexion_cycles']
    except:
        print(f'Either {iparticipant}, their events or their flexion cycles do not exist')
        continue

    # Define range per participant
    mot_file = pd.read_csv(PROJECT_PATH / iparticipant / "1_inverse_kinematic/PR00.mot", header=8, sep='\t')
    knee = np.abs(mot_file[f'knee_angle_{lat}'].values)
    time = mot_file[f'time'].values
    rate = 1 / np.round(np.diff(time, 1), 4)[0]
    lowest_index = np.where(time==events['1st_session']['lowest']/rate)[0][0]
    highest_index = np.where(time == events['1st_session']['highest'] / rate)[0][0]

    lowest_angle = knee[lowest_index] # equivalent to 0
    highest_angle = knee[highest_index] # equivalent to 90

    for itrial in blocks:
        first_time = True
        mot_file = pd.read_csv(itrial, header=8, sep='\t')
        memorization_end = np.sort(np.array(flexion_cycles[itrial.stem])[:, 0])[0]
        memo_phase = mot_file[mot_file.time<memorization_end]

        knee = np.abs(memo_phase[f'knee_angle_{lat}'].values)
        knee = butt_low_filter(knee, order=4, cutoff=6, freq=100 )
        time = memo_phase.time.values

        # Identify the target angles within each block
        if os.path.isfile(f"{PROJECT_PATH / iparticipant}/1_inverse_kinematic/Targets_{itrial.stem}.svg"):
            first_time = False
            repeat = input(f"The correction has been done for {iparticipant} for the targets at {itrial.stem}, do you want to repeat it [y/n]?")
            if repeat =='n':
                continue
            elif repeat =='y':
                print('Repeating target detection correction')

        if first_time or (not first_time and repeat):
            raw_flexions = ((knee - lowest_angle) / (highest_angle - lowest_angle)) * 90
            flexions = np.clip(raw_flexions, a_min=0, a_max=None)

            start_ends_all = detect_cycles(flexions - np.mean(flexions[:400]), time,
                                           event_name1='start', event_name2='rest',
                                           threshold1=10, threshold2=10,
                                           direction1='rising',
                                           min_duration1=5,  #min_duration2=0, # Optional. Minimal duration of phase 1 and phase 2 in seconds.
                                           # max_duration1=30,
                                           # max_duration2=np.Inf, #  Optional. Maximal duration of phase 1 and phase 2 in seconds.
                                           # min_peak_height1=-np.Inf, min_peak_height2=-np.Inf, # Optional. Minimum peak value for phase 1 and phase 2.
                                           # max_peak_height1=np.Inf, max_peak_height2 = np.Inf, # Optional. Maximal peak value for phase 1 and phase 2.
                                           filter_input=True, range_to_center=[-1, 1], )

            # end_targets = time[falling_edge(flexions, 25)[0]]
            end_targets = [x.get("rest") for x in start_ends_all if x.get("rest") != None]


            # Verify that the number of identified cycles is coherent with the expected number of repetitions per block,
            # to facilitate manual interactive correction
            if len(end_targets) != 4 :
                text_output = f'Wrongly detected memorizing target ends for {iparticipant} at block {itrial.stem} with {len(end_targets)} against expected 4'
            else:
                text_output = f'All good for {iparticipant} at block {itrial.stem} with 4 target angles'
            print(text_output)
            # Save the identified cycles as svg for easy post check
            events_to_remove, events_to_add = plot_angle_targets(time, flexions, end_targets, itrial.stem, text_output,
                               f"{PROJECT_PATH / iparticipant}/1_inverse_kinematic/Targets_{itrial.stem}.svg")

            if len(events_to_remove) > 0:
                for iremove in events_to_remove:
                    end_targets = np.delete(end_targets, np.argwhere(end_targets == iremove))

            if len(events_to_add) > 0:
                for iadd in events_to_add:
                    end_targets = np.append(end_targets, np.round(iadd, 2))

            target_dict['target_angles'][itrial.stem] = list(np.sort(end_targets))

            conf.add_conf_field({iparticipant: target_dict})

