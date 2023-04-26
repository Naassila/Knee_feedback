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

n = {"Pre_test_0": 5,
     "Retention_block_1": 6,
     "Feedback_block_1": 24,
     "Retention_block_2": 6,
     "Feedback_block_2": 24,
     "Retention_block_3": 6,
     "Feedback_block_3": 24,
     "Retention_block_4": 6,
     "Retention_block_5": 6,
     'RT-24 hours': 6,
     'Transfer': 6}

conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()

for iparticipant in participants[-1:]:
    print(f"\nparticipant: {iparticipant}")

    flexion_dict = {'flexion_cycles': {}}
    lat = conf.get_conf_field(participant=participants[0], field=['leg']).lower()

    # Extract events identified in c3d
    try:
        json_path = str(PROJECT_PATH / iparticipant / "_conf.json")
        with open(json_path, 'r') as f:
            events = json.load(f)['events']
    except:
        print(f'Either {iparticipant} or their events do not exist')
        continue

    for itrial in events.keys():
        flexion_dict['flexion_cycles'][itrial]= {}
        mot_event = events[itrial]
        event_frames = list(mot_event.values())
        mot_file = pd.read_csv(PROJECT_PATH / iparticipant / f"1_inverse_kinematic/{itrial}.mot", header=8, sep='\t')
        knee = mot_file[f'knee_angle_{lat}'].values

        # Identify the mot angles equivalent to the protocol defined 0 (least extended) to 90 (most extended)
        try:
            lowest_angle = np.mean(knee[mot_event['lowest'] - 5:mot_event['lowest'] + 5])  # equivalent to 0
            highest_angle = np.mean(knee[mot_event['highest'] - 5:mot_event['highest'] + 5])  # equivalent to 90
            event_frames = event_frames[2:]
            event_names = list(mot_event.keys())[2:]
        except:
            print('Confirm you had recorded the events of recording the 0° and 90°')

        # Identify the flexion cycles within each block
        for i, ievent in enumerate(event_frames):
            if event_names[i] in ['Retention_block_2']:
                print(f'skipped {event_names[i]}')
                continue

            if i != len(event_frames) - 1:
                flexions = knee[ievent: event_frames[i + 1]]
                time = mot_file.time.values[ievent: event_frames[i + 1]]
            else:
                flexions = knee[ievent:]
                time = mot_file.time.values[ievent:]

            raw_flexions = ((flexions - lowest_angle) / (highest_angle - lowest_angle)) * 90
            flexions = np.clip(raw_flexions, a_min=0, a_max=None)
            start_ends = detect_cycles(flexions, time, event_name1='start', event_name2='rest', threshold1=5,
                                       threshold2=5, direction1='rising',
                                       # min_duration1=2,  #min_duration2=0, # Optional. Minimal duration of phase 1 and phase 2 in seconds.
                                       # max_duration1=np.Inf, max_duration2=np.Inf, #  Optional. Maximal duration of phase 1 and phase 2 in seconds.
                                       # min_peak_height1=-np.Inf, min_peak_height2=-np.Inf, # Optional. Minimum peak value for phase 1 and phase 2.
                                       # max_peak_height1=np.Inf, max_peak_height2 = np.Inf, # Optional. Maximal peak value for phase 1 and phase 2.
                                       filter_input=True, range_to_center=[-1, 1], )

            # Remove the initial cycles in the pre-test and feedback where the experimenter is positioning the knee
            if event_names[i].startswith(("Pre_test", "Feedback")):
                for idur in range(4):
                    if start_ends[1].get("rest") - start_ends[0].get("start") > 7:
                        start_ends = start_ends[2:]
                    else:
                        break

            # Verify that the number of identified cycles is coherent with the expected number of repetitions per block,
            # to facilitate manual interactive correction
            flexion_start = [x.get("start") for x in start_ends if x.get("start") != None]
            flexion_end = [x.get("rest") for x in start_ends if x.get("rest") != None]
            if (len(flexion_start) != len(flexion_end)) or (len(flexion_start) != n[event_names[i]] * 4):
                text_output = f'Wrongly detected peaks for {iparticipant} at {itrial} / {event_names[i]} with {len(flexion_start)} against expected {n[event_names[i]] * 4}'
            else:
                text_output = f'All good for {iparticipant} at {itrial} / {event_names[i]} with {len(flexion_start)} cycle'
            print(text_output)
            # Save the identified cycles as svg for easy post check
            events_to_remove = plot_flexion_cycles(time, flexions, start_ends, itrial, event_names[i], text_output,
                               f"{PROJECT_PATH / iparticipant}/1_inverse_kinematic/Flexion_cycles_{itrial}_{event_names[i]}.svg")

            if len(events_to_remove)>0:
                for iremove in events_to_remove:
                    irem_index = flexion_start.index(iremove)
                    del flexion_start[irem_index]
                    del flexion_end[irem_index]

            # Creating a list to save in the _conf.json
            cycles = [[flexion_start[iflex], flexion_end[iflex]] for iflex in range(len(flexion_start))]
            flexion_dict['flexion_cycles'][itrial][event_names[i]] = cycles

    conf.add_conf_field({iparticipant: flexion_dict})

