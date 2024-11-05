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
     "RT01": 6,
     "FB01": 24,
     "RT02": 6,
     "FB02": 24,
     "RT03": 6,
     "FB03": 24,
     "RT04": 6,
     "RT15": 6,
     "RT24": 6,
     "TR01": 6,
     "TR02": 6}

conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()

for iparticipant in participants[:]:
    if iparticipant not in ['37tach', '40svnn']:
        continue
    print(f"\nparticipant: {iparticipant}")

    flexion_dict = {'flexion_cycles': {}}
    lat = conf.get_conf_field(participant=iparticipant, field=['leg']).lower()

    blocks = [
        ifile for ifile in (PROJECT_PATH / iparticipant / "1_inverse_kinematic").glob("*.mot")
        if not ifile.stem.startswith('static')
    ]

    try:
        json_path = str(PROJECT_PATH / iparticipant / "_conf.json")
        with open(json_path, 'r') as f:
            events = json.load(f)['events']
    except:
        print(f'Either {iparticipant} or their events do not exist')
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

        knee = np.abs(mot_file[f'knee_angle_{lat}'].values)
        time = mot_file.time.values

        # Identify the flexion cycles within each block

        if os.path.isfile(f"{PROJECT_PATH / iparticipant}/1_inverse_kinematic/Flexion_cycles_{itrial.stem}.svg"):
            first_time = False
            repeat = input(f"The correction has been done for {iparticipant} for the block {itrial.stem}, do you want to repeat it [y/n]?")
            if repeat =='n':
                continue
            elif repeat =='y':
                print('Repeating cycle detection correction')

        if first_time or (not first_time and repeat):
            raw_flexions = ((knee - lowest_angle) / (highest_angle - lowest_angle)) * 90
            flexions = np.clip(raw_flexions, a_min=0, a_max=None)
            # flexions = raw_flexions
            start_ends_all = detect_cycles(flexions-np.median(flexions), time,
                                       event_name1='start', event_name2='rest',
                                       threshold1=10, threshold2=10,
                                       direction1='rising',
                                       #min_duration1=2,  #min_duration2=0, # Optional. Minimal duration of phase 1 and phase 2 in seconds.
                                       max_duration1=30, # max_duration2=np.Inf, #  Optional. Maximal duration of phase 1 and phase 2 in seconds.
                                       # min_peak_height1=-np.Inf, min_peak_height2=-np.Inf, # Optional. Minimum peak value for phase 1 and phase 2.
                                       # max_peak_height1=np.Inf, max_peak_height2 = np.Inf, # Optional. Maximal peak value for phase 1 and phase 2.
                                       filter_input=True, range_to_center=[-1, 1], )

            start_ends = [i for i in start_ends_all if list(i.keys())[0] !='_']
            # Remove the initial cycles in the pre-test and feedback where the experimenter is positioning the knee
            if itrial.stem.startswith(("PR", "FB", "TR01")):
                for idur in range(4):
                    if start_ends[1].get("rest") - start_ends[0].get("start") > 7:
                        start_ends = start_ends[2:]
                    else:
                        break

            # Verify that the number of identified cycles is coherent with the expected number of repetitions per block,
            # to facilitate manual interactive correction
            flexion_start = [x.get("start") for x in start_ends if x.get("start") != None]
            flexion_end = [x.get("rest") for x in start_ends if x.get("rest") != None]
            if (len(flexion_start) != len(flexion_end)) or (len(flexion_start) != n[itrial.stem] * 4):
                text_output = f'Wrongly detected peaks for {iparticipant} at block {itrial.stem} with {len(flexion_start)} against expected {n[itrial.stem] * 4}'
            else:
                text_output = f'All good for {iparticipant} at block {itrial.stem} with {len(flexion_start)} cycle'
            print(text_output)
            # Save the identified cycles as svg for easy post check
            events_to_remove, events_to_add = plot_flexion_cycles(time, flexions, start_ends, itrial.stem, text_output,
                               f"{PROJECT_PATH / iparticipant}/1_inverse_kinematic/Flexion_cycles_{itrial.stem}.svg")

            if len(events_to_remove)>0:
                for iremove in events_to_remove:
                    irem_index = flexion_start.index(iremove)
                    del flexion_start[irem_index]
                    del flexion_end[irem_index]

            if len(events_to_add)>0:
                for istart, irest in zip(events_to_add[0::2], events_to_add[1::2]):
                    flexion_start.append(np.round(istart,2))
                    flexion_end.append(np.round(irest, 2))

            # Creating a list to save in the _conf.json
            cycles = [[flexion_start[iflex], flexion_end[iflex]] for iflex in range(len(flexion_start))]
            flexion_dict['flexion_cycles'][itrial.stem] = cycles


            conf.add_conf_field({iparticipant: flexion_dict})

