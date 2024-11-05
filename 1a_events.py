"""""
Extract Events frames from c3d and save in conf file
"""""

from pathlib import Path
import numpy as np
import ezc3d
from pyosim import Conf
from project_paths import *

conf = Conf(project_path=PROJECT_PATH)
participants = conf.get_participants_to_process()

markers_labels = conf.get_conf_field(participant=participants[0], field=['markers', 'targets'])

for iparticipant in participants[1:38]:
    if iparticipant not in ['03maig', '35aner', '37tach', '40svnn']:
        continue
    print(f"\nparticipant: {iparticipant}")

    directories = conf.get_conf_field(
        participant=iparticipant, field=["markers", "data"]
    )
    assigned = conf.get_conf_field(
        participant=iparticipant, field=["markers", "assigned"]
    )

    for idir in directories:
        print(f"\n\tdirectory: {idir}")

        for itrial in Path(idir).glob("*.c3d"):
            # if itrial.stem != '2nd_session':
            #     continue
            if itrial.stem.startswith('static'):
                continue
            if itrial.stem != '1st_session':
                print(f'Checking events in {itrial.stem}')
                reader = ezc3d.c3d(f"{itrial}")
                if itrial.stem == '1st_session_15':
                    event_labels = ["retention_15_start"]
                    event_frames = np.array([reader['header']['points']['first_frame']])
                else:
                    if 'EVENT' in reader['parameters'].keys():
                        event_labels = reader['parameters']['EVENT']['LABELS']['value']
                        event_times = reader['parameters']['EVENT']['TIMES']['value'][-1]
                        event_mask = [ix for ix, x in enumerate(event_labels) if not x.startswith('Camera')]
                        event_labels = [event_labels[i]+f'_{int(event_times[i])}' if event_labels[i]=='start_tape'
                                        else event_labels[i] for i in event_mask ]
                        event_times = [event_times[i] for i in event_mask]

                        rate = reader.c3d_swig.header().frameRate()
                        event_frames = np.round(np.multiply(event_times,rate), 0).astype(int) + 1
                if itrial.stem.startswith('1st_session_p'):
                    fields_conf = {"events": {'1st_session': dict(zip(event_labels, event_frames.tolist()))}}
                elif itrial.stem.startswith('2nd_session_p'):
                    fields_conf = {"events": {'2nd_session': dict(zip(event_labels, event_frames.tolist()))}}
                else:
                    fields_conf = {"events": {itrial.stem: dict(zip(event_labels, event_frames.tolist()))}}
                conf.add_conf_field({iparticipant: fields_conf})