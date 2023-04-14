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

for iparticipant in participants[-1:]:
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
            reader = ezc3d.c3d(f"{itrial}")
            if 'EVENT' in reader['parameters'].keys():
                event_labels = reader['parameters']['EVENT']['LABELS']['value']
                event_times = reader['parameters']['EVENT']['TIMES']['value'][-1]
                rate = reader.c3d_swig.header().frameRate()
                event_frames = np.round((event_times * rate), 0).astype(int) + 1
                fields_conf = {"events": {itrial.stem: dict(zip(event_labels, event_frames.tolist()))}}
                conf.add_conf_field({iparticipant: fields_conf})