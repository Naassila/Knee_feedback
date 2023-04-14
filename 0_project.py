"""""
 Create project
"""""

import shutil
from project_paths import *
from pyosim import Conf, Project

Initialize = False
change_marker_names = False
# ACTUAL PARTICIPANT TO PROCESS #
participant_to_do = -1

project = Project(path=PROJECT_PATH)

if Initialize:
    project.create_project()


shutil.copy(CONF_TEMPLATE, PROJECT_PATH)
project.update_participants()
conf = Conf(project_path=PROJECT_PATH)

conf.check_confs()

participants = conf.get_participants_to_process()
d = {}
for iparticipant in participants:
    pseudo_in_path = (
        iparticipant[0].upper() + iparticipant[1:-1] + iparticipant[-1].upper()
    )

    trials = (
        f"{RAW_PATH}/{iparticipant}"
    )

    d.update(
        {
            iparticipant: {
                "markers": {"data": [trials]},
            }
        }
    )
conf.add_conf_field(d)

# assign channel fields to targets fields
for ikind, itarget in targets.items():
    for iparticipant in participants:
        print(f"\t{iparticipant} - {ikind}")
        if "assigned" not in conf.get_conf_field(iparticipant, [ikind]):
            if change_marker_names:
                #todo: add a dynamic assignment tool
                fields = FieldsAssignment(
                    directory=conf.get_conf_field(iparticipant, field=[ikind, "data"]),
                    targets=itarget,
                    kind=ikind,
                )
                fields_conf = fields.output
            else:
                fields_conf = {ikind: {"targets": itarget, "assigned": itarget}}
            print("\t\tdone")
            conf.add_conf_field({iparticipant: fields_conf})
