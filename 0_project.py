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
for iparticipant in participants[:]:
    if iparticipant not in ['03maig', '35aner', '37tach', '40svnn']:
        continue
    pseudo_in_path = iparticipant

    trials = (
        f"{RAW_PATH}\\{iparticipant}\\Labeled"
    )

    d.update(
        {
            iparticipant: {
                "markers": {"data": [trials]},
                "emg": {"data": [trials]}
            }
        }
    )
conf.add_conf_field(d)



for iparticipant in participants[1:38]:
    if conf.get_conf_field(iparticipant, field=['leg'])=='L':
        leg_drop = "markers_R"
        leg_keep = "markers_L"
    else:
        leg_drop = "markers_L"
        leg_keep = "markers_R"

    par_target = dict(targets)
    del par_target[leg_drop]
    par_target['markers']=par_target.pop(leg_keep)

    for ikind, itarget in par_target.items():
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
