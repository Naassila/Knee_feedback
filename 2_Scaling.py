from pathlib import Path
import os

from pyosim import Conf
from pyosim import Scale

import opensim as osim
from project_paths import *

model = "gait2354"

conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()

for iparticipant in participants[:]:
    print(f"\nparticipant: {iparticipant}")

    mass = conf.get_conf_field(iparticipant, ["mass"])
    height = conf.get_conf_field(iparticipant, ["height"])
    laterality = conf.get_conf_field(iparticipant, ["leg"])

    mass *= MASS_FACTOR
    static_files = [ifile.stem for ifile in (PROJECT_PATH / iparticipant / "0_markers").glob("*.trc") if 'static' in ifile.stem]
    for istatic in static_files:
        path_kwargs = {
            "model_input": f"{MODELS_PATH / model}_{laterality}.osim",
            "model_output": f"{PROJECT_PATH / iparticipant / '_models' / model}_scaled_{istatic[-1]}.osim",
            "xml_input": f"{TEMPLATES_PATH / model}_scaling_{laterality}.xml",
            "model_name": f"{iparticipant}",
            "xml_output": f"{PROJECT_PATH / iparticipant / '_xml' / model}_scaled_{istatic[-1]}.xml",
            "static_path": f"{PROJECT_PATH / iparticipant / '0_markers' / istatic}.trc",
            "add_model": [],
        }

        Scale(mass=mass, **path_kwargs, remove_unused=False)
