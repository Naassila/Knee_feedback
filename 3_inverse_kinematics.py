from pyosim import Conf
from pyosim import InverseKinematics
from project_paths import *

model = "gait2354"

conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()

for iparticipant in participants[-1:]:
    print(f"\nparticipant: {iparticipant}")

    models = [
        ifile for ifile in (PROJECT_PATH / iparticipant / '_models').glob("*.osim") if "markers" in ifile.stem
    ]

    for imodel in models:

        trials = [
            ifile for ifile in (PROJECT_PATH / iparticipant / "0_markers").glob("*.trc") if imodel.stem[-9:-8] in ifile.stem
        ]

        path_kwargs = {
            "model_input": str(imodel),
            "xml_input": f"{TEMPLATES_PATH / model}_ik.xml",
            "xml_output": f"{PROJECT_PATH / iparticipant / '_xml' / model}_ik.xml",
            "mot_output": f"{PROJECT_PATH / iparticipant / '1_inverse_kinematic'}",
        }

        InverseKinematics(
            **path_kwargs,
            trc_files=trials[:],
            onsets=[],
            prefix="",
            multi=False,
            report_errors=True,
            report_marker_location=True
        )

