from pyosim import Conf
from pyosim import InverseKinematics
from project_paths import *

model = "gait2354"
repeat_IK = False
conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()
blocks_day = {'1': ['PR00', 'RT01', 'RT02', 'RT03', 'RT04', 'RT15', 'FB01',  'FB02', 'FB03', 'static_1'],
              '2' : ['RT24', 'TR01', 'TR02', 'static_2']}

for iparticipant in participants[:-5]:
    if iparticipant not in ['03maig', '35aner', '37tach', '40svnn']:
        continue
    print(f"\nparticipant: {iparticipant}")
    laterality = conf.get_conf_field(iparticipant, ["leg"])

    models = [
        ifile for ifile in (PROJECT_PATH / iparticipant / '_models').glob("*.osim") if "markers" in ifile.stem
    ]

    for imodel in models[:]:

        trials = [
            ifile for ifile in (PROJECT_PATH / iparticipant / "0_markers").glob("*.trc")
            if ifile.stem in blocks_day[imodel.stem[-9:-8]]
        ]

        if not repeat_IK:
            previous_IK = [ifile.stem for ifile in (PROJECT_PATH / iparticipant / "1_inverse_kinematic").glob("*.mot")]
            if len(previous_IK) == 14:
                continue
            else:
                trials = [itrial for itrial in trials if itrial.stem not in previous_IK]
                if len(trials) == 0:
                    continue


        path_kwargs = {
            "model_input": str(imodel),
            "xml_input": f"{TEMPLATES_PATH / model}_ik_{laterality}.xml",
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

