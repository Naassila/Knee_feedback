from pathlib import Path
import os

from pyosim import Conf
from pyosim import Scale

import opensim as osim
import numpy as np
from project_paths import *

def add_marker_to_trc(original_trc, filename, new_marker_data):
    table = osim.TimeSeriesTableVec3()
    rate = float(original_trc.getTableMetaDataAsString('CameraRate'))

    # set metadata
    new_labels = list(original_trc.getColumnLabels())+['Hip']
    table.setColumnLabels(new_labels)
    table.addTableMetaDataString('DataRate', str(rate))
    table.addTableMetaDataString('Units', original_trc.getTableMetaDataAsString('Units'))

    time_vector = np.arange(start=0, stop=1 / rate * original_trc.getNumRows(), step=1 / rate)

    for iframe in range(original_trc.getNumRows()):
        a = np.array([[original_trc.getRowAtIndex(iframe).getElt(imarker,0).get(i) for i in range(3)] for imarker in range(trc_data.getNumColumns())])
        a = np.vstack((a, new_marker_data[iframe])).T
        row = osim.RowVectorVec3(
            [osim.Vec3(a[0, i], a[1, i], a[2, i]) for i in range(a.shape[-1])]
        )
        table.appendRow(time_vector[iframe], row)

    adapter = osim.TRCFileAdapter()
    adapter.write(table, filename)

def hip_from_markers(RA, LA, RP, LP, laterality):
    O_pelvis = np.mean((RA, LA), axis=0)
    mid_PSIS = np.mean((RP, LP), axis=0)
    hip = np.zeros((4, RA.shape[0]))

    PW = np.linalg.norm(RA - LA, axis=1)
    PD = np.linalg.norm(mid_PSIS - O_pelvis, axis=1)
    antero_pos = -0.24 * PD - 9.9
    medio_lat = 0.33 * PW + 7.3
    vertical = -0.3 * PW - 10.9
    if laterality=='L':
        hip_loc = np.array([-medio_lat, antero_pos, vertical])
    else:
        hip_loc = np.array([medio_lat, antero_pos, vertical])

    pla = RA - mid_PSIS
    pla = pla / np.linalg.norm(pla, axis=1)[:, None]

    x = RA - LA
    x = x / np.linalg.norm(x, axis=1)[:, None]

    z = np.cross(x, pla)
    z = z / np.linalg.norm(z, axis=1)[:, None]

    y = np.cross(z, x)

    l = x.shape[0]
    pelvis = np.array(
        [[x[:,0], y[:,0], z[:,0]],
         [x[:,1], y[:,1], z[:,1]],
         [x[:,2], y[:,2], z[:,2]],
        ]
    )

    hip_glob = np.array([pelvis[:, :, i] @ hip_loc[:, i] + O_pelvis[i] for i in range(RA.shape[0])])

    return hip_glob


model = "gait2354"
use_hip_center = False

conf = Conf(project_path=PROJECT_PATH)
conf.check_confs()

participants = conf.get_participants_to_process()

for iparticipant in participants[:]:
    print(f"\nparticipant: {iparticipant}")

    mass = conf.get_conf_field(iparticipant, ["mass"])
    height = conf.get_conf_field(iparticipant, ["height"])
    laterality = conf.get_conf_field(iparticipant, ["leg"])

    mass *= MASS_FACTOR
    static_files = [ifile.stem for ifile in (PROJECT_PATH / iparticipant / "0_markers").glob("*.trc") if 'static' in ifile.stem and 'hip' not in ifile.stem]
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

        if use_hip_center:
            trc_data = osim.TimeSeriesTableVec3(path_kwargs['static_path'])
            nrows = trc_data.getDependentColumn('L.PSIS').nrow()
            LP = np.array([[trc_data.getDependentColumn('L.PSIS').getElt(0, i).get(j) for j in range(3)] for i in range(nrows)])
            LA = np.array([[trc_data.getDependentColumn('L.ASIS').getElt(0, i).get(j) for j in range(3)] for i in range(nrows)])
            RP = np.array([[trc_data.getDependentColumn('R.PSIS').getElt(0, i).get(j) for j in range(3)] for i in range(nrows)])
            RA = np.array([[trc_data.getDependentColumn('R.ASIS').getElt(0, i).get(j) for j in range(3)] for i in range(nrows)])

            hip = np.round(hip_from_markers(RA, LA, RP, LP, laterality), 4)

            path_kwargs['static_path'] = path_kwargs['static_path'][:-4]+'_hip.trc'
            path_kwargs["xml_output"] = f"{PROJECT_PATH / iparticipant / '_xml' / model}_scaled_{istatic[-1]}_hip.xml"
            path_kwargs["xml_input"] = f"{TEMPLATES_PATH / model}_scaling_{laterality}_hip.xml"
            add_marker_to_trc(trc_data, path_kwargs['static_path'], hip)


        Scale(mass=mass, **path_kwargs, remove_unused=False)
