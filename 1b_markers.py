"""""
 Create trc files
"""""

from pathlib import Path
import numpy as np
import opensim as osim
import xarray
from project_paths import blocks
import copy
from pyosim import Conf
from pyosim import Markers3dOsim
from project_paths import *
from Analysis_tools import to_trc, define_blocks, add_nan_markers_xarray
import json



conf = Conf(project_path=PROJECT_PATH)
participants = conf.get_participants_to_process()
update_trc = False

markers_labels = conf.get_conf_field(participant=participants[0], field=['markers', 'targets'])
part_with_sepa_2nd = [i for i in participants if i not in ['01allu', '02diad', '05ozik',
                                                           '06anng', '07vaer', '08fuan', '09haig',
                                                           '10anrg', '11eric', '12seic', '14tose',
                                                           '17laow', '19annn', '24feke',
                                                           '27lalm', '29maer',]]

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
    try:
        json_path = str(PROJECT_PATH / iparticipant / "_conf.json")
        with open(json_path, 'r') as f:
            events = json.load(f)['events']
    except:
        print(f'Either {iparticipant} or their events do not exist')
        continue

    block_bounds = define_blocks(events, blocks)

    for idir in directories:
        print(f"\n\tdirectory: {idir}")

        trials = [itrial for itrial in Path(idir).glob("*.c3d") if itrial.stem!='1st_session'
                  ]
        markers={}
        for itrial in trials:
            assigned = conf.get_conf_field(
                participant=iparticipant, field=["markers", "assigned"]
            )
            print(itrial.stem)
            blacklist = False

            iassign = copy.deepcopy(assigned)
            if iparticipant == '10anrg' and itrial.stem == '1st_session_15':
                iassign[0] = ""
                iassign[3] = ""

            elif iparticipant == "16rohr":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                elif itrial.stem.startswith('1st_session_p'):
                    iassign[0] = ""

            elif iparticipant == "18aran":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""

            elif iparticipant == "22rain":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""
                elif itrial.stem.startswith('1st_session_p'):
                    iassign[1] = ""

            elif iparticipant == "24feke":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""

            elif iparticipant == "27lalm":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""

            elif iparticipant == "72juel":
                if itrial.stem == '1st_session_15':
                    iassign[2] = ""
                if itrial.stem == '2nd_session':
                    iassign[1] = ""
                    iassign[2] = ""

            elif iparticipant == "48joia":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""

            elif iparticipant == "53haur":
                if itrial.stem == '1st_session_15':
                    iassign[2] = ""
                elif itrial.stem.startswith('2nd_session_p'):
                    iassign[1] = ""

            elif iparticipant == "58vika":
                if itrial.stem == '1st_session_15':
                    iassign[2] = ""

            elif iparticipant == "60mael":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""

            elif iparticipant == "73daan":
                if itrial.stem == '1st_session_15':
                    iassign[1] = ""
                    iassign[2] = ""


            nan_idx = [i for i, v in enumerate(iassign) if not v]
            if nan_idx:
                iassign_without_nans = [i for i in iassign if i]
            else:
                iassign_without_nans = iassign

            try:
                new_markers = Markers3dOsim.from_c3d(
                    itrial, usecols=iassign_without_nans, use_cropped_start_time=True, #prefix=":"
                )

                if nan_idx:
                    # if there is any empty assignment, fill the dimension with nan
                    new_markers = add_nan_markers_xarray(new_markers, [assigned[i] for i in nan_idx], nan_idx)
                    print(f"\t{itrial.stem} (NaNs: {nan_idx})")
                else:
                    print(f"\t{itrial.stem}")

                markers[itrial.stem] = new_markers

                # check if dimensions are ok
                if not markers[itrial.stem].data.shape[1] == len(assigned):
                    raise ValueError("Wrong dimensions")
                # break
            except IndexError as e:
                print(f"issue in {itrial}")
                print(e)

                continue  # markers = []

        session_1_markers = xarray.concat([imarker[1][:,:, :-1] for imarker in markers.items() if imarker[0].startswith('1st_session_p')], dim='time')
                # markers.get_labels = markers_labels
        first_frame = session_1_markers.attrs['first_frame']
        last_frame = session_1_markers.shape[-1]+first_frame
        rate = session_1_markers.attrs['rate']
        session_1_markers.attrs['last_frame'] = last_frame
        time_vector = np.arange(start=first_frame, stop=last_frame, step=1 )
        time_vector = time_vector.astype(float)/rate
        session_1_markers['time'] = np.round(time_vector,3)

        if iparticipant in part_with_sepa_2nd:
            session_2_markers = xarray.concat(
                [imarker[1][:, :, :-1] for imarker in markers.items() if imarker[0].startswith('2nd_session_p')],
                dim='time')
            first_frame = session_2_markers.attrs['first_frame']
            last_frame = session_2_markers.shape[-1] + first_frame
            rate = session_2_markers.attrs['rate']
            session_2_markers.attrs['last_frame'] = last_frame
            time_vector = np.arange(start=first_frame, stop=last_frame, step=1)
            time_vector = time_vector.astype(float) / rate
            session_2_markers['time'] =np.round(time_vector,3)

        for iblock, ibound in block_bounds.items():
            # if iblock not in ['RT24', 'TR01', 'TR02']:
            #     continue
            if iblock not in ['RT15', 'RT24', 'TR01', 'TR02']:
                ibound0_idx = np.where(session_1_markers.time.values == ibound[0] / rate)[0][0]
                if iblock == 'RT04':
                    bound_array = session_1_markers[:, :, ibound0_idx:-1]
                else:
                    ibound1_idx = np.where(session_1_markers.time.values == ibound[1] / rate)[0][0]
                    bound_array = session_1_markers[:,:, ibound0_idx:ibound1_idx]

            elif iblock == 'RT15':
                ibound0_idx = np.where(markers['1st_session_15'].time.values == ibound[0] / rate)[0][0]
                bound_array = markers['1st_session_15'][:, :, ibound0_idx:-1]

            elif iblock in ['RT24', 'TR01', 'TR02']:
                if iparticipant in part_with_sepa_2nd:
                    ibound0_idx = np.where(session_2_markers.time.values == ibound[0] / rate)[0][0]
                    if iblock == 'TR02':
                        bound_array = session_2_markers[:, :, ibound0_idx:-1]
                    else:
                        ibound1_idx = np.where(session_2_markers.time.values == ibound[1] / rate)[0][0]
                        bound_array = session_2_markers[:, :, ibound0_idx:ibound1_idx]
                else:
                    ibound0_idx = np.where(np.round(markers['2nd_session'].time.values, 3) == ibound[0] / rate)[0][0]
                    if iblock == 'TR02':
                        bound_array = markers['2nd_session'][:, :, ibound0_idx:-1]
                    else:
                        ibound1_idx = np.where(np.round(markers['2nd_session'].time.values, 3) == ibound[1] / rate)[0][0]
                        bound_array = markers['2nd_session'][:, :, ibound0_idx:ibound1_idx]

            bound_array.attrs['first_frame'] = int(bound_array.time.values[0] * rate)
            bound_array.attrs['last_frame'] = int(bound_array.time.values[-1] * rate) + 1

            trc_filename = (
                f"{PROJECT_PATH / iparticipant}/0_markers/{iblock}.trc"
            )
            if Path(trc_filename).is_file() and update_trc==False:
                continue
            else:
                to_trc(bound_array, filename=Path(trc_filename))

        for itrial in markers.items():
            if itrial[0].startswith('static'):
                trc_filename = (
                    f"{PROJECT_PATH / iparticipant / '0_markers' / itrial[0]}.trc"
                )
                filename = Path(trc_filename)
                to_trc(itrial[1], filename=trc_filename)

