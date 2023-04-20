"""""
 Create trc files
"""""

from pathlib import Path
import numpy as np


from pyosim import Conf
from pyosim import Markers3dOsim
from project_paths import *

conf = Conf(project_path=PROJECT_PATH)
participants = conf.get_participants_to_process()

markers_labels = conf.get_conf_field(participant=participants[0], field=['markers', 'targets'])

for iparticipant in participants:
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
            assigned = conf.get_conf_field(
                participant=iparticipant, field=["markers", "assigned"]
            )

            blacklist = False

            nan_idx = [i for i, v in enumerate(assigned) if not v]
            if nan_idx:
                iassign_without_nans = [i for i in assigned if i]
            else:
                iassign_without_nans = assigned

            try:
                markers = Markers3dOsim.from_c3d(
                    itrial, usecols=iassign_without_nans, #prefix=":"
                )
                if nan_idx:
                    # if there is any empty assignment, fill the dimension with nan
                    for i in nan_idx:
                        markers = np.insert(markers, i, "nan", axis=1)
                    print(f"\t{itrial.stem} (NaNs: {nan_idx})")
                else:
                    print(f"\t{itrial.stem}")

                # check if dimensions are ok
                if not markers.data.shape[1] == len(assigned):
                    raise ValueError("Wrong dimensions")
                # break
            except IndexError as e:
                print(f"issue in {itrial}")
                print(e)

                continue  # markers = []

            if not blacklist:
                # markers.get_labels = markers_labels
                trc_filename = (
                    f"{PROJECT_PATH / iparticipant / '0_markers' / itrial.stem}.trc"
                )

                filename = Path(trc_filename)
                markers.to_trc(filename=trc_filename)

                # # Make sure the directory exists, otherwise create it
                # if not filename.parents[0].is_dir():
                #     filename.parents[0].mkdir()
                #
                # # Make sure the metadata are set
                # if 'rate' not in markers.attrs:
                #     raise ValueError('get_rate is empty. Please fill with `your_variable.get_rate = 100.0` for example')
                # if 'units' not in markers.attrs:
                #     raise ValueError('get_unit is empty. Please fill with `your_variable.get_unit = "mm"` for example')
                # if len(markers.channel.data) == 0:
                #     raise ValueError(
                #         'get_labels is empty. Please fill with `your_variable.get_labels = ["M1", "M2"]` for example')
                #
                # table = osim.TimeSeriesTableVec3()
                # rate = markers.attrs['rate']
                #
                # # set metadata
                # table.setColumnLabels(markers.channel.data)
                # table.addTableMetaDataString('DataRate', str(rate))
                # table.addTableMetaDataString('Units', markers.attrs['units'])
                #
                # time_vector = np.arange(start=0, stop=1 / rate * markers.shape[2], step=1 / rate)
                #
                # for iframe in range(markers.shape[-1]):
                #     a = np.round(markers[:, :, iframe].data, decimals=4)
                #     row = osim.RowVectorVec3(
                #         [osim.Vec3(a[0, i], a[1, i], a[2, i]) for i in range(a.shape[-1])]
                #     )
                #     table.appendRow(time_vector[iframe], row)
                #
                # adapter = osim.TRCFileAdapter()
                # adapter.write(table, str(filename))
                # markers.to_trc(filename=trc_filename, reset_time=False)
