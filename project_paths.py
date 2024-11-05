from pathlib import Path
import socket

station = socket.gethostname()
if station == 'Laptop_naas':
    PROJECT_PATH = Path("C:\\Users\\Nassila\\Documents\\Projects\\Knee_Flexion_auditory_tactile_feedback/Analysis/results")
    RAW_PATH = Path("C:\\Users\\Nassila\\Documents\\Projects\\Knee_Flexion_auditory_tactile_feedback/Raw_data/Young")
    CONF_TEMPLATE = Path("C:\\Users\\Nassila\\Documents\\Projects\\Knee_Flexion_auditory_tactile_feedback/Raw_data/_conf.csv")
elif station == 'w10lloppici':
    PROJECT_PATH = Path("D:/Knee_Flexion_auditory_tactile_feedback/Analysis/results")
    RAW_PATH = Path("D:/Knee_Flexion_auditory_tactile_feedback/Raw_data/Young")
    CONF_TEMPLATE = Path("D:/Knee_Flexion_auditory_tactile_feedback/Raw_data/_conf.csv")
else:
    raise ValueError('No idea where the raw data is')


TEMPLATES_PATH = PROJECT_PATH / "_templates"
MODELS_PATH = PROJECT_PATH / "_models"

MASS_FACTOR = 0.493

targets = {
    "markers_L": [
        "L.PSIS", "L.ASIS", "R.ASIS", "R.PSIS",
        "L.Thigh.Upper", "L.Thigh.Front", "L.Thigh.Rear", "L.Thigh",
        "L.Knee.Lat", "L.Knee.Med",
        "L.Shank.Upper", "L.Shank.Front", "L.Shank.Rear", "L.Shank",
        "L.Ankle.Lat", "L.Ankle.Med",
        "P", "T", "F",
        "L.Heel", "L.Midfoot.Sup", "L.Midfoot.Lat", "L.Toe.Lat", "L.Toe.Med", "L.Toe.Tip"],

    "markers_R": [
        "L.PSIS", "L.ASIS", "R.ASIS", "R.PSIS",
        "R.Thigh.Upper", "R.Thigh.Front", "R.Thigh.Rear", "R.Thigh",
        "R.Knee.Lat", "R.Knee.Med",
        "R.Shank.Upper", "R.Shank.Front", "R.Shank.Rear", "R.Shank",
        "R.Ankle.Lat", "R.Ankle.Med",
        "P", "T", "F",
        "R.Heel", "R.Midfoot.Sup", "R.Midfoot.Lat", "R.Toe.Lat", "R.Toe.Med", "R.Toe.Tip"],

    "emg": ['ST', 'BF', 'RF', 'VM', 'VL']
}

blocks = ['PR00', 'RT01', 'RT02', 'RT03', 'RT04', 'RT15', 'RT24',
                      'FB01',  'FB02', 'FB03',  'TR01', 'TR02'] #PR00 includes min and max events and finishes at RT start
