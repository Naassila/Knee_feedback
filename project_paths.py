from pathlib import Path
import socket

station = socket.gethostname()
if station == 'Laptop_naas':
    PROJECT_PATH = Path("'C:\\Users\\Nassila\\Documents\\Projects\\Knee_Flexion_auditory_tactile_feedback/Analysis/results")
    RAW_PATH = Path("'C:\\Users\\Nassila\\Documents\\Projects\\Knee_Flexion_auditory_tactile_feedback/Raw_data/Young")
    CONF_TEMPLATE = Path("'C:\\Users\\Nassila\\Documents\\Projects\\Knee_Flexion_auditory_tactile_feedback/Raw_data/_conf.csv")
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
    "markers": [
        "L.PSIS", "L.ASIS", "R.ASIS", "R.PSIS",
        "L.Thigh.Upper", "L.Thigh.Front", "L.Thigh.Rear", "L.Thigh",
        "L.Knee.Lat", "L.Knee.Med",
        "L.Shank.Upper", "L.Shank.Front", "L.Shank.Rear", "L.Shank",
        "L.Ankle.Lat", "L.Ankle.Med",
        "P", "T", "F",
        "L.Heel", "L.Midfoot.Sup", "L.Midfoot.Lat", "L.Toe.Lat", "L.Toe.Med", "L.Toe.Tip"]
}
