import os
import shutil
from osrparse import Replay
from tqdm import tqdm


RX_MOD_BITMASK = 128
AP_MOD_BITMASK = 2048
SO_MOD_BITMASK = 4096
RX2_MOD_BITMASK = 8192


MOD_FOLDERS = {
    AP_MOD_BITMASK: "Autopilot",
    RX2_MOD_BITMASK: "Relax2",
    RX_MOD_BITMASK: "Relax",
    SO_MOD_BITMASK: "SpunOut"
}


replay_folder = ''  #add path to your replay


files = [f for f in os.listdir(replay_folder) if f.endswith('.osr')]

for filename in tqdm(files, desc="Processing files", unit="file"):
    original_file_path = os.path.join(replay_folder, filename)

    try:

        replay = Replay.from_path(original_file_path)

        mod_used = None
        for bitmask, folder_name in MOD_FOLDERS.items():
            if (replay.mods & bitmask) != 0:
                mod_used = folder_name
                break

        if mod_used:

            folder_path = os.path.join(replay_folder, mod_used)
            os.makedirs(folder_path, exist_ok=True)
        else:
            folder_path = os.path.join(replay_folder, 'normal')
            os.makedirs(folder_path, exist_ok=True)

        new_file_path = os.path.join(folder_path, filename)

        shutil.move(original_file_path, new_file_path)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
