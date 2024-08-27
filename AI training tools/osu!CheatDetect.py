import os
import shutil
from circleguard import Circleguard
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from collections import Counter


API_KEY = "{your api key}"

replay_directory = "{your replay dir}"
cheatrelax_directory = os.path.join(replay_directory, "relax")
timewarp_directory = os.path.join(replay_directory, "timewarp")
cache_db_path = "./cg_cache.db"
os.makedirs(cheatrelax_directory, exist_ok=True)
os.makedirs(timewarp_directory, exist_ok=True)
cg = Circleguard(API_KEY, db_path=cache_db_path)

def is_timewarped(frametimes):

    average_frametime = sum(frametimes) / len(frametimes)
    frametime_counts = Counter(frametimes)
    most_common_frametimes = frametime_counts.most_common()
    multiple_peaks = len(
        [frametime for frametime, count in most_common_frametimes if count > 1 and frametime != 16.67]) > 1
    high_initial_peak = most_common_frametimes[0][0] < 5 and most_common_frametimes[0][1] > 10
    if average_frametime < 16 and not (multiple_peaks or high_initial_peak):
        return True
    return False

def process_replay(filename):
    replay_path = os.path.join(replay_directory, filename)
    replay = cg.ReplayPath(replay_path)
    cg.load(replay)

    replayur = cg.ur(replay)
    replayframetime = cg.frametimes(replay)

    moved_files = []
    if replayur <= 50:
        dest_path = os.path.join(cheatrelax_directory, filename)
        shutil.move(replay_path, dest_path)
        moved_files.append(f"Moved {filename} to {cheatrelax_directory}")
    elif is_timewarped(replayframetime):
        dest_path = os.path.join(timewarp_directory, filename)
        shutil.move(replay_path, dest_path)
        moved_files.append(f"Moved {filename} to {timewarp_directory}")

    return moved_files


def main():
    replay_files = [f for f in os.listdir(replay_directory) if f.endswith(".osr")]

    replay_files.sort()
    num_workers = min(16, multiprocessing.cpu_count())  # if you have more than 16 cpu cores that is wild
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_replay, filename): filename for filename in replay_files}
        with tqdm(total=len(replay_files), desc="Processing replays") as pbar:
            for future in as_completed(futures):
                try:
                    moved_files = future.result()
                    for msg in moved_files:
                        print(msg)
                except Exception as exc:
                    print(f"File {futures[future]} generated an exception: {exc}")
                pbar.update(1)


if __name__ == "__main__":
    main()
