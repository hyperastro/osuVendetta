from ossapi import Ossapi
import time
import os


api = Ossapi({clientID}, '{ClientSecret}','{callback url}') #put your own api key and put the callback url to something like http://localhost:727/ or a port that isnt being used 


def save_replay(data, filename):
    with open(filename, 'wb') as file:
        file.write(data)



def read_replay_ids(filename):
    try:
        with open(filename, 'r') as file:
            ids = file.readlines()
        return [id.strip() for id in ids if id.strip()]
    except FileNotFoundError:
        print(f"The file at {filename} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



def downloadreplays(ids, save_directory):
    os.makedirs(save_directory, exist_ok=True)  
    for score_id in ids:
        time.sleep(7) # This is a random value I found to work so to not send in to many api Download requests(probably not optimal)
        try:
            replay = api.download_score_mode(mode='osu', score_id=score_id, raw=True)
            if isinstance(replay, bytes):
                replay_data = replay
            else:
                replay_data = replay.data  

            filename = os.path.join(save_directory, f'{score_id}.osr')
            save_replay(replay_data, filename)
            print(f'Replay {score_id} saved to {filename}')
        except Exception as e:
            print(f"An error occurred while downloading replay {score_id}: {e}")


if __name__ == "__main__":
    replay_ids_file = 'notpickedup.txt' 
    save_directory = 'replays'  

    replay_ids = read_replay_ids(replay_ids_file)
    if replay_ids:
        downloadreplays(replay_ids, save_directory)
    else:
        print("No replay IDs to process.")
