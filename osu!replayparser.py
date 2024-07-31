from osrparse import Replay

replay = Replay.from_path("")#put in the path to your replay
first_event = replay.replay_data[0]
print(dir(first_event))
print(vars(first_event))
previous_x, previous_y = 256, 500  #I think this works because of how to osu! playfield is setup and people who play in diffrent resoulutons wont need to change the value
previous_time = 0
converted_data_list = []
KEY_LEFT_MOUSE = 1
KEY_RIGHT_MOUSE = 2

def decode_keys(key_mask):
    keys_pressed = []
    if key_mask & KEY_LEFT_MOUSE:
        keys_pressed.append('M1')
    if key_mask & KEY_RIGHT_MOUSE:
        keys_pressed.append('M2')
    return '+'.join(keys_pressed) if keys_pressed else 'None'
for event in replay.replay_data:
    delta_time = event.time_delta
    x = event.x
    y = event.y
    key_press = event.keys
    key_press_str = decode_keys(key_press)
    delta_x = x - previous_x
    delta_y = y - previous_y
    previous_x, previous_y = x, y
    previous_time += delta_time
    converted_data = (delta_time, delta_x, delta_y, key_press_str)
    converted_data_list.append(converted_data)

newfilename = "REPLAYDATA"+replay.replay_hash+".txt"

output_file_path = str(newfilename)

with open(output_file_path, "w") as file:
    for data in converted_data_list:
        file.write(f"{data[0]},{data[1]},{data[2]},{data[3]}\n")

print(f"Data has been written to {output_file_path}")
for data in converted_data_list:
    print(data)
