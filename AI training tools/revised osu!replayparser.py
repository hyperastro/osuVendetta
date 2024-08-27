from osrparse import Replay

replay = Replay.from_path("")  # put in the path to your replay
first_event = replay.replay_data[0]
print(dir(first_event))
print(vars(first_event))


previous_x, previous_y = 256, 500 #I think this works because of how to osu! playfield is setup and people who play in diffrent resoulutons wont need to change the value
previous_time = 0
convDataArray = []
KEY_LEFT_MOUSE = 1
KEY_RIGHT_MOUSE = 2


def decode_keys(key_mask):
    keys_pressed = []
    if key_mask & KEY_LEFT_MOUSE:
        keys_pressed.append('M1')
    if key_mask & KEY_RIGHT_MOUSE:
        keys_pressed.append('M2')
    return '+'.join(keys_pressed) if keys_pressed else 'None'

# Both the delta of the movements and the absoulte positining is saved so the Neural network can have easier time with temporal analysis and still maintian the context abosulute positing offers
for i in replay.replay_data:
    delta_time = i.time_delta
    x = i.x
    y = i.y
    key_press = i.keys
    key_press_str = decode_keys(key_press)

    deltaX = x - previous_x
    deltaY = y - previous_y
    convertedData = (delta_time, x, y, deltaX, deltaY, key_press_str)
    convDataArray.append(convertedData)

    previous_x, previous_y = x, y
    previous_time += delta_time

newfilename = "REPLAYDATA" + replay.replay_hash + ".txt"
output_file_path = str(newfilename)

with open(output_file_path, "w") as file:
    file.write(f"{replay.mods}\n")
    for data in convDataArray:
        file.write(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]}\n")

print(f"Data has been written to {output_file_path}")
for data in convDataArray:
    print(data)
