import os
import pickle
import numpy as np

# n_examples is number of rows to take
# Returns .pckl file in format (n_examples, n_y_pixels, n_x_pixels, n_t_steps)
def get_preprocessed_data (path, n_examples):
    n_examples_slice = slice(None, n_examples)
    if n_examples == 'ALL':
        n_examples_slice = slice(None, None)

    with open(path, 'rb') as p :
        print("Opening file")
        data = pickle.load(p)[n_examples_slice]
        print("Loaded file")
        return data

# Crops frame into total_side_crops**2 subframes of given crop_size
# Returns array in form (examples, crop_size, crop_size, n_t_steps)
def crop_frames (frames, crop_size, total_side_crops):
    y_pixels, x_pixels = frames.shape[:2]

    start_x = (x_pixels-(crop_size*total_side_crops)) / 2
    start_y = (y_pixels-(crop_size*total_side_crops)) / 2

    crops = []

    for x_pos in range(total_side_crops):
        for y_pos in range(total_side_crops):
            x = int(start_x + crop_size*x_pos)
            y = int(start_y + crop_size*y_pos)

            cropped_region = frames[y:y+crop_size, x:x+crop_size]
            crops.append(cropped_region)


    return np.array(crops)

# Returns array in form (n_clips, crop_size, crop_size, n_t_steps)
def process_data (data, crop_size, total_side_crops):
    examples_size = len(data)
    t_steps = data.shape[-1]

    examples = []
    for i in range(examples_size):
        frames = crop_frames(data[i], crop_size, total_side_crops)
        if i == 0:
            examples = frames
        else:
            examples = np.concatenate((examples, frames), 0)
    
    print("Processed data")

    return np.array(examples)

# Split examples into overlapping windows
# Returns array in form (n_examples, crop_size, crop_size, warmup+t_steps+1)
def window_data (data, warmup, t_steps):
    examples_size = len(data)
    frames_size = data.shape[-1] # Number of frames for each example
    window_size = warmup+t_steps+1

    examples = []

    for example_n in range(examples_size):
        curr_example = data[example_n]

        for frame_n in range(frames_size):
            if frame_n+window_size > frames_size:
                break

            window = curr_example[:, :, frame_n:frame_n+window_size]
            examples.append(window)
    print("Windowed data")
    return np.array(examples)

# Saves dataset as .pkl file at specified path
def save_data (data, path):
    with open(path, 'wb') as p :
        data = pickle.dump(data, p, protocol=4)
        print("Saved data")

data = get_preprocessed_data('../Spiking_model/preprocessed_dataset.pkl', n_examples='ALL')
print(data.shape)
data = process_data(data, crop_size=20, total_side_crops=3)
data = window_data(data, warmup=20, t_steps=8)
np.random.shuffle(data)
save_data(data, "processed_dataset.pkl")
print(data.shape)
