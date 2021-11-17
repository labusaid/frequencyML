import numpy as np
import h5py
import math


data = np.loadtxt("raw_data/OOK_LowGain.txt",skiprows=1)

print(len(data))
print(data)

samples_per_frame = 1024

num_samples = len(data)
num_frames = math.floor(num_samples/samples_per_frame)

print("Number of samples in the dataset:", num_samples)
print("Number of frames required to fit data", num_frames)


with h5py.File("IQDataBase.hdf5", "a") as f:
    
    dst = f.create_dataset("OOK_Low", shape=(num_frames, samples_per_frame, 2))
    for frame in range(num_frames): 
        
        dst[frame] = data[(frame*1024):1024*(frame+1)]
        
        

print("Successfully Saved Data")



