import h5py
import json
import tensorflow as tf
import numpy as np
from numpy import argwhere

# Open the dataset
hdf5_file = h5py.File("GOLD_XYZ_OSC.0001_1024.hdf5",  'r')
# Load the modulation classes. You can also copy and paste the content of classes-fixed.txt.
modulation_classes = json.load(open("classes-fixed.json", 'r'))
#List groups of the hdf5 file
list(hdf5_file.keys())

# Read the HDF5 groups
# Dataset is to large for allocated RAM, so every fourth frame will be used, resulting in a fourth of the dataset being used
data = hdf5_file['X'][0:1277952:8]
modulation_onehot = hdf5_file['Y'][0:1277952:8]
snr = hdf5_file['Z'][0:1277952:8]

#creation of test dataset
test_data = data[1277952::128]
test_mod = modulation_onehot[1277952::128]
test_snr = snr[1277952::128]

# Closes the file
hdf5_file.close()

#converting from onehot to integer
mod_array = np.zeros(len(modulation_onehot))
i = 0
for x in modulation_onehot:
    mod_array[i] = np.where(x==1)[0][0]
    i = i + 1


    
#taking the three arrays and converting them to a tf dataset object
dataset = tf.data.Dataset.from_tensor_slices((data, mod_array))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_mod))


#Tensorflow Dataset Object normalization, preprocessing, and customization block
batch_size = 1024
buf_size = 26624
drop_rem = True

dataset = dataset.shuffle(buffer_size=buf_size)
dataset = dataset.batch(batch_size, drop_remainder=drop_rem)

#model variables
img_width, img_height, img_channels = (1024, 2, 1)
opt = 'Adam'
lss = 'sparse_categorical_crossentropy'


model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(64, 3, activation='relu', 
                  input_shape=(img_width, img_height)),
      tf.keras.layers.MaxPooling1D(2, 2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.selu),
      tf.keras.layers.Dense(128, activation=tf.nn.selu),
      tf.keras.layers.Dense(24, activation=tf.nn.softmax)
    ])

model.compile(optimizer=opt, loss=lss, metrics=['accuracy'])

model.fit(dataset, epochs=5)

# Open the dataset
hdf5_file = h5py.File("GOLD_XYZ_OSC.0001_1024.hdf5",  'r')

to_test = hdf5_file['X'][:1277953:16384]
to_test_label = hdf5_file['Y'][:1277953:16384]

#converting from onehot to integer
mod_array_test = np.zeros(len(to_test_label))
i = 0
for x in to_test_label:
    mod_array_test[i] = np.where(x==1)[0][0]
    i = i + 1

# Closes the file
hdf5_file.close()

#print(to_test)
print(mod_array_test)
model.save("my_model")
print("Prediction on test data")
results = model.predict(to_test)
print("test loss, test acc:", results)

predictions = [p.argmax() for p in results]

print(len(predictions))
print(*[(c, a, c == a) for c, a in zip(mod_array_test, predictions)], sep="\n")