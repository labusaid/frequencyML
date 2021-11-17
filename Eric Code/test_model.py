import h5py
import json
import tensorflow as tf
import numpy as np
from numpy import argwhere
from tensorflow import keras

hdf5_file = h5py.File("IQDataBase.hdf5",  'r')

model = keras.models.load_model('my_model')

to_test = hdf5_file['OOK_High']

print(to_test.shape)

results = model.predict(to_test)

print("test loss, test acc:", results)

predictions = [p.argmax() for p in results]

print(len(predictions))

mod_array_test = np.full(len(predictions), 1) #manully enter the modulation type you are looking for

print(*[(c, a, c==a) for c, a in zip(mod_array_test,predictions)], sep="\n")
