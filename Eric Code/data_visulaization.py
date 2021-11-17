import h5py
import numpy as np
from matplotlib import pyplot as plt

hdf5_file = h5py.File("IQDataBase.hdf5", 'r')

ask8_data = ["ASK8_Low","ASK8_Med","ASK8_High"]
ook_data = ["OOK_Low","OOK_Med","OOK_High"]

data1 = hdf5_file[ook_data[0]][0]
data2 = hdf5_file[ook_data[1]][0]
data3 = hdf5_file[ook_data[2]][0]
x = np.arange(1,1025)


I1 = data1[:,0]
I2 = data2[:,0]
I3 = data3[:,0]

Q1 = data1[:,1]
Q2 = data2[:,1]
Q3 = data3[:,1]


fig, axs = plt.subplots(3,1)
axs[0].plot(x,I1)
axs[1].plot(x,I2)
axs[2].plot(x,I3)

fig2, axs2 = plt.subplots(3,1)
axs2[0].plot(x,Q1)
axs2[1].plot(x,Q2)
axs2[2].plot(x,Q3)


axs[0].title.set_text(ask8_data[0]+" InPhase")
axs[1].title.set_text(ask8_data[1]+" InPhase")
axs[2].title.set_text(ask8_data[2]+" InPhase")

axs2[0].title.set_text(ask8_data[0]+" QuadPhase")
axs2[1].title.set_text(ask8_data[1]+" QuadPhase")
axs2[2].title.set_text(ask8_data[2]+" QuadPhase")
plt.show()
