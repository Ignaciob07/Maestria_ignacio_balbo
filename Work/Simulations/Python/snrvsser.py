import numpy as np
import matplotlib.pyplot as plt

ser_list = [
    7.44e-01,
    6.97e-01,
    5.46e-02,
    7.52e-03,
    2.75e-04,
    1.00e-06,
    0.00e+00,
    0.00e+00,
    0.00e+00
]
noise_dbs = np.array([ii*3 for ii in range(10)])

start=1
stop=6

plt.figure()
plt.plot(noise_dbs[start:stop], ser_list[start:stop])
plt.title("SNR vs SER")
plt.xlabel("SNR db")
plt.ylabel("SER")
plt.grid(True)
plt.show()
