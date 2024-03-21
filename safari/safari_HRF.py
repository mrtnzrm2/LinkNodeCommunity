import numpy as np

N = 61
time_length = 30
T = time_length / N
frame_times = np.linspace(0, N*T, N)
onset, amplitude, duration = 0.0, 1.0, 1.0
exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)

rf_models = [("glover", "Glover HRF", None)]

import matplotlib.pyplot as plt

from nilearn.glm.first_level import compute_regressor

oversampling = 16

fig = plt.figure(figsize=(9, 4))
for i, (rf_model, model_title, labels) in enumerate(rf_models):
    # compute signal of interest by convolution
    signal, _ = compute_regressor(
        exp_condition,
        rf_model,
        frame_times,
        con_id="main",
        oversampling=oversampling,
    )

from scipy.fft import fft, fftfreq

signal = signal.ravel()
noise = np.random.randn(signal.shape[0]) * 0.1
signal = signal + noise

yf = fft(signal)
xf = fftfreq(N, T)[:N//2]

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.show()