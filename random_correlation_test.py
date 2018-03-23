import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import Debye_spectrum_3 as ds

omegaD = 10
t_num = 50000
dt = 0.001
Ntraj = 1000
sampling = 2
points = 25
omegagrid = np.linspace(0.0, 2*omegaD, points)
# test = Generator(n=2,
#                  omegaD=1,
#                  temperature=1,
#                  dt=0.01,
#                  t_num=t_num,
#                  Ntraj=Ntraj)

n2 = ds.Generator(n=2,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)

n4 = ds.Generator(n=4,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)

n6 = ds.Generator(n=6,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)

n8 = ds.	Generator(n=8,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)


# test.random_mult_traj()
# print
# print test.Raver

# n2.random_mult_traj()
# print
# print n2.Raver
# n4.random_mult_traj()
# print
# print n4.Raver
# n6.random_mult_traj()
# print
# print n6.Raver
n8.random_mult_traj()
print
print n8.Raver


# def generate_autocorrelation(seq):
#     length = len(seq)
#     correlation = np.zeros(length)
#     for i in range(length):
#         seq_shift = np.roll(seq, i)
#         seq_shift[:i] = np.zeros(i)
#         correlation[i] = np.dot(seq, seq_shift) / float(length - i)
#
#     return correlation

def generate_autocorrelation(seq):
    n = len(seq)
    if n % 2 == 0:
        length = n // 2
    else:
        length = n // 2 + 1
        seq = np.append(seq, 0.)

    correlation = np.zeros(length)
    for i in range(length):
        seq_shift = np.roll(seq, i)
        seq_shift[:i] = np.zeros(i)
        seq_shift[i+length:] = np.zeros(length-i)
        correlation[i] = np.dot(seq, seq_shift) / float(length)

    return correlation

def fourier_transform(seq, deltat):
    N = len(seq)
    # xf = np.linspace(0, 1/(2*deltat), N//2)
    xf = np.linspace(0, np.pi / deltat, N // 2)
    # The normalization for omega is 2pi/deltat,
    # because we use half of the overall points, so we h
    yf = fft(seq)

    # return xf, 2.0 / N * np.abs(yf[0: N // 2])
    # return xf, np.abs(yf[0: N // 2]) / np.abs(yf[0])
    return xf, yf[0: N // 2].real / yf[0].real

# def fourier_transform(seq, deltat):
#     N = len(seq)
#     w = np.fft.fftfreq(seq.size) * 2 * np.pi / deltat
#     yf = fft(seq)
#     yf *= deltat / (np.sqrt(2 * np.pi))
#
#     return w, yf

def fourier_generic(ft, deltat, omegaD, points):
    N = len(ft)
    tarray = np.linspace(0.0, deltat*N, N)
    omegaarray = np.linspace(0.0, 2*omegaD, points)
    fomega_re = np.zeros(points)
    fomega_im = np.zeros(points)
    for j, element in enumerate(omegaarray):
        cos_array = deltat * np.cos(element * tarray)
        sin_array = deltat * np.sin(element * tarray)

        fomega_re[j] = np.dot(ft, cos_array)
        fomega_im[j] = np.dot(ft, sin_array)
    fomega_abs = np.sqrt(fomega_re**2 + fomega_im**2)
    fomega_normal = fomega_abs/fomega_abs[0]
    return fomega_normal



# cor = np.zeros(test.t_num / 4)
# for j in range(Ntraj):
#     cor += generate_autocorrelation(test.R_traj[:, j])
# cor /= Ntraj

# cor = generate_autocorrelation(test.R_halflength)

# tf, yf = fourier_transform(cor, test.dt)

# cor_n2 = np.zeros(n2.sampling)
# cor_n2 = np.zeros(n2.sampling/2)
# for j in range(Ntraj):
#     cor_n2 += generate_autocorrelation(n2.R_traj[:, j])
# cor_n2 /= Ntraj

# cor_n4 = np.zeros(n4.sampling)
# cor_n4 = np.zeros(n4.sampling/2)
# for j in range(Ntraj):
#     cor_n4 += generate_autocorrelation(n4.R_traj[:, j])
# cor_n4 /= Ntraj
#
# cor_n6 = np.zeros(n6.sampling)
# cor_n6 = np.zeros(n6.sampling/2)
# for j in range(Ntraj):
#     cor_n6 += generate_autocorrelation(n6.R_traj[:, j])
# cor_n6 /= Ntraj

# cor_n8 = np.zeros(n8.sampling)
cor_n8 = np.zeros(n8.sampling/2)
for j in range(Ntraj):
    cor_n8 += generate_autocorrelation(n8.R_traj[:, j])
cor_n8 /= Ntraj


#----get the first half of the correlation
# cor_n2 = cor_n2[: n2.sampling//4]
# cor_n4 = cor_n4[: n4.sampling//2]
# cor_n6 = cor_n6[: n6.sampling//2]

# cor_n8 = cor_n8[: n8.sampling//4]
# np.savetxt('test.txt',cor_n8)


# tfn2 = np.zeros(t_num//(2*sampling))
# yfn2 = np.zeros(t_num//(2*sampling))
# tfn4 = np.zeros(t_num//(2*sampling))
# yfn4 = np.zeros(t_num//(2*sampling))
# tfn6 = np.zeros(t_num//(2*sampling))
# yfn6 = np.zeros(t_num//(2*sampling))
# tfn8 = np.zeros(t_num//(2*sampling))
# yfn8 = np.zeros(t_num//(2*sampling))
# for j in range(Ntraj):
#     tf_temp, yf_temp = fourier_transform(n2.R_traj[:, j], n2.dt)
#     tfn2 += tf_temp
#     yfn2 += yf_temp
# tfn2 /= Ntraj
# yfn2 /= Ntraj
#
# for j in range(Ntraj):
#     tf_temp, yf_temp = fourier_transform(n4.R_traj[:, j], n4.dt)
#     tfn4 += tf_temp
#     yfn4 += yf_temp
# tfn4 /= Ntraj
# yfn4 /= Ntraj
#
# for j in range(Ntraj):
#     tf_temp, yf_temp = fourier_transform(n6.R_traj[:, j], n6.dt)
#     tfn6 += tf_temp
#     yfn6 += yf_temp
# tfn6 /= Ntraj
# yfn6 /= Ntraj
#
# for j in range(Ntraj):
#     tf_temp, yf_temp = fourier_transform(n8.R_traj[:, j], n8.dt)
#     tfn8 += tf_temp
#     yfn8 += yf_temp
# tfn8 /= Ntraj
# yfn8 /= Ntraj


# tfn2, yfn2 = fourier_transform(cor_n2, n2.dt)
# tfn4, yfn4 = fourier_transform(cor_n4, n4.dt)
# tfn6, yfn6 = fourier_transform(cor_n6, n6.dt)
tfn8, yfn8 = fourier_transform(cor_n8, n8.dt)


# yfn2 = fourier_generic(cor_n2, n2.dt, n2.omegaD, points)
# yfn4 = fourier_generic(cor_n4, n4.dt, n4.omegaD, points)
# yfn6 = fourier_generic(cor_n6, n6.dt, n6.omegaD, points)
# yfn8 = fourier_generic(cor_n8, n8.dt, n8.omegaD, points)


# yfn2 = fft(cor_n2)
# yfn2 /= np.abs(yfn2[0])
# yfn4 = fft(cor_n4)
# yfn6 = fft(cor_n6)
# yfn8 = fft(cor_n8)
# yfn8 /= np.abs(yfn8[0])


# plt.figure()
# plt.plot(n2.t[: n2.sampling // 4], cor_n2, label='n2')
# plt.figure()
# plt.plot(n4.t[: n4.sampling // 2], cor_n4, label='n4')
# plt.figure()
# plt.plot(n6.t[: n6.sampling // 2], cor_n6, label='n6')
# plt.figure()
# plt.plot(n8.t[: n8.sampling // 4], cor_n8, label='n8')

# plt.figure()
# # plt.plot(n2.t[: n2.sampling], cor_n2, label='n2')
# plt.plot(cor_n2, label='n2')
# plt.figure()
# # plt.plot(n4.t[: n4.sampling], cor_n4, label='n4')
# plt.plot(cor_n4, label='n4')
# plt.figure()
# # plt.plot(n6.t[: n6.sampling], cor_n6, label='n6')
# plt.plot(cor_n6, label='n6')
plt.figure()
# plt.plot(n8.t[: n8.sampling], cor_n8, label='n8')
plt.plot(cor_n8, label='n8')
plt.legend()

plt.figure()
# plt.plot(omegagrid, yfn2, 'o-', label='n2')
# plt.plot(omegagrid, yfn4, 'o-', label='n4')
# plt.plot(omegagrid, yfn6, 'o-', label='n6')
# plt.plot(omegagrid, yfn8, 'o-', label='n8')
# plt.plot(yfn2[: 100], label='n2')
# plt.plot(yfn4, label='n4')
# plt.plot(yfn6, label='n6')
# plt.plot(yfn8[: 100], label='n8')
# plt.plot(tfn2, yfn2, 'o-', label='n2')
# plt.plot(tfn4, yfn4, 'o-', label='n4')
# plt.plot(tfn6, yfn6, 'o-', label='n6')
plt.plot(tfn8, yfn8, 'o-', label='n8')
plt.plot(tfn8, 1/(1+(tfn8/omegaD)**16), label='analytical:n=8')
# plt.figure()
# plt.plot(test.t[test.t_num / 4 :], test.R_halflength)
plt.legend()
plt.show()
