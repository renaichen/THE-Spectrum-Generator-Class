import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt
from scipy.fftpack import fft


class Generator(object):
    """
    Class which implements the generation of the random part
    and damping part (with memory kernel) for displacement of
    anchor bath particle y_l according to the paper: J. Chem. Phys. 69, 2525 (1978)
    """
    def __init__(self, n, omegaD,
                 mass=1, temperature=1, dt=0.001, t_num=1000, Ntraj=1, sampling_rate=10):

        # set the parameters and constants
        self.n = n
        self.omegaD = omegaD
        self.kB = 0.001987   #kcal/(mol K)
        # self.kB = 0.008314  # kJ/(mol K)
        self.mass = mass
        self.temperature = temperature
        self.dt = dt
        self.Ntraj = Ntraj

        # Initialize the coefficients
        self._ap = np.zeros(self.n)
        self._bp = np.zeros(self.n)
        self._sigma = 1/(2*np.sin((3*np.pi)/(2.0*self.n)))
        self._coe_rho = (self.kB * self.temperature / self.mass) * (
                            2 ** (2 * self.n - 1) * np.sin(
                                 3 * np.pi / (2 * self.n))) / self.omegaD ** 3  # for the coefficient of random number

        # set dynamical variables
        self.t_num = t_num
        self.sampling = t_num // sampling_rate
        self.t = np.linspace(0, self.dt * self.t_num, self.t_num)
        self.R_length = np.zeros(self.t_num)
        self.R_sampling = np.zeros(self.sampling)


    def compute_ap(self):
        for p in range(self.n):
            self._ap[p] = np.cos((2 * p + 1) * np.pi / (2.0 * self.n))

    def compute_bp(self):
        for p in range(self.n):
            self._bp[p] = np.sin((2 * p + 1) * np.pi / (2.0 * self.n))

    def compute_coe_rho(self):
        for i in range(1, self.n):
            self._coe_rho *= np.sin(i * np.pi / (2 * self.n)) ** 2

    def get_coefficients(self):
        self.compute_ap()
        self.compute_bp()
        self.compute_coe_rho()
        # return self._coe_rho

    def random_evolve_euler(self, x, v):
        tstep = 0
        while tstep < (self.t_num-1):
            # x[tstep, self.n / 2] = self._coe_rho * random.gauss(0, 1)
            x[tstep, self.n / 2] = np.sqrt(self._coe_rho) * random.gauss(0, 1)

            # One has to be very careful, in using this x[..., n/2] elements, they are the original Gaussian random,
            # so the following x[..., i] should be treated differently than x[..., i-1], although they appear the same

            for i in range(self.n / 2, 0, -1):
                x[tstep + 1, i - 1] = x[tstep, i - 1] + v[tstep, i - 1] * self.dt
                # v[tstep + 1, i - 1] = v[tstep, i - 1] + self.omegaD ** 2 * \
                #                         (x[tstep, i] - x[tstep, i - 1] -
                #                             2 * self._bp[i - 1] / self.omegaD * v[tstep, i - 1]) * self.dt
                v[tstep + 1, i - 1] = v[tstep, i - 1] + self.omegaD ** 2 * \
                                                        (x[tstep, i]/np.sqrt(self.dt) - x[tstep, i - 1] -
                                                         2 * self._bp[i - 1] / self.omegaD * v[tstep, i - 1]) * self.dt
            tstep += 1

    def random_evolve_vv(self, x, v):
        tstep = 0
        a = np.zeros((self.t_num, self.n / 2))
        while tstep < (self.t_num-1):
            # x[tstep, self.n / 2] = self._coe_rho * random.gauss(0, 1)
            x[tstep, self.n / 2] = np.sqrt(self._coe_rho) * random.gauss(0, 1)

            # One has to be very careful, in using this x[..., n/2] elements, they are the original Gaussian random,
            # so the following x[..., i] should be treated differently than x[..., i-1], although they appear the same

            for i in range (self.n/2, 0, -1):
                # a[tstep, i - 1] =  self.omegaD ** 2 * \
                #                         (x[tstep, i] - x[tstep, i - 1] -
                #                             2 * self._bp[i - 1] / self.omegaD * v[tstep, i - 1])
                a[tstep, i - 1] = - self.omegaD ** 2 * x[tstep, i - 1]
                x[tstep + 1, i - 1] = x[tstep, i - 1] + v[tstep, i - 1] * self.dt + 0.5 * a[tstep, i - 1] * self.dt ** 2
                a[tstep + 1, i - 1] = - self.omegaD ** 2 * x[tstep + 1, i - 1]
                # v[tstep + 1, i - 1] = v[tstep, i - 1] + 0.5 * (a[tstep, i - 1] + a[tstep + 1, i - 1]) * self.dt \
                #                         - 2 * self._bp[i - 1] * self.omegaD * v[tstep, i - 1] * self.dt\
                #                         + self.omegaD ** 2 * x[tstep, i] * self.dt
                v[tstep + 1, i - 1] = v[tstep, i - 1] + 0.5 * (a[tstep, i - 1] + a[tstep + 1, i - 1]) * self.dt \
                                      - 2 * self._bp[i - 1] * self.omegaD * v[tstep, i - 1] * self.dt \
                                      + self.omegaD ** 2 * x[tstep, i] * np.sqrt(self.dt)
            tstep += 1


    def random_mult_traj(self):
        self.compute_bp()
        self.compute_coe_rho()
        sampling = self.sampling
        self.R_traj = np.zeros((sampling, self.Ntraj))
        if self.n % 2 == 0:     # it is even
            traj = 0
            while traj<self.Ntraj:
                """
                zz is used to represent the Z matrix that is decomposed from the set of 2nd-DEs, 
                dzdt is just the derivative elements of that
                """
                zz = np.zeros((self.t_num, self.n / 2 + 1))
                dzdt = np.zeros((self.t_num, self.n / 2 + 1))
                zz[0, :] = np.ones(self.n / 2 + 1)

                self.random_evolve_euler(zz, dzdt)
                # self.random_evolve_vv(zz, dzdt)

                self.R_traj[:, traj] = zz[self.t_num - sampling :, 0]
                # self.R_traj[:, traj] = zz[self.t_num / 2:, self.n/2]
                self.R_length[:] += zz[:, 0]
                # self.R_length[:] += zz[:, self.n/2]
                print traj
                traj += 1

                # print zz[2500, 0], zz[250, self.n/2]

            self.R_sampling[:] = self.R_length[self.t_num - sampling :]
            self.Raver = np.sum(self.R_sampling)/(sampling)


        else:    # it is odd
            pass



omegaD = 2
t_num = 10000
dt = 0.02
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

n2 = Generator(n=2,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)

n4 = Generator(n=4,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)

n6 = Generator(n=6,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)

n8 = Generator(n=8,
                 omegaD=omegaD,
                 temperature=1,
                 dt=dt,
                 t_num=t_num,
                 Ntraj=Ntraj,
               sampling_rate=sampling)


n2.random_mult_traj()
print
print n2.Raver
n4.random_mult_traj()
print
print n4.Raver
n6.random_mult_traj()
print
print n6.Raver
n8.random_mult_traj()
print
print n8.Raver


def generate_autocorrelation(seq):
    length = len(seq)
    correlation = np.zeros(length)
    for i in range(length):
        seq_shift = np.roll(seq, i)
        seq_shift[:i] = np.zeros(i)
        correlation[i] = np.dot(seq, seq_shift) / float(length - i)

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


cor_n2 = np.zeros(n2.sampling)
for j in range(Ntraj):
    cor_n2 += generate_autocorrelation(n2.R_traj[:, j])
cor_n2 /= Ntraj
#
cor_n4 = np.zeros(n4.sampling)
for j in range(Ntraj):
    cor_n4 += generate_autocorrelation(n4.R_traj[:, j])
cor_n4 /= Ntraj
#
cor_n6 = np.zeros(n6.sampling)
for j in range(Ntraj):
    cor_n6 += generate_autocorrelation(n6.R_traj[:, j])
cor_n6 /= Ntraj
#
cor_n8 = np.zeros(n8.sampling)
for j in range(Ntraj):
    cor_n8 += generate_autocorrelation(n8.R_traj[:, j])
cor_n8 /= Ntraj


#----get the first half of the correlation
cor_n2 = cor_n2[: n2.sampling//2]
cor_n4 = cor_n4[: n4.sampling//2]
cor_n6 = cor_n6[: n6.sampling//2]
cor_n8 = cor_n8[: n8.sampling//2]


tfn2, yfn2 = fourier_transform(cor_n2, n2.dt)
tfn4, yfn4 = fourier_transform(cor_n4, n4.dt)
tfn6, yfn6 = fourier_transform(cor_n6, n6.dt)
tfn8, yfn8 = fourier_transform(cor_n8, n8.dt)


plt.figure()
plt.plot(n2.t[: n2.sampling // 2], cor_n2, label='n2')
plt.figure()
plt.plot(n4.t[: n4.sampling // 2], cor_n4, label='n4')
plt.figure()
plt.plot(n6.t[: n6.sampling // 2], cor_n6, label='n6')
plt.figure()
plt.plot(n8.t[: n8.sampling // 2], cor_n8, label='n8')

# plt.figure()
# plt.plot(n2.t[: n2.sampling], cor_n2, label='n2')
# plt.figure()
# plt.plot(n4.t[: n4.sampling], cor_n4, label='n4')
# plt.figure()
# plt.plot(n6.t[: n6.sampling], cor_n6, label='n6')
# plt.figure()
# plt.plot(n8.t[: n8.sampling], cor_n8, label='n8')

plt.figure()
plt.plot(tfn2, yfn2, 'o-', label='n2')
plt.plot(tfn4, yfn4, 'o-', label='n4')
plt.plot(tfn6, yfn6, 'o-', label='n6')
plt.plot(tfn8, yfn8, 'o-', label='n8')
plt.plot(tfn8, 1/(1+(tfn8/omegaD)**16), label='analytical:n=8')
# plt.figure()
# plt.plot(test.t[test.t_num / 4 :], test.R_halflength)
plt.legend()
plt.show()
