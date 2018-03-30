import numpy as np
import matplotlib.pyplot as plt
import Debye_spectrum_3 as ds
import time
import random

#------------------------------------------------------

kB = 0.001987
gammaL = 1.0
gammaR = 1.0
Tl = 500.
Tr = 10.
tBegin = 0.0
tEnd = 500
dt = 0.01

tArray = np.arange(tBegin, tEnd, dt)
tsize = len(tArray)
Utraj = np.zeros(tsize)
Ktraj = np.zeros(tsize)
K1traj = np.zeros(tsize)
K2traj = np.zeros(tsize)
powerLtraj = np.zeros(tsize)
powerRtraj = np.zeros(tsize)
power12traj = np.zeros(tsize)

m1 = 12.0
m2 = 12.0
omega1 = 2.5
k12 = m1 * omega1**2
# x10 = 50
x012 = 1.5
traj = 0

start_time = time.time()
Ntraj2 = 120
while traj < Ntraj2:

    x1 = np.zeros(tsize)
    x2 = np.zeros(tsize)

    v1 = np.zeros(tsize)
    v2 = np.zeros(tsize)

    U = np.zeros(tsize)
    K = np.zeros(tsize)
    K1 = np.zeros(tsize)
    K2 = np.zeros(tsize)
    UintL = np.zeros(tsize)
    UintR = np.zeros(tsize)

    fint = np.zeros(tsize)
    powerL = np.zeros(tsize)
    powerR = np.zeros(tsize)
    power12 = np.zeros(tsize)

    x1[0] = 50.0
    x2[0] = 52.0

    v1[0] = 0.1
    v2[0] = 0.0

    f1new = 0.0
    f2new = 0.0

    tstep = 0
    while tstep < (tsize-1):

        xiL = random.gauss(0, 1)
        xiR = random.gauss(0, 1)
        f1old = f1new
        f2old = f2new

# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt + (0.5/m1)*f1old*dt**2
        x2[tstep+1] = x2[tstep] + v2[tstep]*dt + (0.5/m2)*f2old*dt**2
        f1new = k12*(x2[tstep+1]-x1[tstep+1]-x012)
        f2new = -f1new

        v1[tstep+1] = v1[tstep] + 0.5*((f1old+f1new)/m1) * dt - gammaL * v1[tstep] * dt \
                            + np.sqrt(2 * kB * Tl * gammaL / m1 ) * xiL * np.sqrt(dt)
        v2[tstep+1] = v2[tstep] + 0.5*((f2old+f2new)/m2) * dt - gammaR * v2[tstep] * dt \
                            + np.sqrt(2 * kB * Tr * gammaR / m2) * xiR * np.sqrt(dt)
# #----------------------------------------------------------------------------------
        U[tstep] = 0.5*k12*(x2[tstep]-x1[tstep]-x012)**2

        power12[tstep] = 0.5 * f1old * (v2[tstep] + v1[tstep])
        # if tstep > 0:
        #     powerL[tstep] = -gammaL * m1 * v1[tstep - 1] * 0.5 * (v1[tstep] + v1[tstep - 1]) \
        #                     + 0.5 * (v1[tstep] + v1[tstep - 1]) * np.sqrt(2 * kB * Tl * m1 * gammaL) * xiL / np.sqrt(dt)
        #     powerR[tstep] = -gammaR * m2 * v2[tstep - 1] * 0.5 * (v2[tstep] + v2[tstep - 1]) \
        #                     + 0.5 * (v2[tstep] + v2[tstep - 1]) * np.sqrt(2 * kB * Tr * m2 * gammaR) * xiR / np.sqrt(dt)

        powerL[tstep + 1] = -gammaL * m1 * v1[tstep] * 0.5 * (v1[tstep] + v1[tstep + 1]) \
                        + 0.5 * (v1[tstep] + v1[tstep + 1]) * np.sqrt(2 * kB * Tl * m1 * gammaL) * xiL / np.sqrt(dt)
        powerR[tstep + 1] = -gammaR * m2 * v2[tstep] * 0.5 * (v2[tstep] + v2[tstep + 1]) \
                        + 0.5 * (v2[tstep] + v2[tstep + 1]) * np.sqrt(2 * kB * Tr * m2 * gammaR) * xiR / np.sqrt(dt)


        K1[tstep] = 0.5 * m1 * v1[tstep] ** 2
        K2[tstep] = 0.5 * m2 * v2[tstep] ** 2
        K[tstep] = K1[tstep] + K2[tstep]

        tstep += 1
        print "time steps: ", tstep

    Utraj += U
    Ktraj += K
    K1traj += K1
    K2traj += K2
    powerLtraj += powerL
    powerRtraj += powerR
    power12traj += power12
    # damper_traj += damper

    traj += 1
    print "MD traj #: ", traj

Utraj /= Ntraj2
Ktraj /= Ntraj2
K1traj /= Ntraj2
K2traj /= Ntraj2
powerLtraj /= Ntraj2
powerRtraj /= Ntraj2
power12traj /= Ntraj2
Et = Utraj[1:-1] + Ktraj[1:-1]  # log(0) is bad and should be avoided

##----------
run_time = time.time() - start_time
print 'run time is: ', run_time / 60

NN = tsize / 2
# Ksteady = Ktraj[3000:]
# Kaver = np.sum(Ksteady)/len(Ksteady)
# print Kaver / kB
K1steady = K1traj[NN:]
K1aver = np.sum(K1steady)/len(K1steady)
print K1aver * 2 / kB
K2steady = K2traj[NN:]
K2aver = np.sum(K2steady)/len(K2steady)
print K2aver * 2 / kB
PsteadyL = powerLtraj[NN:]
PaverL = np.sum(PsteadyL)/len(PsteadyL)
print PaverL, np.std(PsteadyL), np.std(PsteadyL) / np.sqrt(NN)
Psteady12 = power12traj[NN:]
Paver12 = np.sum(Psteady12)/len(Psteady12)
print Paver12
PsteadyR = powerRtraj[NN:]
PaverR = np.sum(PsteadyR)/len(PsteadyR)
print PaverR


##--------------------plottings--------------------------------
# plt.figure()
# plt.plot(rand_arrayL)
# plt.plot(damperR)
# plt.plot(rand_arrayR)
plt.figure()
plt.plot(x1)
plt.plot(x2)
# plt.plot(v1)
# plt.plot(v2)
# # plt.plot(timeplot, np.log(Et))
# # plt.plot(timeplot, fitplot)
plt.figure()
# plt.plot(x1-xL)
# plt.plot(x2-xR)
plt.plot(Et)
# plt.plot(K1)
# plt.plot(K2)
# plt.plot(Ktraj)
# plt.plot(Utraj)
plt.figure()
plt.plot(power12traj)
plt.plot(powerLtraj)
plt.plot(powerRtraj)
# plt.plot(power12traj)
# plt.plot(powerLtraj-powerRtraj)
# plt.figure()
# plt.plot(term1)
# plt.plot(term2)

plt.show()


