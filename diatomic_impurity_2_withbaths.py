import numpy as np
import matplotlib.pyplot as plt
import Debye_spectrum_3 as ds
import time

#------------------------------------------------------
nL = 8
omegaDL = 5.
massL = 32.
t_numL = 300000
dt1L = 0.01
Ntraj1L = 1
temperatureL = 500.0

nR = 8
omegaDR = 5.
massR = 32.
t_numR = 300000
dt1R = 0.01
Ntraj1R = 1
temperatureR = 10.0

tBegin = 0.0
tEnd = 1500
dt = 0.01
kB = 0.00198
sp_objL = ds.Generator(n=nL,
                 mass=massL,
                 omegaD=omegaDL,
                 temperature=temperatureL,
                 dt=dt1L,
                 t_num=t_numL,
                 Ntraj=Ntraj1L,
                 )

rand_arrayL = sp_objL.give_me_random_series(dt)

sp_objR = ds.Generator(n=nR,
                 mass=massR,
                 omegaD=omegaDR,
                 temperature=temperatureR,
                 dt=dt1R,
                 t_num=t_numR,
                 Ntraj=Ntraj1R,
                 )

rand_arrayR = sp_objR.give_me_random_series(dt)
# print n8._coe_rho

points = len(rand_arrayL)
tArray = np.linspace(tBegin, tEnd, points)
tsize = len(tArray)
Utraj = np.zeros(tsize)
Ktraj = np.zeros(tsize)
K1traj = np.zeros(tsize)
K2traj = np.zeros(tsize)
powerLtraj = np.zeros(tsize)
powerRtraj = np.zeros(tsize)
power12traj = np.zeros(tsize)
powerLsqtraj = np.zeros(tsize)
powerRsqtraj = np.zeros(tsize)
power12sqtraj = np.zeros(tsize)
damper_trajL = np.zeros(tsize)
damper_trajR = np.zeros(tsize)

m1 = 12.0
m2 = 12.0
mu = m1 * m2 / (m1 + m2)
omega1 = 2.5
k12 = mu * omega1**2
# x10 = 50
x012 = 1.5
traj = 0
epsilon = 0.3
sigma = 3.5
AL = 23 * 1e5  #  1eV = 23kcal/mol
alphaL = 5e-0
AR = 23 * 1e5  #  1eV = 23kcal/mol
alphaR = 5e-0

start_time = time.time()
Ntraj2 = 300
while traj < Ntraj2:
    sp_objL = ds.Generator(n=nL,
                           mass=massL,
                           omegaD=omegaDL,
                           temperature=temperatureL,
                           dt=dt1L,
                           t_num=t_numL,
                           Ntraj=Ntraj1L,
                           )
    rand_arrayL = sp_objL.give_me_random_series(dt)

    sp_objR = ds.Generator(n=nR,
                           mass=massR,
                           omegaD=omegaDR,
                           temperature=temperatureR,
                           dt=dt1R,
                           t_num=t_numR,
                           Ntraj=Ntraj1R,
                           )
    rand_arrayR = sp_objR.give_me_random_series(dt)

    x1 = np.zeros(tsize)
    x2 = np.zeros(tsize)
    xL = np.zeros(tsize)
    xR = np.zeros(tsize)

    v1 = np.zeros(tsize)
    v2 = np.zeros(tsize)

    U = np.zeros(tsize)
    K = np.zeros(tsize)
    K1 = np.zeros(tsize)
    K2 = np.zeros(tsize)
    UintL = np.zeros(tsize)
    UintR = np.zeros(tsize)

    fint = np.zeros(tsize)
    fLt = np.zeros(tsize)
    fRt = np.zeros(tsize)
    powerL = np.zeros(tsize)
    powerR = np.zeros(tsize)
    power12 = np.zeros(tsize)
    powerLsq = np.zeros(tsize)
    powerRsq = np.zeros(tsize)
    power12sq = np.zeros(tsize)
    term1 = np.zeros(tsize)
    term2 = np.zeros(tsize)

    damperL = np.zeros(tsize)
    damperR = np.zeros(tsize)

    xL[0] = 47.5
    x1[0] = 50.0
    x2[0] = 52.0
    xR[0] = 54.5

    v1[0] = 0.
    v2[0] = 0.0

    f1new = 0.0
    f2new = 0.0
    fL = 0.0
    fR = 0.0

    tstep = 0
    while tstep < (tsize-1):
#
        f1old = f1new
        f2old = f2new
        damperL[tstep] = sp_objL.damp_getter(fL)
        damperR[tstep] = sp_objR.damp_getter(fR)

# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt + (0.5/m1)*f1old*dt**2
        x2[tstep+1] = x2[tstep] + v2[tstep]*dt + (0.5/m2)*f2old*dt**2
        xL[tstep+1] = xL[0] + damperL[tstep] + rand_arrayL[tstep]
        xR[tstep+1] = xR[0] + damperR[tstep] + rand_arrayR[tstep]
        f1new = k12*(x2[tstep+1]-x1[tstep+1]-x012)
        f2new = -f1new
        fint[tstep + 1] = f1new

        fL = - AL * alphaL * np.exp(-alphaL * (x1[tstep + 1] - xL[tstep + 1]))
        fR = AR * alphaR * np.exp(-alphaR * (xR[tstep + 1] - x2[tstep + 1]))
        fLt[tstep] = fL
        fRt[tstep] = fR

        # f2old = 48 * epsilon * sigma**12 / (x1[tstep+1]-x2[tstep+1])**13 \
        #             - 24 * epsilon * sigma**6/(x1[tstep+1]-x2[tstep+1])**7
        # f2old = 10/(x1[tstep+1]-x2[tstep+1])
        f1new -= fL
        f2new -= fR
#
        v1[tstep+1] = v1[tstep] + 0.5*((f1old+f1new)/m1) * dt
        v2[tstep+1] = v2[tstep] + 0.5*((f2old+f2new)/m2) * dt
# #----------------------------------------------------------------------------------
        U[tstep] = 0.5*k12*(x2[tstep]-x1[tstep]-x012)**2
        UintL[tstep] = AL * np.exp(-alphaL * (x1[tstep] - xL[tstep]))
        UintR[tstep] = AR * np.exp(-alphaR * (xR[tstep] - x2[tstep]))
        U[tstep] += UintL[tstep]
        U[tstep] += UintR[tstep]

        # powerL[tstep + 1] = fint[tstep + 1] * v1[tstep + 1] + (UintL[tstep + 1] - UintL[tstep]) / dt
        # powerR[tstep + 1] = -fint[tstep + 1] * v2[tstep + 1] + (UintR[tstep + 1] - UintR[tstep]) / dt
        # # powerL[tstep] = f1old * v1[tstep] + 0.5 * (UintL[tstep] - UintL[tstep - 1])/dt
        # # powerR[tstep] = f1old * v1[tstep] + 0.5 * (UintR[tstep] - UintR[tstep - 1])/dt
        # power12[tstep + 1] = 0.5 * fint[tstep + 1] * (v2[tstep + 1] - v1[tstep + 1])
        # term1[tstep + 1] = fint[tstep + 1] * v1[tstep + 1]
        # term2[tstep + 1] = (UintL[tstep + 1] - UintL[tstep]) / dt

        if tstep > 0:
            powerL[tstep] = 0.5 * fL * ((xL[tstep + 1] - xL[tstep]) / dt + v1[tstep])
            powerR[tstep] = 0.5 * fR * ((xR[tstep + 1] - xR[tstep]) / dt + v2[tstep])
            power12[tstep] = 0.5 * fint[tstep] * (v2[tstep] + v1[tstep])
            powerLsq[tstep] = powerL[tstep] * powerL[tstep]
            powerRsq[tstep] = powerR[tstep] * powerR[tstep]
            power12sq[tstep] = power12[tstep] * power12[tstep]
            term1[tstep] = fint[tstep] * v1[tstep]

        # U[tstep] += 4 * epsilon * sigma**12 / (x1[tstep]-x2[tstep])**12\
        #                 - 4 * epsilon * sigma**6/(x1[tstep]-x2[tstep])**6
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
    powerLsqtraj += powerLsq
    powerRsqtraj += powerRsq
    power12sqtraj += power12sq
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
powerLsqtraj /= Ntraj2
powerRsqtraj /= Ntraj2
power12sqtraj /= Ntraj2
# damper_traj /= Ntraj2
Et = Utraj[1:-1] + Ktraj[1:-1]  # log(0) is bad and should be avoided
timeplot = tArray[1:-1]
# num_slope = 10000
# slope, b = np.polyfit(tArray[1:num_slope], np.log(Utraj[1:num_slope]+Ktraj[1:num_slope]), 1)
# print slope
# fitplot = slope * timeplot + b

##----------
run_time = time.time() - start_time
print 'run time is: ', run_time / 60.
NN = 50000
# Ksteady = Ktraj[30000:]
# Kaver = np.sum(Ksteady)/len(Ksteady)
# print Kaver
K1steady = K1traj[NN:]
T1aver = np.mean(K1steady) * 2 / kB
T1std = np.std(K1steady) * 2 / kB
print 'T1 = ', T1aver, T1std
K2steady = K2traj[NN:]
T2aver = np.mean(K2steady) * 2 / kB
T2std = np.std(K2steady) * 2 / kB
print 'T2 = ', T2aver, T2std

PsteadyL = powerLtraj[NN:]
PsqsteadyL = powerLsqtraj[NN:]
JLaver = np.mean(PsteadyL)
JLstd = np.std(PsteadyL)
JLstd_true = np.sqrt(np.mean(PsqsteadyL) - JLaver**2)
print 'heatL = ', JLaver, JLstd, JLstd_true
Psteady12 = power12traj[NN:]
Psqsteady12 = power12sqtraj[NN:]
J12aver = np.mean(Psteady12)
J12std = np.std(Psteady12)
J12std_true = np.sqrt(np.mean(Psqsteady12) - J12aver**2)
print 'heat12 = ', J12aver, J12std, J12std_true
PsteadyR = powerRtraj[NN:]
PsqsteadyR = powerRsqtraj[NN:]
JRaver = np.mean(PsteadyR)
JRstd = np.std(PsteadyR)
JRstd_true = np.sqrt(np.mean(PsqsteadyR) - JRaver**2)
print 'heatR = ', JRaver, JRstd, JRstd_true


##-----------write-data-out---------
filename = time.strftime('diatomic-%m-%d-%H%M.txt')
with open(filename, "w") as f:
    f.write("trajectory number: %d\n" %(Ntraj2))
    f.write("time_step: %f\n" %(dt))
    f.write("number of steps: %d\n" %(t_numL/2))
    f.write("TL = %d, TR = %d\n" %(temperatureL, temperatureR))
    # f.write("T1 = %f\n" %(T1aver))
    # f.write("T2 = %f\n" %(T2aver))
    f.write("JL = %f, STDJL = %f\n" %(JLaver, JLstd_true))
    f.write("J12 = %f, STDJ12 = %f\n" %(J12aver, J12std_true))
    f.write("JR = %f, STDJR = %f\n" %(JRaver, JRstd_true))


##--------------------plottings--------------------------------
# plt.figure()
# plt.plot(rand_arrayL)
# plt.plot(damperR)
# plt.plot(rand_arrayR)
# plt.figure()
# plt.plot(x1)
# plt.plot(x2)
# plt.plot(xL)
# plt.plot(xR)
# plt.figure()
# plt.plot(x1-xL)
# plt.plot(x2-xR)
# plt.plot(v1)
# plt.plot(v2)
# # plt.plot(timeplot, np.log(Et))
# # plt.plot(timeplot, fitplot)
# plt.figure()
# plt.plot(K1traj)
# plt.plot(K2traj)
plt.figure()
plt.plot(Et)
# plt.plot(Ktraj)
# plt.plot(Utraj)
plt.figure()
plt.plot(power12traj)
# plt.plot(powerLtraj)
# plt.plot(powerRtraj)
# plt.plot(powerLtraj-powerRtraj)
# plt.figure()
# plt.plot(term1)
# plt.plot(term2)
# plt.figure()
# plt.plot(fLt)
# plt.plot(fRt)
plt.show()


