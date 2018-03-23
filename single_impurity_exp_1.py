import numpy as np
import matplotlib.pyplot as plt
import Debye_spectrum_3 as ds
from scipy.fftpack import fft, ifft


omegaD = 1
t_num = 500000
dt1 = 0.001
Ntraj1 = 1
sampling = 2
temperature = 0.0
#------------------------------------------------------
tBegin = 0.001
tEnd = 250
dt2 = 0.001

m1 = 1.0
k12 = 0.5
x012 = 10
traj = 0
tstep = 0
epsilon = 0.3
sigma = 3.5

mass = 1.0

A = 1e3
alpha = 1e-1

def rk_4th_n8(sspo, scpo, f, dt):
    ap = np.zeros(4)
    bp = np.zeros(4)
    sspn = np.zeros(4)
    scpn = np.zeros(4)
    dssp1 = np.zeros(4)
    dssp2 = np.zeros(4)
    dssp3 = np.zeros(4)
    dssp4 = np.zeros(4)
    dscp1 = np.zeros(4)
    dscp2 = np.zeros(4)
    dscp3 = np.zeros(4)
    dscp4 = np.zeros(4)
    m = 1.
    for i in range(4):
        ap[i] = np.cos((2*i+1)*np.pi/16)
        bp[i] = np.sin((2*i+1)*np.pi/16)
        dssp1[i] = (-bp[i] * omegaD * sspo[i] + ap[i] * omegaD * scpo[i]) * dt
        dscp1[i] = (-bp[i] * omegaD * scpo[i] - ap[i] * omegaD * sspo[i] + f) * dt
        dssp2[i] = (-bp[i] * omegaD * (sspo[i] + 0.5 * dssp1[i]) +
                        ap[i] * omegaD * (scpo[i] + 0.5 * dscp1[i])) * dt
        dscp2[i] = (-bp[i] * omegaD * (scpo[i] + 0.5 * dscp1[i]) -
                        ap[i] * omegaD * (sspo[i] + 0.5 * dssp1[i]) + f) * dt
        dssp3[i] = (-bp[i] * omegaD * (sspo[i] + 0.5 * dssp2[i]) +
                    ap[i] * omegaD * (scpo[i] + 0.5 * dscp2[i])) * dt
        dscp3[i] = (-bp[i] * omegaD * (scpo[i] + 0.5 * dscp2[i]) -
                    ap[i] * omegaD * (sspo[i] + 0.5 * dssp2[i]) + f) * dt
        dssp4[i] = (-bp[i] * omegaD * (sspo[i] + dssp3[i]) +
                    ap[i] * omegaD * (scpo[i] + dscp3[i])) * dt
        dscp4[i] = (-bp[i] * omegaD * (scpo[i] + dscp3[i]) -
                    ap[i] * omegaD * (sspo[i] + dssp3[i]) + f) * dt
        sspn[i] = sspo[i] + 1 / 6.0 * (dssp1[i] + 2 * dssp2[i] + 2 * dssp3[i] + dssp4[i])
        scpn[i] = scpo[i] + 1 / 6.0 * (dscp1[i] + 2 * dscp2[i] + 2 * dscp3[i] + dscp4[i])
    ss = 1/(2*np.sin(3*np.pi/16))
    return sspn, scpn, 1/(m*omegaD*ss)*np.sum(2*ap*bp*sspn+(bp*bp-ap*ap)*scpn)

def just_get_n8(sspo, scpo, f, dt):
    ap = np.zeros(4)
    bp = np.zeros(4)
    sspn = np.zeros(4)
    scpn = np.zeros(4)
    m = 1.
    for i in range(4):
        ap[i] = np.cos((2*i+1)*np.pi/16)
        bp[i] = np.sin((2*i+1)*np.pi/16)
        sspn[i] = sspo[i] + (-bp[i]*omegaD*sspo[i]+ap[i]*omegaD*scpo[i])*dt
        scpn[i] = scpo[i] + (-bp[i] * omegaD * scpo[i] - ap[i] * omegaD * sspo[i] + f) * dt
    ss = 1/(2*np.sin(3*np.pi/16))
    return sspn, scpn, 1/(m*omegaD*ss)*np.sum(2*ap*bp*sspn+(bp*bp-ap*ap)*scpn)

def Ft(n):
    time = np.linspace(0.0, 100.0, 1000)
    f = np.zeros(len(time))
    ap = np.zeros(n/2)
    bp = np.zeros(n/2)
    for i in range(n/2):
        ap[i] = np.cos((2*i+1)*np.pi/(2*n))
        bp[i] = np.sin((2*i+1)*np.pi/(2*n))
        f += np.exp(-bp[i]*time)*(2*ap[i]*bp[i]*np.sin(ap[i]*time)+(bp[i]**2-ap[i]**2)*np.cos(ap[i]*time))
    ss = 1 / (2 * np.sin(3 * np.pi / (2*n)))
    return f*time/ss

# yy = Ft(4)
# plt.plot(yy)


#-----------fourier transform of the analytic spectrum-------------
#
# omega_array = np.linspace(0.0, 4 * omegaD, t_num)
# func = omega_array**2 / (1 + (omega_array/omegaD)**12)
# time = np.linspace(0.1, 1/omegaD, t_num/2)
# # func = 1 / (1 + (omega_array/omegaD)**16)
# ft_func = ifft(func)
# trans = np.exp(-ft_func[:t_num/2]*time)
# trans = -ft_func[:t_num/2]*time
#
# plt.figure()
# plt.plot(func)
# plt.figure()
# # plt.plot(time, np.log(trans))
# plt.plot(time, trans)
#-----------------------------------------------------------------

n8 = ds.Generator(n=4,
                      mass=1,
                 omegaD=omegaD,
                 temperature=temperature,
                 dt=dt1,
                 t_num=t_num,
                 Ntraj=Ntraj1,
               sampling_rate=sampling)

rand_array = n8.give_me_random_series(dt2)

points = len(rand_array)
tArray = np.linspace(tBegin, tEnd, points)
tsize = tArray.size
Utraj = np.zeros(tsize)
Ktraj = np.zeros(tsize)
damper_traj = np.zeros(tsize)

al = 0.0005
D =1

Ntraj2 = 1
while traj<Ntraj2:

    n8 = ds.Generator(n=4,
                      mass=1,
                 omegaD=omegaD,
                 temperature=temperature,
                 dt=dt1,
                 t_num=t_num,
                 Ntraj=Ntraj1,
               sampling_rate=sampling)

    rand_array = n8.give_me_random_series(dt2)

    x1 = np.zeros(tsize)
    x2 = np.zeros(tsize)

    v1 = np.zeros(tsize)
    
    U = np.zeros(tsize)
    K = np.zeros(tsize)

    damper = np.zeros(tsize)

    # x1[0] = 0.0
    x1[0] = 50
    x2[0] = 55

    v1[0] = 5

    ssp0 = np.zeros(4)
    scp0 = np.zeros(4)
    ssp = np.zeros(4)
    scp = np.zeros(4)
    ap = np.zeros(4)
    bp = np.zeros(4)
    for i in range(4):
        ap[i] = np.cos((2 * i + 1) * np.pi / 16)
        bp[i] = np.sin((2 * i + 1) * np.pi / 16)
    ss = 1 / (2 * np.sin(3 * np.pi / 16))


    # f1new = k12*(x2[0]-x1[0]-x012)
    f1new = 0.0
    f2old = 0.0

    tstep = 0
    while tstep < (tsize-1):
#
        f1old = f1new
        # damper[tstep] = n8.damp_getter(-f1old)
        # damper[tstep] = n8.damp_getter(-f2old)
        ssp0 = ssp
        scp0 = scp
        # ssp, scp, damper[tstep] = just_get_n8(ssp0, scp0, -f2old, dt2)
        # ssp, scp, damper[tstep] = rk_4th_n8(ssp0, scp0, -f2old, dt2)

        for i in range(4):
            ssp[i] = ssp0[i] + (-bp[i] * omegaD * ssp0[i] + ap[i] * omegaD * scp0[i]) * dt2
            scp[i] = scp0[i] + (-bp[i] * omegaD * scp0[i] - ap[i] * omegaD * ssp0[i] - f2old) * dt2
        damper[tstep] = 1/(mass*omegaD*ss)*np.sum(2*ap*bp*ssp+(bp*bp-ap*ap)*scp)

# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt2 + (0.5/m1)*f1old*dt2**2
        x2[tstep+1] = x2[0] + damper[tstep] #+ rand_array[tstep]
        # x2[tstep+1] = x2[0]*np.sin(omegaD*tArray[tstep])/(omegaD*tArray[tstep]) + damper[tstep]# + rand_array[tstep]
        # f1new = k12*(x2[tstep+1]-x1[tstep+1]-x012)
        f1new = -k12*(x1[tstep+1]-x012)
        # f1new = -2*al*D*(np.exp(-al*(x1[tstep+1]-x012))-np.exp(2*al*(x1[tstep+1]-x012)))
        f2old = -A*alpha*np.exp(-alpha*(x2[tstep+1]-x1[tstep+1])) #+ np.exp(-0.5*(x1[tstep+1]-0.0))
        # f2old = 48 * epsilon * sigma**12 / (x1[tstep+1]-x2[tstep+1])**13 \
        #             - 24 * epsilon * sigma**6/(x1[tstep+1]-x2[tstep+1])**7
        # f2old = 10/(x1[tstep+1]-x2[tstep+1])

        print f1new, f2old
        f1new += f2old
#
        v1[tstep+1] = v1[tstep] + 0.5*((f1old+f1new)/m1) * dt2
# #----------------------------------------------------------------------------------

        # U[tstep] = 0.5*k12*(x2[tstep]-x1[tstep]-x012)**2
        U[tstep] = 0.5*k12*(x1[tstep]-x012)**2
        U[tstep] += A*np.exp(-alpha*(x2[tstep]-x1[tstep]))
        # U[tstep] = D*(1-np.exp(-al*(x1[tstep]-x012)))**2
        # U[tstep] += 4 * epsilon * sigma**12 / (x1[tstep]-x2[tstep])**12\
        #                 - 4 * epsilon * sigma**6/(x1[tstep]-x2[tstep])**6
        K[tstep] = 0.5*m1*v1[tstep]**2

        tstep += 1
        print "time steps: ", tstep

    Utraj += U
    Ktraj += K
    damper_traj += damper

    traj += 1
    print "MD traj #: ", traj

Utraj /= Ntraj2
Ktraj /= Ntraj2
damper_traj /= Ntraj2
m, b = np.polyfit(tArray[:10000], np.log(Utraj[:10000]+Ktraj[:10000]), 1)
print m
fitplot = m * tArray + b
# Lambdat = np.delete(Utraj+Ktraj, -1)
# # Lambdat = Lambdat[len(Lambdat)/2:]
# tnofirst = np.delete(tArray, -1)
# # tnofirst = tnofirst[len(tnofirst)/2:]
# Lambdat = np.log(Lambdat)
# Lambda = -Lambdat/tnofirst
# Lambda = fft(Lambda)
# xaxis, Lambda = ds.fourier_transform(Lambda, dt2)

# print Utraj + Ktraj

##--------------------plottings--------------------------------
# plt.figure()
# # # plt.plot(n8.R_traj[:, 90])
# plt.plot(n8.R_sampling)
plt.figure()
plt.plot(x1)
plt.plot(x2)
# plt.plot(x2-x1)
# plt.plot(rand_array)
# plt.figure()
# plt.plot(damper_traj)
# plt.figure()
# plt.plot(Lambda)
plt.figure()
plt.plot(tArray[:-1], np.log(Utraj[:-1]+Ktraj[:-1]))
plt.plot(tArray[:-1], fitplot[:-1])
# plt.plot(Utraj+Ktraj)
# plt.plot(Ktraj)
# plt.plot(Utraj)
plt.show()
