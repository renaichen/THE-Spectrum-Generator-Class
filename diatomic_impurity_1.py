import numpy as np
import matplotlib.pyplot as plt
import Debye_spectrum_3 as ds
from scipy.fftpack import fft, ifft


#------------------------------------------------------
omegaD = 10
mass = 30.

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
    return 1/(m*omegaD*ss)*np.sum(2*ap*bp*sspn+(bp*bp-ap*ap)*scpn)


m1 = 1.0
m2 = 1.0
k12 = 0.2
x012 = 10.0
traj = 0
epsilon = 0.3
sigma = 3.5
A = 1e5
alpha = 1e-1

tBegin = 0.001
tEnd = 100
dt = 0.001

tArray = np.arange(tBegin, tEnd, dt)
tsize = len(tArray)
Utraj = np.zeros(tsize)
Ktraj = np.zeros(tsize)
damper_traj = np.zeros(tsize)

Ntraj2 = 1
while traj<Ntraj2:

    x1 = np.zeros(tsize)
    x2 = np.zeros(tsize)
    xL = np.zeros(tsize)
    xR = np.zeros(tsize)

    v1 = np.zeros(tsize)
    v2 = np.zeros(tsize)

    U = np.zeros(tsize)
    K = np.zeros(tsize)

    damperL = np.zeros(tsize)
    damperR = np.zeros(tsize)

    xL[0] = 10
    x1[0] = 13
    x2[0] = 25
    xR[0] = 30

    v1[0] = -1
    v2[0] = 1

    ssp10 = np.zeros(4)
    scp10 = np.zeros(4)
    ssp1 = np.zeros(4)
    scp1 = np.zeros(4)
    ssp20 = np.zeros(4)
    scp20 = np.zeros(4)
    ssp2 = np.zeros(4)
    scp2 = np.zeros(4)
    ap = np.zeros(4)
    bp = np.zeros(4)
    for i in range(4):
        ap[i] = np.cos((2 * i + 1) * np.pi / 16)
        bp[i] = np.sin((2 * i + 1) * np.pi / 16)
    ss = 1 / (2 * np.sin(3 * np.pi / 16))

    # f1new = k12*(x2[0]-x1[0]-x012)
    f1new = 0.0
    f2new = 0.0
    fL = 0.0
    fR = 0.0

    tstep = 0
    while tstep < (tsize-1):
#
        f1old = f1new
        f2old = f2new
        ssp10 = ssp1
        scp10 = scp1
        ssp20 = ssp2
        scp20 = scp2
        # damper[tstep] = n8.damp_getter(-f1old)
        # damper[tstep] = n8.damp_getter(-f2old)
        # ssp, scp, damper[tstep] = just_get_n8(ssp, scp, -f2old, dt)
        # damperL[tstep] = just_get_n8(ssp, scp, fL, dt)
        # damperR[tstep] = just_get_n8(ssp, scp, fR, dt)
        # ssp, scp, damper[tstep] = rk_4th_n8(ssp, scp, -f2old, dt)

        for i in range(4):
            ssp1[i] = ssp10[i] + (-bp[i] * omegaD * ssp10[i] + ap[i] * omegaD * scp10[i]) * dt
            scp1[i] = scp10[i] + (-bp[i] * omegaD * scp10[i] - ap[i] * omegaD * ssp10[i] + fL) * dt
            ssp2[i] = ssp20[i] + (-bp[i] * omegaD * ssp20[i] + ap[i] * omegaD * scp20[i]) * dt
            scp2[i] = scp20[i] + (-bp[i] * omegaD * scp20[i] - ap[i] * omegaD * ssp20[i] + fR) * dt
        damperL[tstep] = 1 / (mass * omegaD * ss) * np.sum(2 * ap * bp * ssp1 + (bp * bp - ap * ap) * scp1)
        damperR[tstep] = 1 / (mass * omegaD * ss) * np.sum(2 * ap * bp * ssp2 + (bp * bp - ap * ap) * scp2)

# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt + (0.5/m1)*f1old*dt**2
        x2[tstep+1] = x2[tstep] + v2[tstep]*dt + (0.5/m2)*f2old*dt**2
        xL[tstep+1] = xL[0] + damperL[tstep]
        xR[tstep+1] = xR[0] + damperR[tstep]
        f1new = k12*(x2[tstep+1]-x1[tstep+1]-x012)
        f2new = -k12*(x2[tstep+1]-x1[tstep+1]-x012)
        fL = -A*alpha*np.exp(-alpha*(x1[tstep+1]-xL[tstep+1]))
        fR = A*alpha*np.exp(-alpha*(xR[tstep+1]-x2[tstep+1]))
        # f2old = 48 * epsilon * sigma**12 / (x1[tstep+1]-x2[tstep+1])**13 \
        #             - 24 * epsilon * sigma**6/(x1[tstep+1]-x2[tstep+1])**7
        # f2old = 10/(x1[tstep+1]-x2[tstep+1])
        # f1new -= fL
        # f2new -= fR

        # print f1new, f2new
#
        v1[tstep+1] = v1[tstep] + 0.5*((f1old+f1new)/m1) * dt
        v2[tstep+1] = v2[tstep] + 0.5*((f2old+f2new)/m2) * dt
# #----------------------------------------------------------------------------------
        U[tstep] = 0.5*k12*(x2[tstep] - x1[tstep]-x012)**2
        # U[tstep] += A*np.exp(-alpha*(x1[tstep+1]-xL[tstep+1])) + A*np.exp(-alpha*(xR[tstep+1]-x2[tstep+1]))

        # U[tstep] += 4 * epsilon * sigma**12 / (x1[tstep]-x2[tstep])**12\
        #                 - 4 * epsilon * sigma**6/(x1[tstep]-x2[tstep])**6
        K[tstep] = 0.5*m1*v1[tstep]**2 + 0.5*m2*v2[tstep]**2

        tstep += 1
        print "time steps: ", tstep

    Utraj += U
    Ktraj += K
    # damper_traj += damper

    traj += 1
    print "MD traj #: ", traj

Utraj /= Ntraj2
Ktraj /= Ntraj2
damper_traj /= Ntraj2

##--------------------plottings--------------------------------
# plt.figure()
# # # plt.plot(n8.R_traj[:, 90])
# plt.plot(n8.R_sampling)
plt.figure()
plt.plot(x1)
plt.plot(x2)
# plt.plot(x2-x1)
plt.figure()
# plt.plot(np.log(Utraj+Ktraj))
plt.plot(Utraj+Ktraj)
# plt.plot(Ktraj)
# plt.plot(Utraj)
plt.show()
