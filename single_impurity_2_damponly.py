import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------
omegaD = 1.
mass = 32.

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


m1 = 12.0
omega1 = 0.4
k12 = m1 * omega1**2
x10 = 50
x012 = 2.0
traj = 0
epsilon = 0.3
sigma = 3.5
A = 1e5
alpha = 5e-1

tBegin = 0.001
tEnd = 500
dt = 0.001

tArray = np.arange(tBegin, tEnd, dt)
tsize = len(tArray)
Utraj = np.zeros(tsize)
Ktraj = np.zeros(tsize)
damper_traj = np.zeros(tsize)

Ntraj2 = 1
while traj < Ntraj2:

    x1 = np.zeros(tsize)
    xR = np.zeros(tsize)

    v1 = np.zeros(tsize)

    U = np.zeros(tsize)
    K = np.zeros(tsize)

    damperR = np.zeros(tsize)

    x1[0] = 57.0
    xR[0] = 71.0

    v1[0] = 2.

    ssp10 = np.zeros(4)
    scp10 = np.zeros(4)
    ssp1 = np.zeros(4)
    scp1 = np.zeros(4)

    ap = np.zeros(4)
    bp = np.zeros(4)
    for i in range(4):
        ap[i] = np.cos((2 * i + 1) * np.pi / 16)
        bp[i] = np.sin((2 * i + 1) * np.pi / 16)
    ss = 1 / (2 * np.sin(3 * np.pi / 16))

    f1new = 0.0
    fR = 0.0

    tstep = 0
    while tstep < (tsize-1):
#
        f1old = f1new
        ssp10 = ssp1
        scp10 = scp1
        # damperR[tstep] = just_get_n8(ssp, scp, fR, dt)
        # ssp, scp, damper[tstep] = rk_4th_n8(ssp, scp, -f2old, dt)

        for i in range(4):
            ssp1[i] = ssp10[i] + (-bp[i] * omegaD * ssp10[i] + ap[i] * omegaD * scp10[i]) * dt
            scp1[i] = scp10[i] + (-bp[i] * omegaD * scp10[i] - ap[i] * omegaD * ssp10[i] + fR) * dt
        damperR[tstep] = 1 / (mass * omegaD * ss) * np.sum(2 * ap * bp * ssp1 + (bp * bp - ap * ap) * scp1)

# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt + (0.5/m1)*f1old*dt**2
        xR[tstep+1] = xR[0] + damperR[tstep]
        f1new = -k12*(x1[tstep+1]-x10-x012)
        fR = A*alpha*np.exp(-alpha*(xR[tstep+1]-x1[tstep+1]))
        # f2old = 48 * epsilon * sigma**12 / (x1[tstep+1]-x2[tstep+1])**13 \
        #             - 24 * epsilon * sigma**6/(x1[tstep+1]-x2[tstep+1])**7
        # f2old = 10/(x1[tstep+1]-x2[tstep+1])
        f1new -= fR

        # print f1new, f2new
#
        v1[tstep+1] = v1[tstep] + 0.5*((f1old+f1new)/m1) * dt
# #----------------------------------------------------------------------------------
        U[tstep] = 0.5*k12*(x1[tstep]-x10-x012)**2
        U[tstep] += A*np.exp(-alpha*(xR[tstep]-x1[tstep]))

        # U[tstep] += 4 * epsilon * sigma**12 / (x1[tstep]-x2[tstep])**12\
        #                 - 4 * epsilon * sigma**6/(x1[tstep]-x2[tstep])**6
        K[tstep] = 0.5*m1*v1[tstep]**2

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
Et = Utraj[:-1] + Ktraj[:-1]
timeplot = tArray[:-1]
num_slope = 100000
slope, b = np.polyfit(tArray[:num_slope], np.log(Utraj[:num_slope]+Ktraj[:num_slope]), 1)
# slope, b = np.polyfit(timeplot, np.log(Et), 1)
print slope
fitplot = slope * timeplot + b
##--------------------plottings--------------------------------
# plt.figure()
# plt.plot(n8.R_sampling)
plt.figure()
plt.plot(x1)
plt.figure()
plt.plot(timeplot, np.log(Et))
plt.plot(timeplot, fitplot)
# plt.plot(Et)
# plt.plot(Ktraj)
# plt.plot(Utraj)
plt.show()
