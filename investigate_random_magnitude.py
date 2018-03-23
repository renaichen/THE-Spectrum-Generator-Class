import numpy as np
import matplotlib.pyplot as plt
import Debye_spectrum_3 as ds


#------------------------------------------------------
n = 8
omegaD = 1.
mass = 32.
t_num = 1000000
dt1 = 0.001
Ntraj1 = 1
temperature = 1

tBegin = 0.0
tEnd = 500
dt = 0.001

n8 = ds.Generator(n=n,
                 mass=mass,
                 omegaD=omegaD,
                 temperature=temperature,
                 dt=dt1,
                 t_num=t_num,
                 Ntraj=Ntraj1,
                 )

rand_array = n8.give_me_random_series(dt)
print n8._coe_rho

points = len(rand_array)
tArray = np.linspace(tBegin, tEnd, points)
tsize = len(tArray)
Utraj = np.zeros(tsize)
Ktraj = np.zeros(tsize)
damper_traj = np.zeros(tsize)

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

Ntraj2 = 0
while traj < Ntraj2:
    n8 = ds.Generator(n=n,
                      mass=mass,
                      omegaD=omegaD,
                      temperature=temperature,
                      dt=dt1,
                      t_num=t_num,
                      Ntraj=Ntraj1,
                      )
    rand_array = n8.give_me_random_series(dt)

    x1 = np.zeros(tsize)
    xR = np.zeros(tsize)

    v1 = np.zeros(tsize)

    U = np.zeros(tsize)
    K = np.zeros(tsize)

    damperR = np.zeros(tsize)

    x1[0] = 57.0
    xR[0] = 71.0

    v1[0] = 2.

    f1new = 0.0
    fR = 0.0

    tstep = 0
    while tstep < (tsize-1):
#
        f1old = f1new
        damperR[tstep] = n8.damp_getter(fR)

# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt + (0.5/m1)*f1old*dt**2
        xR[tstep+1] = xR[0] + damperR[tstep] + rand_array[tstep]
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

# Utraj /= Ntraj2
# Ktraj /= Ntraj2
# damper_traj /= Ntraj2
# Et = Utraj[1:-1] + Ktraj[1:-1]  # log(0) is bad and should be avoided
# timeplot = tArray[1:-1]

##--------------------plottings--------------------------------
plt.figure()
plt.plot(n8.R_sampling)
# plt.figure()
# plt.plot(x1)
# plt.figure()
# plt.plot(timeplot, np.log(Et))
# plt.plot(timeplot, fitplot)
# plt.plot(Et)
# plt.plot(Ktraj)
# plt.plot(Utraj)
plt.show()
