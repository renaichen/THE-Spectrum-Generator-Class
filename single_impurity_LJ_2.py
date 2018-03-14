import numpy as np
import matplotlib.pyplot as plt
import Debye_spectrum_3 as ds


omegaD = 1
t_num = 10000
dt1 = 0.1
Ntraj1 = 50
sampling = 2
#------------------------------------------------------
tBegin = 0
tEnd = 7000
dt2 = 0.1

m1 = 1.0
k12 = 0.1
x012 = 10.0
traj = 0
tstep = 0

Ntraj2 = 400
while traj<Ntraj2:

    n8 = ds.Generator(n=8,
                 omegaD=omegaD,
                 temperature=10,
                 dt=dt1,
                 t_num=t_num,
                 Ntraj=Ntraj1,
               sampling_rate=sampling)

    rand_array = n8.give_me_random_series(dt2)
    if traj == 0:
        points = len(rand_array)
        tArray = np.linspace(tBegin, tEnd, points)
        tsize = tArray.size
        Utraj = np.zeros(tsize)
        Ktraj = np.zeros(tsize)
        damper_traj = np.zeros(tsize)

    x1 = np.zeros(tsize)
    x2 = np.zeros(tsize)

    v1 = np.zeros(tsize)
    
    U = np.zeros(tsize)
    K = np.zeros(tsize)

    damper = np.zeros(tsize)

    x1[0] = 0.0
    x2[0] = 15

    v1[0] = 0

    f1new = k12*(x2[0]-x1[0]-x012)

    tstep = 0
    while tstep < (tsize-1):
#
        f1old = f1new
        damper[tstep] = n8.damp_getter(-f1old)
# #-----------EOM integrator: using the velocity verlet algorithm-----------------------
        x1[tstep+1] = x1[tstep] + v1[tstep]*dt2 + (0.5/m1)*f1old*dt2**2
        x2[tstep+1] = x2[0] + damper[tstep] + rand_array[tstep]
        # x2[tstep+1] = x2[0] + damper[tstep]# + rand_array[tstep]
        f1new = k12*(x2[tstep+1]-x1[tstep+1]-x012)
#
        v1[tstep+1] = v1[tstep] + 0.5*((f1old+f1new)/m1) * dt2
# #----------------------------------------------------------------------------------

        U[tstep] = 0.5*k12*(x2[tstep]-x1[tstep]-x012)**2

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

print n8._coe_rho

##--------------------plottings--------------------------------
plt.figure()
# plt.plot(n8.R_traj[:, 90])
plt.plot(n8.R_sampling)
# plt.plot(x1)
# plt.plot(rand_array)
plt.figure()
plt.plot(damper_traj)
plt.figure()
plt.plot(Utraj+Ktraj)
# plt.plot(cor_n8)
plt.show()
