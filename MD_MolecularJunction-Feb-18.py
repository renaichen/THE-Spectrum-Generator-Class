# Solving Equations of Motion - Alkane Molecular Junction
#======Importing=============
import os
import sys
import time
import numpy as np
import math
import random
import matplotlib.pyplot as plt
pathname = os.path.dirname(sys.argv[0])
fullpath = os.path.abspath(pathname)
#=============================================================================================================
start_time = time.time()
#################time parameters###########################
tBegin=0
tEnd=3 #1 = 48.88882 fs  #simulation time
dt=	0.001 #0.01 = 0.488882 fs #timestep size
sqrtdt = np.sqrt(dt)
t = np.arange(tBegin, tEnd, dt)
tsize = t.size 
numoftraj = 20 #number of traj
#################units###########################
mH = 1.008 #[g/mol]
mC = 12.011 #[g/mol]
mAu = 196.96657 #[g/mol]
mN = 14.007 #[g/mol]
mS = 32.059 #[g/mol]
kb = 0.001987191 #[kcal /mol K]
Tunit = 300. #[K]
pi = 3.14159265359 
coulombconst = 332.0716 #[kcal A / mol e^2]
dielectric = 1.# vacuum approx 80 for water
################thermal parameters################
Kpin =1.0*500. #[kcal/mol A^2] force constant of terminal carbons --- pinning potential
#gammaL=0.488882*0.25  # = 0.0488882 =  1 ps^-1
#gammaR= 0.488882*0.75 # = 0.0488882 = 1 ps^-1
gammaL = 4.88882 #4.88882  = 0.1 fs^-1 ;  0.488882 = 10 ps^-1
gammaR = 4.88882 #4.88882  = 0.1 fs^-1 ;  0.488882 = 10 ps^-1
#TL = 0.8*Tunit #[K]
#TR = 16./15.*Tunit #[K]
TL = 300.#[K] Temperature of left electrode
TR = 600.#[K] Temperature of right electrode
Teff = (gammaL*TL+gammaR*TR)/(gammaL+gammaR) #[K] weighted average of the two baths
#Teff = 1000. #for equilbrium NVT dynamics 
gamma = 0.488882 #for equilbrium NVT dynamics 
print "Teff", Teff
equiltime = tEnd/2.#dt #tsize/2
dynamics = 'NVE'
#dynamics = 'Noneq' #'NVT','Noneq','NVE'
cutoffradius = 15 #[A]
NeighborListRefreshRate = tsize #tsize means only make neighbor list at first timestep
TrajOutputRate = 10 #10 if 0.001


#import molecular geometry
molfile = "/c2h2-python.pdb"

# how many atoms?
N = 0
for line in open(os.path.abspath(pathname)+molfile): 
	listN = line.split()
	id = listN[0]
	if id == 'HETATM':
		N+=1
		
# how many carbons?
numofcarbons = 0
for line in open(os.path.abspath(pathname)+molfile): 
	lists = line.split()
	id = lists[0]
	#print id
	if id == 'HETATM':	
		if lists[2] == 'C':
			numofcarbons+=1
		
print 'number of carbons', numofcarbons

############array declarations################	
#############################################	
xold = np.zeros(N); yold = np.zeros(N); zold = np.zeros(N)
Lterminaleq = np.zeros(3); Rterminaleq = np.zeros(3)
atomtype = list()
mass = np.zeros(N)
Ax  = np.zeros(N); Ay  = np.zeros(N); Az  = np.zeros(N)
bonding = np.zeros((N,N-1),dtype=int)
TarrayN = np.zeros(numofcarbons)
Tarraytotal = np.zeros(tsize)
Energyarraytotal = np.zeros(tsize+1)
Energyarray = np.zeros(tsize+1)
chix = np.zeros(N); chiy = np.zeros(N); chiz = np.zeros(N)
etax = np.zeros(N); etay = np.zeros(N); etaz = np.zeros(N)
Qtotal = np.zeros(tsize+1)
QL = np.zeros(tsize+1)
QR = np.zeros(tsize+1)
DeltaEarray = np.zeros(tsize+1)
Qtotalcum = np.zeros(tsize+1)
DeltaEarraycum = np.zeros(tsize+1)
neighbors = np.zeros((N,N),dtype=int)
neighbors.fill(-1)

#############make list of atomtypes##################
for line in open(os.path.abspath(pathname)+molfile): 
	#print line	
	lists = line.split()
	#print "length", len(list)
	id = lists[0]
	#print id
	if id == 'HETATM':		
		atomtype.append(lists[2])
		if lists[2]=='H':
			mass[float(lists[1])-1]=mH
		elif lists[2]=='C':
			mass[float(lists[1])-1]=mC
		elif lists[2]=='Au':
			mass[float(lists[1])-1]=mAu
		elif lists[2]=='N':
			mass[float(lists[1])-1]=mN
		elif lists[2]=='S':
			mass[float(lists[1])-1]=mS
		else:
			print "No match for atomtype"
		
		xold[float(lists[1])-1] = float(lists[5])
		yold[float(lists[1])-1] = float(lists[6])
		zold[float(lists[1])-1] = float(lists[7])

	elif id == 'CONECT':
		lists = line.split()
		j = 0
		while j<len(lists)-2:
			bonding[int(lists[1])-1][j] = int(lists[j+2])			
			j+=1

#print bonding
#############make list of bonds##################
bonds = []
j = 0
while j<N:
	i = 0
	while i<N-1:		
		if bonding[j,i] != 0:
			bond = []
			bond.append(bonding[j,i])
			bond.append(j+1)									
			writetobondlist = 1
			if len(bonds)>0:
				l = 0
				while l<len(bonds):												
					if bond[0] == bonds[l][1] and bond[1] == bonds[l][0]:
						writetobondlist = 0
					l+=1
				if writetobondlist == 1:
					bonds.append(bond)
			else:
				bonds.append(bond)				
		i+=1
		#print bonds
	j+=1	
print 'bonds', bonds


#############make list of angles##################
angles = []
j = 0
while j<N:
	i = 0
	while i<N-1:		
		if bonding[j,i] != 0:
			angle = []
			anglehold = []
			angle.append(bonding[j,i])
			angle.append(j+1)
			anglehold = list(angle)				
			k = 0
			while k<N-1:
				angle = list(anglehold)
				#is particle connected to jth particle
				if bonding[j,k]!=anglehold[0] and bonding[j,k]!=anglehold[1] and bonding[j,k]!=0:					
					angle.append(bonding[j,k])
					writetoeanglelist = 1
					if len(angles)>0:
						l = 0
						while l<len(angles):												
							if angle[0] == angles[l][2] and angle[2] == angles[l][0]:
								writetoeanglelist = 0
							l+=1
						if writetoeanglelist == 1:
							angles.append(angle)
					else:
						angles.append(angle)					
				k+=1	
		i+=1
		#print angle
	j+=1	

print 'angles', angles

#############make list of dihedrals##################
#dihedrals = list(angles)
dihedrals = []
kk = 0
while kk<len(angles): #go through every 3 connection and see if there is a four connection
	dihedral = angles[kk]
	dihedralhold = list(dihedral)
	#is particle on left end of angle connected to another particle?
	j = dihedral[0]-1
	k = 0
	while k<N-1:
		dihedral = list(dihedralhold)
		if bonding[j,k]!=dihedralhold[0] and bonding[j,k]!=dihedralhold[1] and bonding[j,k]!=dihedralhold[2] and bonding[j,k]!=0:
			dihedral.insert(0,bonding[j,k])
			writetoeanglelist = 1
			if len(dihedrals)>0:
				l = 0
				while l<len(dihedrals):												
					if dihedral[0] == dihedrals[l][3] and dihedral[3] == dihedrals[l][0]:
						writetoeanglelist = 0
					l+=1
				if writetoeanglelist == 1:
					dihedrals.append(dihedral)
			else:
				dihedrals.append(dihedral)
		k+=1
	#is particle on right end of angle connected to another particle?
	j = dihedral[2]-1
	k = 0
	while k<N-1:
		dihedral = list(dihedralhold)
		if bonding[j,k]!=dihedralhold[0] and bonding[j,k]!=dihedralhold[1] and bonding[j,k]!=dihedralhold[2] and bonding[j,k]!=0:
			dihedral.append(bonding[j,k])
			writetoeanglelist = 1
			if len(dihedrals)>0:
				l = 0
				while l<len(dihedrals):												
					if dihedral[0] == dihedrals[l][3] and dihedral[3] == dihedrals[l][0]:
						writetoeanglelist = 0
					l+=1
				if writetoeanglelist == 1:
					dihedrals.append(dihedral)
			else:
				dihedrals.append(dihedral)
		k+=1
	kk+=1	
print 'dihedrals', dihedrals

#############make list of 1-4 pairs##################
onefourpairs = []
kk = 0
while kk<len(dihedrals):
	onefourpair = []
	onefourpair.append(dihedrals[kk][0])
	onefourpair.append(dihedrals[kk][3])
	writetoeanglelist = 1
	if len(onefourpairs)>0:
		l = 0
		while l<len(onefourpairs):												
			if onefourpair[0] == onefourpairs[l][0] and onefourpair[1] == onefourpairs[l][1] or onefourpair[0] == onefourpairs[l][1] and onefourpair[1] == onefourpairs[l][0]:
				writetoeanglelist = 0
			l+=1
		if writetoeanglelist == 1:
			onefourpairs.append(onefourpair)
	else:
		onefourpairs.append(onefourpair)
	kk+=1	
print 'onefourpairs', onefourpairs

######################atomtypes#########################
print 'atomtype', atomtype

###########finding terminal carbons for thermostatting##########
Lterm = 0
Lterminaleq[0] = xold[Lterm]
Lterminaleq[1] = yold[Lterm]
Lterminaleq[2] = zold[Lterm]

#findlastcarbon

if N>1:
	i = N-1
	while i>=0:
		if atomtype[i] == 'C':		
			Rterm = i
			Rterminaleq[0] = xold[i]
			Rterminaleq[1] = yold[i]
			Rterminaleq[2] = zold[i]
			i = 0
		i-=1
else:
	Rterm = 0
	Rterminaleq[0] = xold[0]
	Rterminaleq[1] = yold[0]
	Rterminaleq[2] = zold[0]


	
#print atomtype[Lterm]
#print atomtype[Rterm]
#print Lterm, Rterm


#####################################################################
#change indicies of the bonds, angles, dihedrals, and onefourpairs
bonds = [[(xx-1) for xx in row] for row in bonds]
#print bonds
angles = [[(xx-1) for xx in row] for row in angles]
#print angles
dihedrals = [[(xx-1) for xx in row] for row in dihedrals]
#print dihedrals
onefourpairs = [[(xx-1) for xx in row] for row in onefourpairs]
#print onefourpairs
######################################################################
######################################################################
#########################Neighborlist	
def NeighborList(xarg,yarg,zarg):
	neighbors.fill(-1)
	i = 0
	while i<N:
		j = i+1
		neighborcount=0
		while j<N:			
			#are i and j bonded?
			bonded = 0 #assume they are not
			l = 0
			while l<len(bonds):
				if i == bonds[l][0] and j == bonds[l][1] or i == bonds[l][1] and j == bonds[l][0]:
					bonded = 1
					#print bonded
					l = len(bonds) # breaks loop
				l+=1		
			if bonded ==0:
				#print 'bonds',i, j, bond[0], bond[1]
				X = xold[i]-xold[j]
				Y = yold[i]-yold[j]
				Z = zold[i]-zold[j]
				r = np.sqrt(X**2.+Y**2.+Z**2.)	
				#print r, cutoffradius
				if r<cutoffradius:	
					neighbors[i,neighborcount]=j
					neighborcount+=1
			j+=1
		i+=1	

#####################################################################
#####################Energy Function##################################
def Energy(xarg,yarg,zarg,vxarg,vyarg,vzarg):
	E = 0
	x = list(xarg)
	y = list(yarg)
	z = list(zarg)
	vx = list(vxarg)
	vy = list(vyarg)
	vz = list(vzarg)
	#############################################################################
	########################nonbonded forces#####################################
	i = 0
	while i<N:
		if atomtype[i] == 'H':
			sigmai = 2.5
			epsiloni = 0.030
			qi = 0.06
		elif atomtype[i] == 'C':
			carbonbondcounter = 0
			l = 0
			while l<len(bonding[i]):
				if atomtype[bonding[i][l]-1] == 'C':
					carbonbondcounter +=1
				l+=1
			if carbonbondcounter==0: #CH4
				sigmai = 3.5
				epsiloni = 0.066
				qi = -0.24
			elif carbonbondcounter==1: #CH3R
				sigmai = 3.5
				epsiloni = 0.066
				qi = -0.18
			elif carbonbondcounter==2: #CH2R2
				sigmai = 3.5
				epsiloni = 0.066
				qi = -0.12			
			elif carbonbondcounter==3: #CHR3
				sigmai = 3.5
				epsiloni = 0.066
				qi = -0.06
		j = i+1
		while j<N:
			#are i and j bonded?
			bonded = 0 #assume they are not
			l = 0
			while l<len(bonds):
				if i == bonds[l][0] and j == bonds[l][1] or i == bonds[l][1] and j == bonds[l][0]:
					bonded = 1
					l = len(bonds) # breaks loop
				l+=1
			if bonded == 0:
				#print i,j,atomtype[i],atomtype[j]
				#is it H to H
				if atomtype[j] == 'H':
					sigmaj = 2.5
					epsilonj = 0.030			
					qj = 0.06
				elif atomtype[j] == 'C':
					carbonbondcounter = 0
					l = 0
					while l<len(bonding[j]):
						if atomtype[bonding[j][l]-1] == 'C':
							carbonbondcounter +=1
						l+=1
					if carbonbondcounter==0: #CH4 or C
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.24
					elif carbonbondcounter==1: #CH3R
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.18
					elif carbonbondcounter==2: #CH2R2
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.12			
					elif carbonbondcounter==3: #CHR3
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.06	
				sigma = np.sqrt(sigmai*sigmaj)
				epsilon = np.sqrt(epsiloni*epsilonj)			
				#is it a 1-4 pair
				itisaonefourpair = 0		
				l = 0
				while l<len(onefourpairs):
					if i == onefourpairs[l][0] and j == onefourpairs[l][1] or i == onefourpairs[l][1] and j == onefourpairs[l][0]:
						itisaonefourpair = 1
						l = len(onefourpairs) # breaks loop
					l+=1
					
				if itisaonefourpair==1:
					fij = 0.5		
				else:
					fij = 1.0
				X = x[i]-x[j]
				Y = y[i]-y[j]
				Z = z[i]-z[j]
				r = np.sqrt(X**2.+Y**2.+Z**2.)	
				E+= fij*(coulombconst*(qi*qj)/(dielectric*r)+4.*epsilon*((sigma/r)**12.-(sigma/r)**6.))
				j+=1
			else:
				j+=1
		i+=1
	
	##############bonds
	i = 0
	while i<len(bonds):	
		bond = bonds[i]	
		if atomtype[bond[0]]=='H' and atomtype[bond[1]]=='C' or atomtype[bond[0]]=='C' and atomtype[bond[1]]=='H':
			#print "H-C"		
			req = 1.090
			K  = 340.
			#K = 0.04
		else:
			#print "C-C"
			req = 1.529
			K = 268.	
								
		X = x[bond[0]]-x[bond[1]]
		Y = y[bond[0]]-y[bond[1]]
		Z = z[bond[0]]-z[bond[1]]
		r = np.sqrt(X**2.+Y**2.+Z**2.)
		E+= K*(r-req)**2.		
		#print 'energybonds', E/energyunit
		i+=1
		
	##############pinning potential#################
	X = x[Lterm]-Lterminaleq[0]
	Y = y[Lterm]-Lterminaleq[1]
	Z = z[Lterm]-Lterminaleq[2]
	r = np.sqrt(X**2.+Y**2.+Z**2.)
	E+= Kpin*(r)**2.	

	#if numofcarbons > 1:
	X = x[Rterm]-Rterminaleq[0]
	Y = y[Rterm]-Rterminaleq[1]
	Z = z[Rterm]-Rterminaleq[2]
	r = np.sqrt(X**2.+Y**2.+Z**2.)
	E+= Kpin*(r)**2.	
	#print 'energybonds', E/energyunit
			
	
	#################angles ##CHARRM22 parameters
	i = 0
	while i<len(angles):	
		angle = angles[i]
		if atomtype[angle[0]]=='C' and atomtype[angle[1]]=='C' and atomtype[angle[2]]=='C':	
			#print "C-C-C"
			thetaeq = 112.7/57.2957795
			K = 58.35		
		elif atomtype[angle[0]]=='H' and atomtype[angle[1]]=='C' and atomtype[angle[2]]=='H':
			#print "H-C-H"
			thetaeq = 107.8/57.2957795
			K = 33.00	
		else:
			#print "C-C-H"
			thetaeq = 110.7/57.2957795
			K = 37.50	
		
		v1 = np.zeros(3)
		v2 = np.zeros(3)
		v1[0] = x[angle[1]]-x[angle[0]]
		v1[1] = y[angle[1]]-y[angle[0]]
		v1[2] = z[angle[1]]-z[angle[0]]
		
		v2[0] = x[angle[2]]-x[angle[1]]
		v2[1] = y[angle[2]]-y[angle[1]]
		v2[2] = z[angle[2]]-z[angle[1]]
		
		magv1 = np.sqrt(v1[0]**2.+v1[1]**2.+v1[2]**2.)
		magv2 = np.sqrt(v2[0]**2.+v2[1]**2.+v2[2]**2.)
		
		theta = np.arccos(-(v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])/(magv1*magv2))		
		E+= K*(theta-thetaeq)**2.
		i+=1
		
	#################dihedrals
	i = 0
	while i<len(dihedrals):	
		dihedral = dihedrals[i]
		if atomtype[dihedral[0]]=='C' and atomtype[dihedral[1]]=='C' and atomtype[dihedral[2]]=='C' and atomtype[dihedral[3]]=='C':
			#print "C-C-C-C"
			V1 = 1.74
			V2 = -0.157
			V3 = 0.279
			f1 = 0
			f2 = 0
			f3 = 0
		elif atomtype[dihedral[0]]=='H' and atomtype[dihedral[1]]=='C' and atomtype[dihedral[2]]=='C' and atomtype[dihedral[3]]=='H':
			#print "H-C-C-H"
			V1 = 0.0
			V2 = 0.0
			V3 = 0.318
			f1 = 0
			f2 = 0
			f3 = 0
		else:
			#print "H-C-C-C"	
			V1 = 0.0
			V2 = 0.0
			V3 = 0.366
			f1 = 0
			f2 = 0
			f3 = 0
			
		v1 = np.zeros(3)#p2-p1
		v2 = np.zeros(3)#p3-p2
		v3 = np.zeros(3)#p4-p3
		v1[0] = x[dihedral[1]]-x[dihedral[0]]
		v1[1] = y[dihedral[1]]-y[dihedral[0]]
		v1[2] = z[dihedral[1]]-z[dihedral[0]]
		
		v2[0] = x[dihedral[2]]-x[dihedral[1]]	
		v2[1] = y[dihedral[2]]-y[dihedral[1]]
		v2[2] = z[dihedral[2]]-z[dihedral[1]]
	
		v3[0] = x[dihedral[3]]-x[dihedral[2]]
		v3[1] = y[dihedral[3]]-y[dihedral[2]]
		v3[2] = z[dihedral[3]]-z[dihedral[2]]
		
		v1xv2 = np.zeros(3)
		v1xv2[0] = (v1[1]*v2[2]-v1[2]*v2[1])
		v1xv2[1] = (v1[2]*v2[0]-v1[0]*v2[2])
		v1xv2[2] = (v1[0]*v2[1]-v1[1]*v2[0])
		
		v2xv3 = np.zeros(3)
		v2xv3[0] = (v2[1]*v3[2]-v2[2]*v3[1])
		v2xv3[1] = (v2[2]*v3[0]-v2[0]*v3[2])
		v2xv3[2] = (v2[0]*v3[1]-v2[1]*v3[0])
		
		magv2 = np.sqrt(v2[0]**2.+v2[1]**2.+v2[2]**2.)
		
		magv1xv2 = np.sqrt(v1xv2[0]**2.+v1xv2[1]**2.+v1xv2[2]**2.)
		magv2xv3 = np.sqrt(v2xv3[0]**2.+v2xv3[1]**2.+v2xv3[2]**2.)
		
		v1dotv2 = v2[0]*v1[0]+v2[1]*v1[1]+v2[2]*v1[2]
		v2dotv3 = v2[0]*v3[0]+v2[1]*v3[1]+v2[2]*v3[2]
   
		arg1 = magv2*(v1[0]*v2xv3[0]+v1[1]*v2xv3[1]+v1[2]*v2xv3[2])
		arg2 = v1xv2[0]*v2xv3[0]+v1xv2[1]*v2xv3[1]+v1xv2[2]*v2xv3[2]
		theta = np.arctan2(arg1,arg2)
		#print theta*57.2957795
		E+= 0.5*V1*(1.+np.cos(theta+f1))+0.5*V2*(1.-np.cos(2.*theta+f2))+0.5*V3*(1.+np.cos(3.*theta+f3))
		i+=1		
		
		
		
		
	############add kinetic energy#########################################	
	i = 0
	while i<N:			
		E+= 0.5*mass[i]*(vx[i]**2.+ vy[i]**2.+ vz[i]**2.)
		i+=1	
	
	return E


#####################Force Function##################################
######################################################################
######################################################################
def Forces(xarg,yarg,zarg,fxarg,fyarg,fzarg):
	x = list(xarg)
	y = list(yarg)
	z = list(zarg)
	#print fxarg
	fx = np.zeros(N)
	fy = np.zeros(N)
	fz = np.zeros(N)
	#########################################################################
	########################nonbonded forces#################################
	i = 0
	while i<N:
		J = 0
		while J<N:
			j = neighbors[i][J]
			if j==-1:
				J=N#end of list
			else:
###############################particle i parameters#############################################################				
				if atomtype[i] == 'H':
					sigmai = 2.5
					epsiloni = 0.030
					qi = 0.06
				elif atomtype[i] == 'C':
					carbonbondcounter = 0
					l = 0
					while l<len(bonding[i]):
						if atomtype[bonding[i][l]-1] == 'C':
							carbonbondcounter +=1
						l+=1
					if carbonbondcounter==0: #CH4
						sigmai = 3.5
						epsiloni = 0.066
						qi = -0.24
					elif carbonbondcounter==1: #CH3R
						sigmai = 3.5
						epsiloni = 0.066
						qi = -0.18
					elif carbonbondcounter==2: #CH2R2
						sigmai = 3.5
						epsiloni = 0.066
						qi = -0.12			
					elif carbonbondcounter==3: #CHR3
						sigmai = 3.5
						epsiloni = 0.066
						qi = -0.06

###############################particle j parameters#############################################################						
				if atomtype[j] == 'H':
					sigmaj = 2.5
					epsilonj = 0.030			
					qj = 0.06
				elif atomtype[j] == 'C':
					carbonbondcounter = 0
					l = 0
					while l<len(bonding[j]):
						if atomtype[bonding[j][l]-1] == 'C':
							carbonbondcounter +=1
						l+=1
					if carbonbondcounter==0: #CH4 or C
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.24
					elif carbonbondcounter==1: #CH3R
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.18
					elif carbonbondcounter==2: #CH2R2
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.12			
					elif carbonbondcounter==3: #CHR3
						sigmaj = 3.5
						epsilonj = 0.066
						qj = -0.06	
				sigma = np.sqrt(sigmai*sigmaj)
				epsilon = np.sqrt(epsiloni*epsilonj)			
				#is it a 1-4 pair
				itisaonefourpair = 0		
				l = 0
				while l<len(onefourpairs):
					if i == onefourpairs[l][0] and j == onefourpairs[l][1] or i == onefourpairs[l][1] and j == onefourpairs[l][0]:
						itisaonefourpair = 1
						l = len(onefourpairs) # breaks loop
					l+=1
					
				if itisaonefourpair==1:
					fij = 0.5		
				else:
					fij = 1.0
				X = x[i]-x[j]
				Y = y[i]-y[j]
				Z = z[i]-z[j]
				r = np.sqrt(X**2.+Y**2.+Z**2.)	
				#if atomtype[i] == 'C' and atomtype[j] == 'C':
				#	print r
				fx[i] += fij*(coulombconst*(qi*qj)/(dielectric*r**2.)+48.*epsilon*sigma**12./r**13.-24.*epsilon*sigma**6./r**7.)*X/r
				fy[i] += fij*(coulombconst*(qi*qj)/(dielectric*r**2.)+48.*epsilon*sigma**12./r**13.-24.*epsilon*sigma**6./r**7.)*Y/r
				fz[i] += fij*(coulombconst*(qi*qj)/(dielectric*r**2.)+48.*epsilon*sigma**12./r**13.-24.*epsilon*sigma**6./r**7.)*Z/r
				fx[j] -= fij*(coulombconst*(qi*qj)/(dielectric*r**2.)+48.*epsilon*sigma**12./r**13.-24.*epsilon*sigma**6./r**7.)*X/r      
				fy[j] -= fij*(coulombconst*(qi*qj)/(dielectric*r**2.)+48.*epsilon*sigma**12./r**13.-24.*epsilon*sigma**6./r**7.)*Y/r
				fz[j] -= fij*(coulombconst*(qi*qj)/(dielectric*r**2.)+48.*epsilon*sigma**12./r**13.-24.*epsilon*sigma**6./r**7.)*Z/r	
				J+=1
		i+=1
	
	##############bonds
	i = 0
	while i<len(bonds):	
		bond = bonds[i]	
		if atomtype[bond[0]]=='H' and atomtype[bond[1]]=='C' or atomtype[bond[0]]=='C' and atomtype[bond[1]]=='H':
			#print "H-C"		
			req = 1.090
			K  = 340.
			#K = 0.04
		else:
			#print "C-C"
			req = 1.529
			K = 268.	
								
		X = x[bond[0]]-x[bond[1]]
		Y = y[bond[0]]-y[bond[1]]
		Z = z[bond[0]]-z[bond[1]]
		r = np.sqrt(X**2.+Y**2.+Z**2.)
		fx[bond[0]] += -2.*K*(r-req)*X/r
		fy[bond[0]] += -2.*K*(r-req)*Y/r
		fz[bond[0]] += -2.*K*(r-req)*Z/r
		fx[bond[1]] += 2.*K*(r-req)*X/r
		fy[bond[1]] += 2.*K*(r-req)*Y/r
		fz[bond[1]] += 2.*K*(r-req)*Z/r	
		#E+= K*(r-req)**2.
		#print 'energybonds', E/energyunit
		i+=1
		
	##############pinning forces on terminal carbons							
	X = x[Lterm]-Lterminaleq[0]
	Y = y[Lterm]-Lterminaleq[1]
	Z = z[Lterm]-Lterminaleq[2]
	r = np.sqrt(X**2.+Y**2.+Z**2.)
	fx[Lterm] += -2.*Kpin*X
	fy[Lterm] += -2.*Kpin*Y
	fz[Lterm] += -2.*Kpin*Z	

	#if numofcarbons > 1:
	X = x[Rterm]-Rterminaleq[0]
	Y = y[Rterm]-Rterminaleq[1]
	Z = z[Rterm]-Rterminaleq[2]
	r = np.sqrt(X**2.+Y**2.+Z**2.)
	fx[Rterm] += -2.*Kpin*X
	fy[Rterm] += -2.*Kpin*Y
	fz[Rterm] += -2.*Kpin*Z	
	
	#################angles ##CHARRM22 parameters
	i = 0
	while i<len(angles):	
		angle = angles[i]
		if atomtype[angle[0]]=='C' and atomtype[angle[1]]=='C' and atomtype[angle[2]]=='C':	
			#print "C-C-C"
			thetaeq = 112.7/57.2957795
			K = 58.35		
		elif atomtype[angle[0]]=='H' and atomtype[angle[1]]=='C' and atomtype[angle[2]]=='H':
			#print "H-C-H"
			thetaeq = 107.8/57.2957795
			K = 33.00	
		else:
			#print "C-C-H"
			thetaeq = 110.7/57.2957795
			K = 37.50	
		
		v1 = np.zeros(3)
		v2 = np.zeros(3)
		v1[0] = x[angle[1]]-x[angle[0]]
		v1[1] = y[angle[1]]-y[angle[0]]
		v1[2] = z[angle[1]]-z[angle[0]]
		
		v2[0] = x[angle[2]]-x[angle[1]]
		v2[1] = y[angle[2]]-y[angle[1]]
		v2[2] = z[angle[2]]-z[angle[1]]
		
		magv1 = np.sqrt(v1[0]**2.+v1[1]**2.+v1[2]**2.)
		magv2 = np.sqrt(v2[0]**2.+v2[1]**2.+v2[2]**2.)
		
		theta = np.arccos(-(v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])/(magv1*magv2))	
		
		v1xv2 = np.zeros(3)
		v1xv2[0] = (v1[1]*v2[2]-v1[2]*v2[1])
		v1xv2[1] = (v1[2]*v2[0]-v1[0]*v2[2])
		v1xv2[2] = (v1[0]*v2[1]-v1[1]*v2[0])
		
		v1xv1xv2 = np.zeros(3)
		v2xv1xv2 = np.zeros(3)
		
		v1xv1xv2[0] = (v1[1]*v1xv2[2]-v1[2]*v1xv2[1])
		v1xv1xv2[1] = (v1[2]*v1xv2[0]-v1[0]*v1xv2[2])
		v1xv1xv2[2] = (v1[0]*v1xv2[1]-v1[1]*v1xv2[0])
		v2xv1xv2[0] = (v2[1]*v1xv2[2]-v2[2]*v1xv2[1])
		v2xv1xv2[1] = (v2[2]*v1xv2[0]-v2[0]*v1xv2[2])
		v2xv1xv2[2] = (v2[0]*v1xv2[1]-v2[1]*v1xv2[0])
		
		magv1xv1xv2 = np.sqrt(v1xv1xv2[0]**2.+v1xv1xv2[1]**2.+v1xv1xv2[2]**2.)
		magv2xv1xv2 = np.sqrt(v2xv1xv2[0]**2.+v2xv1xv2[1]**2.+v2xv1xv2[2]**2.)
		
		
		fx[angle[0]] += -2.*K*(theta-thetaeq)/magv1*v1xv1xv2[0]/magv1xv1xv2
		fy[angle[0]] += -2.*K*(theta-thetaeq)/magv1*v1xv1xv2[1]/magv1xv1xv2
		fz[angle[0]] += -2.*K*(theta-thetaeq)/magv1*v1xv1xv2[2]/magv1xv1xv2
		fx[angle[2]] += -2.*K*(theta-thetaeq)/magv2*v2xv1xv2[0]/magv2xv1xv2
		fy[angle[2]] += -2.*K*(theta-thetaeq)/magv2*v2xv1xv2[1]/magv2xv1xv2
		fz[angle[2]] += -2.*K*(theta-thetaeq)/magv2*v2xv1xv2[2]/magv2xv1xv2	
		fx[angle[1]] -= -2.*K*(theta-thetaeq)/magv1*v1xv1xv2[0]/magv1xv1xv2-2.*K*(theta-thetaeq)/magv2*v2xv1xv2[0]/magv2xv1xv2
		fy[angle[1]] -= -2.*K*(theta-thetaeq)/magv1*v1xv1xv2[1]/magv1xv1xv2-2.*K*(theta-thetaeq)/magv2*v2xv1xv2[1]/magv2xv1xv2
		fz[angle[1]] -= -2.*K*(theta-thetaeq)/magv1*v1xv1xv2[2]/magv1xv1xv2-2.*K*(theta-thetaeq)/magv2*v2xv1xv2[2]/magv2xv1xv2		
		
		
		#print theta, thetaeq
		#print (theta-thetaeq)
		#print 'energyangles', E/energyunit		
		i+=1
	
	#################dihedrals
	i = 0
	while i<len(dihedrals):
		dihedral = dihedrals[i]
		#print dihedral
		if atomtype[dihedral[0]]=='C' and atomtype[dihedral[1]]=='C' and atomtype[dihedral[2]]=='C' and atomtype[dihedral[3]]=='C':
			#print "C-C-C-C"
			V1 = 1.74
			V2 = -0.157
			V3 = 0.279
			f1 = 0
			f2 = 0
			f3 = 0
		elif atomtype[dihedral[0]]=='H' and atomtype[dihedral[1]]=='C' and atomtype[dihedral[2]]=='C' and atomtype[dihedral[3]]=='H':
			#print "H-C-C-H"
			V1 = 0.0
			V2 = 0.0
			V3 = 0.318
			f1 = 0
			f2 = 0
			f3 = 0
		else:
			#print "H-C-C-C"	
			V1 = 0.0
			V2 = 0.0
			V3 = 0.366
			f1 = 0
			f2 = 0
			f3 = 0
			
		v1 = np.zeros(3)#p2-p1
		v2 = np.zeros(3)#p3-p2
		v3 = np.zeros(3)#p4-p3
		v1[0] = x[dihedral[1]]-x[dihedral[0]]
		v1[1] = y[dihedral[1]]-y[dihedral[0]]
		v1[2] = z[dihedral[1]]-z[dihedral[0]]
		
		v2[0] = x[dihedral[2]]-x[dihedral[1]]	
		v2[1] = y[dihedral[2]]-y[dihedral[1]]
		v2[2] = z[dihedral[2]]-z[dihedral[1]]
	
		v3[0] = x[dihedral[3]]-x[dihedral[2]]
		v3[1] = y[dihedral[3]]-y[dihedral[2]]
		v3[2] = z[dihedral[3]]-z[dihedral[2]]
		
		v1xv2 = np.zeros(3)
		v1xv2[0] = (v1[1]*v2[2]-v1[2]*v2[1])
		v1xv2[1] = (v1[2]*v2[0]-v1[0]*v2[2])
		v1xv2[2] = (v1[0]*v2[1]-v1[1]*v2[0])
		
		v2xv3 = np.zeros(3)
		v2xv3[0] = (v2[1]*v3[2]-v2[2]*v3[1])
		v2xv3[1] = (v2[2]*v3[0]-v2[0]*v3[2])
		v2xv3[2] = (v2[0]*v3[1]-v2[1]*v3[0])
		
		magv2 = np.sqrt(v2[0]**2.+v2[1]**2.+v2[2]**2.)
		
		magv1xv2 = np.sqrt(v1xv2[0]**2.+v1xv2[1]**2.+v1xv2[2]**2.)
		magv2xv3 = np.sqrt(v2xv3[0]**2.+v2xv3[1]**2.+v2xv3[2]**2.)
		
		v1dotv2 = v2[0]*v1[0]+v2[1]*v1[1]+v2[2]*v1[2]
		v2dotv3 = v2[0]*v3[0]+v2[1]*v3[1]+v2[2]*v3[2]
   
		arg1 = magv2*(v1[0]*v2xv3[0]+v1[1]*v2xv3[1]+v1[2]*v2xv3[2])
		arg2 = v1xv2[0]*v2xv3[0]+v1xv2[1]*v2xv3[1]+v1xv2[2]*v2xv3[2]
		theta = np.arctan2(arg1,arg2)
		
		G1 = np.zeros(3)
		G1[0]=v1xv2[0]*magv2/(magv1xv2**2.0)
		G1[1]=v1xv2[1]*magv2/(magv1xv2**2.0)
		G1[2]=v1xv2[2]*magv2/(magv1xv2**2.0)
		
		G2 = np.zeros(3)
		G2[0]= -v1xv2[0]*v1dotv2/(magv2*(magv1xv2)**2.0)-v2xv3[0]*v2dotv3/(magv2*(magv2xv3)**2.0)
		G2[1]= -v1xv2[1]*v1dotv2/(magv2*(magv1xv2)**2.0)-v2xv3[1]*v2dotv3/(magv2*(magv2xv3)**2.0)
		G2[2]= -v1xv2[2]*v1dotv2/(magv2*(magv1xv2)**2.0)-v2xv3[2]*v2dotv3/(magv2*(magv2xv3)**2.0)
		
		G3 = np.zeros(3)
		G3[0]=v2xv3[0]*magv2/(magv2xv3**2.0)
		G3[1]=v2xv3[1]*magv2/(magv2xv3**2.0)
		G3[2]=v2xv3[2]*magv2/(magv2xv3**2.0)
		
	
		fprime = 0.5*(V1*np.sin(theta+f1)-2.*V2*np.sin(2.*theta+f2)+3.*V3*np.sin(3.*theta+f3))

		fx[dihedral[0]]+=-G1[0]*fprime
		fy[dihedral[0]]+=-G1[1]*fprime
		fz[dihedral[0]]+=-G1[2]*fprime

		        
		fx[dihedral[1]]+=(G1[0]-G2[0])*fprime
		fy[dihedral[1]]+=(G1[1]-G2[1])*fprime
		fz[dihedral[1]]+=(G1[2]-G2[2])*fprime
		
		
		fx[dihedral[2]]+= (G2[0]-G3[0])*fprime
		fy[dihedral[2]]+= (G2[1]-G3[1])*fprime
		fz[dihedral[2]]+= (G2[2]-G3[2])*fprime
		               
		fx[dihedral[3]]+= G3[0]*fprime
		fy[dihedral[3]]+= G3[1]*fprime
		fz[dihedral[3]]+= G3[2]*fprime
		i+=1
		
	fxarg[:]=fx[:]	
	fyarg[:]=fy[:]
	fzarg[:]=fz[:]
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
def Temperature(vxarg,vyarg,vzarg,timestep):
	vx = list(vxarg)
	vy = list(vyarg)
	vz = list(vzarg)
	T = 0
	i = 0
	carboncounter = 0
	while i<N:
		T += (2/(3*N*kb))*1.0/2.0*mass[i]*(vx[i]**2.0+vy[i]**2.0+vz[i]**2.0)
		if atomtype[i] == 'C' and t[timestep]>equiltime:
			#print t[timestep],equiltime
			TarrayN[carboncounter]+=(2/(3*int((tEnd-equiltime)/dt+1)*kb))*1.0/2.0*mass[i]*(vx[i]**2.0+vy[i]**2.0+vz[i]**2.0)
			carboncounter+=1
		i+=1
	#print timestep	 
	Tarraytotal[timestep]+=T
	return T

##############################Main Program##################################
traj = 0; 
while traj<numoftraj:
	Q = 0
	stepcounter=1
	neighborcounter = 1
	if traj==0:
		traj_file=open(os.path.abspath(pathname)+"/traj.xyz", "w") #windows
	print 'traj', traj	
	#######Declaring and zeroing arrays for this traj####################################		
	#xold = np.zeros(N)
	vxold = np.zeros(N)
	fxold = np.zeros(N)
	xnew = np.zeros(N)
	vxnew = np.zeros(N)
	fxnew = np.zeros(N)
	###
	#yold = np.zeros(N)
	vyold = np.zeros(N)
	fyold = np.zeros(N)
	ynew = np.zeros(N)
	vynew = np.zeros(N)
	fynew = np.zeros(N)
	###
	#zold = np.zeros(N)
	vzold = np.zeros(N)
	fzold = np.zeros(N)
	znew = np.zeros(N)
	vznew = np.zeros(N)
	fznew = np.zeros(N)
	###

	

	
	##############setting intital velocites and removing linear momentum##############################
	px = 0; py = 0;	pz = 0
	i = 0
	while i<N:
		vxold[i] = np.sqrt(kb*Teff/mass[i])*np.random.normal(0,1)
		vyold[i] = np.sqrt(kb*Teff/mass[i])*np.random.normal(0,1)
		vzold[i] = np.sqrt(kb*Teff/mass[i])*np.random.normal(0,1)
		#vxold[i] = 0.
		#vyold[i] = 0.
		#vzold[i] = 0.
		px+=mass[i]*vxold[i]
		py+=mass[i]*vyold[i]
		pz+=mass[i]*vzold[i]
		i+=1
	
	#print px,py,pz	
	i = 0
	while i<N:
		vxold[i] -= px/(N*mass[i])
		vyold[i] -= py/(N*mass[i])
		vzold[i] -= pz/(N*mass[i])
		i+=1

	
	######################################################################	
	###########call force and energy to get initial condition
	NeighborList(xold,yold,zold)
	Forces(xold,yold,zold,fxold,fyold,fzold)
	Energyarraytotal[0] =Energy(xold,yold,zold,vxold,vyold,vzold)
	Energyarray[0] +=Energy(xold,yold,zold,vxold,vyold,vzold)
	
	timestep = 0
	while timestep<tsize:
		#print timestep
		
		################new forces from last timestep become old forces at every timestep after the first#################
		if timestep>0:
			i = 0			
			while i<N:
				xold[i] = xnew[i]
				yold[i] = ynew[i]
				zold[i] = znew[i]
				vxold[i] = vxnew[i]
				vyold[i] = vynew[i]
				vzold[i] = vznew[i]
				fxold[i] = fxnew[i]
				fyold[i] = fynew[i]
				fzold[i] = fznew[i]
				fxnew[i] = 0
				fynew[i] = 0
				fznew[i] = 0
				i+=1
		
		if dynamics == 'NVE':		
			E = Energy(xold,yold,zold,vxold,vyold,vzold)
		else:
			E = 0
		T = Temperature(vxold,vyold,vzold,timestep)

		#print traj, timestep
		#print 'energy', E
		#print 'temp', T
		

					
		################Move Positions		
		i = 0
		while i<N:	
			if dynamics == 'Noneq':
				#######for temperature gradient
				if Lterm != Rterm:
					if i==Lterm:
						chixL = random.gauss(0,1)
						chiyL = random.gauss(0,1)
						chizL = random.gauss(0,1)	
						etaxL = random.gauss(0,1)
						etayL = random.gauss(0,1)
						etazL = random.gauss(0,1)	
					
						AxL = 0.5*(fxold[i]/mass[i]-gammaL*vxold[i])*dt*dt + np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*(dt**1.5)*(0.5*chixL+1/(2.*np.sqrt(3))*etaxL)
						AyL = 0.5*(fyold[i]/mass[i]-gammaL*vyold[i])*dt*dt + np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*(dt**1.5)*(0.5*chiyL+1/(2.*np.sqrt(3))*etayL)				
						AzL = 0.5*(fzold[i]/mass[i]-gammaL*vzold[i])*dt*dt + np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*(dt**1.5)*(0.5*chizL+1/(2.*np.sqrt(3))*etazL)
					
						xnew[i] = xold[i] + vxold[i]*dt+AxL
						ynew[i] = yold[i] + vyold[i]*dt+AyL
						znew[i] = zold[i] + vzold[i]*dt+AzL	
					
					
					elif i==Rterm:
						chixR = random.gauss(0,1)
						chiyR = random.gauss(0,1)
						chizR = random.gauss(0,1)	
						etaxR = random.gauss(0,1)
						etayR = random.gauss(0,1)
						etazR = random.gauss(0,1)
						
						AxR = 0.5*(fxold[i]/mass[i]-gammaR*vxold[i])*dt*dt + np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*(dt**1.5)*(0.5*chixR+1/(2.*np.sqrt(3))*etaxR)
						AyR = 0.5*(fyold[i]/mass[i]-gammaR*vyold[i])*dt*dt + np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*(dt**1.5)*(0.5*chiyR+1/(2.*np.sqrt(3))*etayR)				
						AzR = 0.5*(fzold[i]/mass[i]-gammaR*vzold[i])*dt*dt + np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*(dt**1.5)*(0.5*chizR+1/(2.*np.sqrt(3))*etazR)				
					
						xnew[i] = xold[i] + vxold[i]*dt+AxR
						ynew[i] = yold[i] + vyold[i]*dt+AyR
						znew[i] = zold[i] + vzold[i]*dt+AzR	
					else:
						xnew[i] = xold[i] + vxold[i]*dt+0.5*(fxold[i]/mass[i])*dt*dt
						ynew[i] = yold[i] + vyold[i]*dt+0.5*(fyold[i]/mass[i])*dt*dt
						znew[i] = zold[i] + vzold[i]*dt+0.5*(fzold[i]/mass[i])*dt*dt
				
				if Lterm == Rterm: ###this means that you are either simulating methane or another single carbon structure
					if i==Lterm:
						chixL = random.gauss(0,1)
						chiyL = random.gauss(0,1)
						chizL = random.gauss(0,1)	
						etaxL = random.gauss(0,1)
						etayL = random.gauss(0,1)
						etazL = random.gauss(0,1)	
						
						chixR = random.gauss(0,1)
						chiyR = random.gauss(0,1)
						chizR = random.gauss(0,1)	
						etaxR = random.gauss(0,1)
						etayR = random.gauss(0,1)
						etazR = random.gauss(0,1)
					
						AxL = 0.5*(fxold[i]/mass[i]-gammaL*vxold[i])*dt*dt + np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*(dt**1.5)*(0.5*chixL+1/(2.*np.sqrt(3))*etaxL)
						AyL = 0.5*(fyold[i]/mass[i]-gammaL*vyold[i])*dt*dt + np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*(dt**1.5)*(0.5*chiyL+1/(2.*np.sqrt(3))*etayL)				
						AzL = 0.5*(fzold[i]/mass[i]-gammaL*vzold[i])*dt*dt + np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*(dt**1.5)*(0.5*chizL+1/(2.*np.sqrt(3))*etazL)
						
						AxR = 0.5*(-gammaR*vxold[i])*dt*dt + np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*(dt**1.5)*(0.5*chixR+1/(2.*np.sqrt(3))*etaxR)
						AyR = 0.5*(-gammaR*vyold[i])*dt*dt + np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*(dt**1.5)*(0.5*chiyR+1/(2.*np.sqrt(3))*etayR)				
						AzR = 0.5*(-gammaR*vzold[i])*dt*dt + np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*(dt**1.5)*(0.5*chizR+1/(2.*np.sqrt(3))*etazR)
					
						xnew[i] = xold[i] + vxold[i]*dt+(AxL+AxR)
						ynew[i] = yold[i] + vyold[i]*dt+(AyL+AyR)
						znew[i] = zold[i] + vzold[i]*dt+(AzL+AzR)
					
					else:
						xnew[i] = xold[i] + vxold[i]*dt+0.5*(fxold[i]/mass[i])*dt*dt
						ynew[i] = yold[i] + vyold[i]*dt+0.5*(fyold[i]/mass[i])*dt*dt
						znew[i] = zold[i] + vzold[i]*dt+0.5*(fzold[i]/mass[i])*dt*dt		

			if dynamics == 'NVT':
				####for equilibrium NVT dynamics
				chix[i] = random.gauss(0,1)
				chiy[i] = random.gauss(0,1)
				chiz[i] = random.gauss(0,1)	
				etax[i] = random.gauss(0,1)
				etay[i] = random.gauss(0,1)
				etaz[i] = random.gauss(0,1)	
	
				Ax[i] = 0.5*(fxold[i]/mass[i]-gamma*vxold[i])*dt*dt + np.sqrt(2.0*kb*gamma*Teff*mass[i]**(-1.))*(dt**1.5)*(0.5*chix[i]+1/(2.*np.sqrt(3))*etax[i])
				Ay[i] = 0.5*(fyold[i]/mass[i]-gamma*vyold[i])*dt*dt + np.sqrt(2.0*kb*gamma*Teff*mass[i]**(-1.))*(dt**1.5)*(0.5*chiy[i]+1/(2.*np.sqrt(3))*etay[i])				
				Az[i] = 0.5*(fzold[i]/mass[i]-gamma*vzold[i])*dt*dt + np.sqrt(2.0*kb*gamma*Teff*mass[i]**(-1.))*(dt**1.5)*(0.5*chiz[i]+1/(2.*np.sqrt(3))*etaz[i])			
				xnew[i] = xold[i] + vxold[i]*dt+Ax[i]
				ynew[i] = yold[i] + vyold[i]*dt+Ay[i]
				znew[i] = zold[i] + vzold[i]*dt+Az[i]	
			
			if dynamics == 'NVE':
				#### for NVE dynamics			
				xnew[i] = xold[i] + vxold[i]*dt+0.5*(fxold[i]/mass[i])*dt*dt
				ynew[i] = yold[i] + vyold[i]*dt+0.5*(fyold[i]/mass[i])*dt*dt
				znew[i] = zold[i] + vzold[i]*dt+0.5*(fzold[i]/mass[i])*dt*dt
	
			i+=1

		Forces(xnew,ynew,znew,fxnew,fynew,fznew)
		
		i = 0
		while i<N:
			if dynamics == 'Noneq':
				if Lterm != Rterm:
					if i==Lterm:
						vxnew[i] = vxold[i] + 0.5*(fxold[i]/mass[i]+fxnew[i]/mass[i])*dt -gammaL*vxold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*chixL-gammaL*AxL
						vynew[i] = vyold[i] + 0.5*(fyold[i]/mass[i]+fynew[i]/mass[i])*dt -gammaL*vyold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*chiyL-gammaL*AyL
						vznew[i] = vzold[i] + 0.5*(fzold[i]/mass[i]+fznew[i]/mass[i])*dt -gammaL*vzold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*chizL-gammaL*AzL					
					elif i==Rterm:#############must change from elif to if for carbon or methane                  
						vxnew[i] = vxold[i] + 0.5*(fxold[i]/mass[i]+fxnew[i]/mass[i])*dt -gammaR*vxold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*chixR-gammaR*AxR
						vynew[i] = vyold[i] + 0.5*(fyold[i]/mass[i]+fynew[i]/mass[i])*dt -gammaR*vyold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*chiyR-gammaR*AyR
						vznew[i] = vzold[i] + 0.5*(fzold[i]/mass[i]+fznew[i]/mass[i])*dt -gammaR*vzold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*chizR-gammaR*AzR						
					else:		
						vxnew[i] = vxold[i] + 0.5*((fxold[i]+fxnew[i])/mass[i])*dt 
						vynew[i] = vyold[i] + 0.5*((fyold[i]+fynew[i])/mass[i])*dt 
						vznew[i] = vzold[i] + 0.5*((fzold[i]+fznew[i])/mass[i])*dt 
					
					
				if Lterm == Rterm:	###this means that you are either simulating methane or another single carbon structure
					if i==Lterm:
						vxnew[i] = vxold[i] + 0.5*(fxold[i]/mass[i]+fxnew[i]/mass[i])*dt -gammaL*vxold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*chixL-gammaL*AxL
						vynew[i] = vyold[i] + 0.5*(fyold[i]/mass[i]+fynew[i]/mass[i])*dt -gammaL*vyold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*chiyL-gammaL*AyL
						vznew[i] = vzold[i] + 0.5*(fzold[i]/mass[i]+fznew[i]/mass[i])*dt -gammaL*vzold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaL*TL*mass[i]**(-1.))*chizL-gammaL*AzL					
				
						vxnew[i] +=  -gammaR*vxold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*chixR-gammaR*AxR
						vynew[i] +=  -gammaR*vyold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*chiyR-gammaR*AyR
						vznew[i] +=  -gammaR*vzold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gammaR*TR*mass[i]**(-1.))*chizR-gammaR*AzR						
					else:		
						vxnew[i] = vxold[i] + 0.5*((fxold[i]+fxnew[i])/mass[i])*dt 
						vynew[i] = vyold[i] + 0.5*((fyold[i]+fynew[i])/mass[i])*dt 
						vznew[i] = vzold[i] + 0.5*((fzold[i]+fznew[i])/mass[i])*dt 						
			
			if dynamics == 'NVT':						
			
				vxnew[i] = vxold[i] + 0.5*(fxold[i]/mass[i]+fxnew[i]/mass[i])*dt -gamma*vxold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gamma*Teff*mass[i]**(-1.))*chix[i]-gamma*Ax[i]
				vynew[i] = vyold[i] + 0.5*(fyold[i]/mass[i]+fynew[i]/mass[i])*dt -gamma*vyold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gamma*Teff*mass[i]**(-1.))*chiy[i]-gamma*Ay[i]
				vznew[i] = vzold[i] + 0.5*(fzold[i]/mass[i]+fznew[i]/mass[i])*dt -gamma*vzold[i]*dt+sqrtdt*np.sqrt(2.0*kb*gamma*Teff*mass[i]**(-1.))*chiz[i]-gamma*Az[i]	

			if dynamics == 'NVE':
				vxnew[i] = vxold[i] + 0.5*((fxold[i]+fxnew[i])/mass[i])*dt 
				vynew[i] = vyold[i] + 0.5*((fyold[i]+fynew[i])/mass[i])*dt 
				vznew[i] = vzold[i] + 0.5*((fzold[i]+fznew[i])/mass[i])*dt 			

			i+=1	

################NeighborListRefresh and TrajOutput######################################
		if int(timestep/neighborcounter)==NeighborListRefreshRate:
			print timestep, "NeighborList Update"
			NeighborList(xnew,ynew,znew)
			neighborcounter+=1

		#if int(timestep/stepcounter)==TrajOutputRate:
		#	if t[timestep] >=equiltime: #equilibrate for half of trajectory before output
		#		if traj ==0:
		#			print >> traj_file, N, int((tEnd-equiltime)/(TrajOutputRate*dt)), int(stepcounter-equiltime/(TrajOutputRate*dt)+1), '\n'
		#			i = 0
		#			while i<N:
		#				print >> traj_file,atomtype[i],xold[i],yold[i],zold[i]
		#				i+=1				
		#	stepcounter+=1
#########################################################################################
		if dynamics == 'NVT':	
			i = 0
			while i<N:
				Qtotal[timestep+1] += -(mass[i]*0.5*(vxnew[i]+vxold[i])*(math.sqrt(2.0*kb*Teff*gamma*mass[i]**(-1))*sqrtdt*chix[i] - gamma*(xnew[i]-xold[i])))
				Qtotal[timestep+1] += -(mass[i]*0.5*(vynew[i]+vyold[i])*(math.sqrt(2.0*kb*Teff*gamma*mass[i]**(-1))*sqrtdt*chiy[i] - gamma*(ynew[i]-yold[i])))
				Qtotal[timestep+1] += -(mass[i]*0.5*(vznew[i]+vzold[i])*(math.sqrt(2.0*kb*Teff*gamma*mass[i]**(-1))*sqrtdt*chiz[i] - gamma*(znew[i]-zold[i])))
				Q += -(mass[i]*0.5*(vxnew[i]+vxold[i])*(math.sqrt(2.0*kb*Teff*gamma*mass[i]**(-1))*sqrtdt*chix[i] - gamma*(xnew[i]-xold[i])))
				Q += -(mass[i]*0.5*(vynew[i]+vyold[i])*(math.sqrt(2.0*kb*Teff*gamma*mass[i]**(-1))*sqrtdt*chiy[i] - gamma*(ynew[i]-yold[i])))
				Q += -(mass[i]*0.5*(vznew[i]+vzold[i])*(math.sqrt(2.0*kb*Teff*gamma*mass[i]**(-1))*sqrtdt*chiz[i] - gamma*(znew[i]-zold[i])))
				i+=1
			
			Qtotalcum[timestep+1]+=Q
			Energyarraytotal[timestep+1] = Energy(xnew,ynew,znew,vxnew,vynew,vznew)
			DeltaE = Energyarraytotal[timestep+1]-Energyarraytotal[0]
			DeltaEarraycum[timestep+1] += DeltaE 
			Energyarray[timestep+1]+=Energyarraytotal[timestep+1]
			#print 'heat', Q,  'Delta E', Energyarray[timestep+1]-Energyarray[timestep]
			print traj, timestep, Q,  'Delta E', DeltaE	

		if dynamics == 'Noneq':	
			#Q = 0
			i=Lterm
			QL[timestep+1] += -(mass[i]*0.5*(vxnew[i]+vxold[i])*(math.sqrt(2.0*kb*TL*gammaL*mass[i]**(-1))*sqrtdt*chixL - gammaL*(xnew[i]-xold[i])))
			QL[timestep+1] += -(mass[i]*0.5*(vynew[i]+vyold[i])*(math.sqrt(2.0*kb*TL*gammaL*mass[i]**(-1))*sqrtdt*chiyL - gammaL*(ynew[i]-yold[i])))
			QL[timestep+1] += -(mass[i]*0.5*(vznew[i]+vzold[i])*(math.sqrt(2.0*kb*TL*gammaL*mass[i]**(-1))*sqrtdt*chizL - gammaL*(znew[i]-zold[i])))	
			Q += -(mass[i]*0.5*(vxnew[i]+vxold[i])*(math.sqrt(2.0*kb*TL*gammaL*mass[i]**(-1))*sqrtdt*chixL - gammaL*(xnew[i]-xold[i])))
			Q += -(mass[i]*0.5*(vynew[i]+vyold[i])*(math.sqrt(2.0*kb*TL*gammaL*mass[i]**(-1))*sqrtdt*chiyL - gammaL*(ynew[i]-yold[i])))
			Q += -(mass[i]*0.5*(vznew[i]+vzold[i])*(math.sqrt(2.0*kb*TL*gammaL*mass[i]**(-1))*sqrtdt*chizL - gammaL*(znew[i]-zold[i])))				
	
			i=Rterm
			QR[timestep+1] += -(mass[i]*0.5*(vxnew[i]+vxold[i])*(math.sqrt(2.0*kb*TR*gammaR*mass[i]**(-1))*sqrtdt*chixR - gammaR*(xnew[i]-xold[i])))
			QR[timestep+1] += -(mass[i]*0.5*(vynew[i]+vyold[i])*(math.sqrt(2.0*kb*TR*gammaR*mass[i]**(-1))*sqrtdt*chiyR - gammaR*(ynew[i]-yold[i])))
			QR[timestep+1] += -(mass[i]*0.5*(vznew[i]+vzold[i])*(math.sqrt(2.0*kb*TR*gammaR*mass[i]**(-1))*sqrtdt*chizR - gammaR*(znew[i]-zold[i])))	
			Q += -(mass[i]*0.5*(vxnew[i]+vxold[i])*(math.sqrt(2.0*kb*TR*gammaR*mass[i]**(-1))*sqrtdt*chixR - gammaR*(xnew[i]-xold[i])))
			Q += -(mass[i]*0.5*(vynew[i]+vyold[i])*(math.sqrt(2.0*kb*TR*gammaR*mass[i]**(-1))*sqrtdt*chiyR - gammaR*(ynew[i]-yold[i])))
			Q += -(mass[i]*0.5*(vznew[i]+vzold[i])*(math.sqrt(2.0*kb*TR*gammaR*mass[i]**(-1))*sqrtdt*chizR - gammaR*(znew[i]-zold[i])))	

			
			Qtotalcum[timestep+1]+=Q
			Qtotal[timestep+1]=QL[timestep+1]+QR[timestep+1]
			
			#if timestep>0:
			#	Qtotal[timestep]+=Qtotal[timestep-1]
			
			Energyarraytotal[timestep+1] = Energy(xnew,ynew,znew,vxnew,vynew,vznew)
			DeltaE = Energyarraytotal[timestep+1]-Energyarraytotal[0]
			DeltaEarraycum[timestep+1] += DeltaE  
			Energyarray[timestep+1]+=Energyarraytotal[timestep+1]
			#print 'heat', Q,  'Delta E', Energyarray[timestep+1]-Energyarray[timestep]
			#print traj, timestep, Q,  'Delta E', DeltaE

#########################################################################################
		timestep+=1
	if traj==0:
		traj_file.close()
	traj+=1	

#########################################################################################
temptotal_file=open(os.path.abspath(pathname)+"/temptotal.txt", "w") #windows
tempchain_file=open(os.path.abspath(pathname)+"/tempchain.txt", "w") #windows
heatcum_file=open(os.path.abspath(pathname)+"/Energycum.txt", "w") #windows
energyflux_file=open(os.path.abspath(pathname)+"/EnergyFlux.txt", "w") #windows
i = 0
while i<tsize:
	print >> temptotal_file, t[i], float(Tarraytotal[i]/numoftraj)
	print >> heatcum_file, t[i], Qtotalcum[i]/numoftraj, DeltaEarraycum[i]/numoftraj
	print >> energyflux_file, t[i], Qtotal[i]/numoftraj, QL[i]/numoftraj, QR[i]/numoftraj, Energyarray[i]/numoftraj
	i+=1
	
i = 0
while i<numofcarbons:
	print >> tempchain_file, i+1, float(TarrayN[i]/numoftraj)
	i+=1
	
temptotal_file.close()
tempchain_file.close()
heatcum_file.close()
energyflux_file.close()

print("--- %s seconds ---" % (time.time() - start_time))

plt.figure()
plt.plot(Energyarray/numoftraj)
plt.show()

