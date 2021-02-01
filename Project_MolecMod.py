# -*- coding: utf-8 -*-
import numpy as np
from topology import Molecule
from topology import Topology
import matplotlib.pyplot as plt
import csv 

##############################################################################
def ReadTrajectory(trajFile):
    """This will read the WHOLE trajectory into memory. This will not be possible
later very large trajectores. I'll let you worry about how to deal with that yourself..."""
    trajectory=[]
    with open(trajFile, "r") as tF:
        line = tF.readline()
        while line is not "":
            #first line is number of atoms
            N = int(line.strip())
            tF.readline().strip() # second line is a comment that we throw away

            q = []
            for i in range(N):
                line = tF.readline().strip().split(" ")
                for c in line[1:]:
                    if c is not "":
                        q.append(float(c))
            trajectory.append(np.array(q))

            line = tF.readline()

    return trajectory, N

##############################################################################
def calcDistance(t,n):
    for s,q in enumerate(t):
        print("Processing step {}".format(s))
        q = q.reshape(n, 3) 
        dr = q - q[:, np.newaxis] 
        r2 = np.linalg.norm(dr, axis=2)
    return r2;

##############################################################################
def calcAtomAtomPotential(r1, r2, r0, k):
    r=np.linalg.norm(r1-r2)
    potential_atom_atom=0.5*k*(r-r0)**2
    return potential_atom_atom

##############################################################################
def calcAtomAtomForce(r1, r2, r0, k):
    force=np.zeros([2,3])
    force[0] = -k*(np.linalg.norm(r1-r2)-r0)*((r1-r2)/np.linalg.norm(r1-r2))
    force[1] = -k*(np.linalg.norm(r1-r2)-r0)*((r2-r1)/np.linalg.norm(r2-r1))
    return force

############################################################################### 
def calcBondPotentials(atoms, r0, k):
    bond_potentials = np.array([])
    for atom_i in atoms:
        for atom_j in atoms:
            bond_potentials = np.append(bond_potentials,calcAtomAtomPotential(atom_i,atom_j, r0, k))
    return bond_potentials

###############################################################################
def calcBondEnergy(atoms, r0, k):
    bond_potentials = calcBondPotentials(atoms, r0, k)
    bond_energy = 0
    for potential_i in bond_potentials:
        for potential_j in bond_potentials:
            bond_energy = bond_energy + (potential_i - potential_j)
    return bond_energy

##############################################################################
#This is used for the typology class
#r1,r2 are Nx3 dimensional vectors
#r0, k are Nx1 dimensional vectors
#numpy does all the calculations at once
def calcAtomAtomPotential_MultipleAtoms(r1, r2, r0, k):
    r=np.linalg.norm(r1-r2, axis=1)
    potential_atom_atom=0.5*k*(r-r0)**2
    return potential_atom_atom

##############################################################################
def calcBondForce_MultipleAtoms(top,xyzarray):
    indexr1=top.getBond_i_indeces()
    indexr2=top.getBond_j_indeces()

    coord_r1=xyzarray[indexr1]
    coord_r2=xyzarray[indexr2]

    r0=top.getBond_r0()
    k=top.getBond_k()
    
    return(calcAtomAtomForce_MultipleAtoms(coord_r1,coord_r2,r0,k))

##############################################################################
def calcAtomAtomForce_MultipleAtoms(r1, r2, r0, k):
    r=np.linalg.norm(r1-r2, axis=1)
    r_reverse=np.linalg.norm(r2-r1, axis=1)
    force1 = np.transpose((np.transpose(r1-r2)/r)*(-k*(r-r0)))
    force2 = np.transpose((np.transpose(r2-r1)/r_reverse)*(-k*(r-r0)))
    return np.reshape(np.array([force1, force2]),(2*len(r0),3))
    #ATTENTION
    #return array is shaped like
    #[force on r1 of first bond]
    #[force on r1 of second bond]
    #...
    #[force on r1 of last bond]
    #[force on r2 of first bond]
    #[force on r2 of second bond]
    #...
    #[force on r2 of last bond]
            

##############################################################################
def calcAnglePotential(r1, r2, r3, theta0, k_theta):
    v1 = r1-r2
    v2 = r1-r3
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    theta = np.arccos(np.dot(v1,v2)/(v1_norm*v2_norm))
    
    angle_potential = 0.5*k_theta*(theta-theta0)**2
    
    return angle_potential

##############################################################################
def calcAnglePotential_MultipleAtoms(r1, r2, r3, theta0, k_theta):
    v1 = r1-r2
    v2 = r1-r3
    v1_norm = np.linalg.norm(r1-r2, axis=1)
    v2_norm = np.linalg.norm(r1-r3, axis=1)
    dots = (v1 * v2).sum(axis=1)
    theta = np.arccos(dots/(v1_norm*v2_norm))
    angle_potential = 0.5*k_theta*(theta-theta0)**2
    
    return angle_potential

##############################################################################
def calcAngleForce_MultipleAtoms(top,xyzarray):
    indexr1=top.getAngle_i_indeces()
    indexr2=top.getAngle_j_indeces()
    indexr3=top.getAngle_k_indeces()

    coord_r1=xyzarray[indexr1]
    coord_r2=xyzarray[indexr2]
    coord_r3=xyzarray[indexr3]

    k0=top.getAngle_k0()
    theta0=top.getAngle_theta0()
    
    #return(calcAnglePotential_MultipleAtoms(coord_r1, coord_r2, coord_r3, theta0, k0))
    return(calcAngleForce(coord_r1, coord_r2, coord_r3, theta0, k0 ))
    
##############################################################################
def calcAngleForce(r1, r2, r3, theta0, k_theta):
    v1 = r1-r2
    v2 = r1-r3
    v1_norm = np.linalg.norm(r1-r2, axis=1)
    v2_norm = np.linalg.norm(r1-r3, axis=1)
    dots = (v1 * v2).sum(axis=1)
    theta = np.arccos(dots/(v1_norm*v2_norm))
   
    p_a = np.cross(v1,np.cross(v1,v2))
    p_a = p_a/np.linalg.norm(p_a)
    f_a = np.transpose(np.transpose(p_a)*(-2*k_theta*(theta-theta0)))/np.linalg.norm(v1)
    #print(f_a)
    
    p_c = np.cross(r2-r3,np.cross(v1,v2))
    p_c = p_a/np.linalg.norm(p_c)
    f_c = np.transpose(np.transpose(p_c)*(-2*k_theta*(theta-theta0)))/np.linalg.norm(v2)
    #print(f_c)
    
    f_b = -f_a - f_c
    #print(f_b)
    
    #force=np.zeros(3,3)
    return np.reshape(np.array([f_a, f_b, f_c]),(3*len(theta),3))
    #pay ATTENTION
    #return array is shaped like
    #[f_a of angle 1]
    #[f_a of angle 2]
    #[f_b of angle 1]
    #[f_b of angle 2]
    #[f_c of anlge 1]
    #[f_c of angle 2]

##############################################################################
def numIntegrateVelocityVerlet(top, xyzarray, timestep, nr_timesteps):
    velocities = np.zeros([len(top.getElements()),3])
    for i in range(0, len(top.getElements())):
        u = np.random.uniform(size=3)
        u /= np.linalg.norm(u) # normalize
        vRand = 0.01*u
        velocities[i] = vRand
    with open("output.xyz", "w") as f:
        writer = csv.writer(f,delimiter=' ',lineterminator='\n')        
        for j in range(0, nr_timesteps):
            print('timestep:', j)
            #udate positions
            bond_forces = calcBondForce_MultipleAtoms(top, xyzarray)
            angle_forces = calcAngleForce_MultipleAtoms(top, xyzarray)
            
            #bondforces
            bond_forces_split = np.split(bond_forces, 2)
            
            indexr1=top.getBond_i_indeces()
            indexr2=top.getBond_j_indeces()
            
            xyzarray[indexr1] += timestep*velocities[indexr1] + (((timestep)**2)/2) *np.transpose((np.transpose(bond_forces_split[0])/top.getMasses()[indexr1]))
            xyzarray[indexr2] += timestep*velocities[indexr2] + (((timestep)**2)/2) *np.transpose((np.transpose(bond_forces_split[1])/top.getMasses()[indexr2]))
            
            
            #angleforces
            angle_forces_split = np.split(angle_forces, 3)
            
            indexr1=top.getAngle_i_indeces()
            indexr2=top.getAngle_j_indeces()
            indexr3=top.getAngle_k_indeces()
    
            xyzarray[indexr1] +=  (((timestep)**2)/2) *np.transpose((np.transpose(angle_forces_split[0])/top.getMasses()[indexr1]))
            xyzarray[indexr2] +=  (((timestep)**2)/2) *np.transpose((np.transpose(angle_forces_split[1])/top.getMasses()[indexr2]))
            xyzarray[indexr3] +=  (((timestep)**2)/2) *np.transpose((np.transpose(angle_forces_split[2])/top.getMasses()[indexr3]))
            
            ################
            #Lennard Jones
            #NEED TO REPLACE FOR LOOPS BY NUMPY SYNTAX
            lennardjonesforces = calcLennardJonesForces(top, xyzarray)
            for i in range(0, len(top.getElements())):
                for k in range(0,len(top.getElements())):
                       xyzarray[k] += (((timestep)**2)/2) *np.transpose((np.transpose(lennardjonesforces[i][k])/top.getMasses()[k]))
                        
            ################
            
            #keep the old velocities for temperature 
            velocities_old = np.copy(velocities)
            
            #update velocities
            
            #calc forces of updated xyzarray
            bond_forces_new = calcBondForce_MultipleAtoms(top, xyzarray)
            angle_forces_new = calcAngleForce_MultipleAtoms(top, xyzarray)
            
            bond_forces_split_new = np.split(bond_forces_new, 2)
            angle_forces_split_new = np.split(angle_forces_new, 3)
            
            indexr1=top.getBond_i_indeces()
            indexr2=top.getBond_j_indeces()
            
            velocities[indexr1] += (timestep/2)*np.transpose(np.transpose((bond_forces_split[0]+bond_forces_split_new[0]))/top.getMasses()[indexr1])
            velocities[indexr2] += (timestep/2)*np.transpose(np.transpose((bond_forces_split[1]+bond_forces_split_new[1]))/top.getMasses()[indexr2])
            
            indexr1=top.getAngle_i_indeces()
            indexr2=top.getAngle_j_indeces()
            indexr3=top.getAngle_k_indeces()
            
            velocities[indexr1] += (timestep/2)*np.transpose((np.transpose(angle_forces_split[0]+angle_forces_split_new[0]))/top.getMasses()[indexr1])
            velocities[indexr2] += (timestep/2)*np.transpose((np.transpose(angle_forces_split[1]+angle_forces_split_new[1]))/top.getMasses()[indexr2])
            velocities[indexr3] += (timestep/2)*np.transpose((np.transpose(angle_forces_split[2]+angle_forces_split_new[2]))/top.getMasses()[indexr3])
            
            ################################
            #Lennard Jones
            #NEED TO REPLACE FOR LOOPS BY NUMPY SYNTAX
            lennardjonesforces_new = calcLennardJonesForces(top, xyzarray)
            
            for i in range(0, len(top.getElements())):
                for k in range(0,len(top.getElements())):
                        velocities[k] += (timestep/2)*np.transpose(np.transpose((lennardjonesforces[i][k]+lennardjonesforces_new[i][k]))/top.getMasses()[k])
            ################################
            
            print(xyzarray)
            #write positions in xyz file
            writer.writerow([len(top.getElements())])
            writer.writerow(['t=']+[j])
            k = 0
            for atom in xyzarray:
                writer.writerow([top.getElements()[k]]+[atom[0]]+[atom[1]]+[atom[2]])
                k += 1
            
            #KEEP TEMPERATURE STABLE
            #velocities = rescaleVelocity(velocities_old, velocities)
            #######################
            
            #Temperature
            print("The temperature is: ", getTemperature(top, velocities))
            
    #return(positions, velocities)
    return 1

##############################################################################
def numIntegrateEuler(top, xyzarray, timestep, nr_timesteps):
    velocities = np.zeros([len(top.getElements()),3])
    for i in range(0, len(top.getElements())):
        u = np.random.uniform(size=3)
        u /= np.linalg.norm(u) # normalize
        vRand = 0.01*u
        velocities[i] = vRand
    with open("output.xyz", "w") as f:
        writer = csv.writer(f,delimiter=' ',lineterminator='\n')        
        for j in range(0, nr_timesteps):
            print('timestep:', j)
            #udate positions
            bond_forces = calcBondForce_MultipleAtoms(top, xyzarray)
            angle_forces = calcAngleForce_MultipleAtoms(top, xyzarray)
            
            #bondforces
            bond_forces_split = np.split(bond_forces, 2)
            
            indexr1=top.getBond_i_indeces()
            indexr2=top.getBond_j_indeces()
            
            xyzarray[indexr1] += timestep*velocities[indexr1] + (((timestep)**2)/2) *np.transpose((np.transpose(bond_forces_split[0])/top.getMasses()[indexr1]))
            xyzarray[indexr2] += timestep*velocities[indexr2] + (((timestep)**2)/2) *np.transpose((np.transpose(bond_forces_split[1])/top.getMasses()[indexr2]))
            
            
            #angleforces
            angle_forces_split = np.split(angle_forces, 3)
            
            indexr1=top.getAngle_i_indeces()
            indexr2=top.getAngle_j_indeces()
            indexr3=top.getAngle_k_indeces()
    
            xyzarray[indexr1] +=  (((timestep)**2)/2) *np.transpose((np.transpose(angle_forces_split[0])/top.getMasses()[indexr1]))
            xyzarray[indexr2] +=  (((timestep)**2)/2) *np.transpose((np.transpose(angle_forces_split[1])/top.getMasses()[indexr2]))
            xyzarray[indexr3] +=  (((timestep)**2)/2) *np.transpose((np.transpose(angle_forces_split[2])/top.getMasses()[indexr3]))
            
            ################
            #Lennard Jones
            #NEED TO REPLACE FOR LOOPS BY NUMPY SYNTAX
            lennardjonesforces = calcLennardJonesForces(top, xyzarray)
            for i in range(0, len(top.getElements())):
                for k in range(0,len(top.getElements())):
                       xyzarray[k] += (((timestep)**2)/2) *np.transpose((np.transpose(lennardjonesforces[i][k])/top.getMasses()[k]))
                        
            ################
            
            #keep the old velocities for temperature 
            velocities_old = np.copy(velocities)
            
            #update velocities
            indexr1=top.getBond_i_indeces()
            indexr2=top.getBond_j_indeces()
            
            velocities[indexr1] += timestep*np.transpose(np.transpose((bond_forces_split[0]))/top.getMasses()[indexr1])
            velocities[indexr2] += timestep*np.transpose(np.transpose((bond_forces_split[1]))/top.getMasses()[indexr2])
            
            indexr1=top.getAngle_i_indeces()
            indexr2=top.getAngle_j_indeces()
            indexr3=top.getAngle_k_indeces()
            
            velocities[indexr1] += timestep*np.transpose((np.transpose(angle_forces_split[0]))/top.getMasses()[indexr1])
            velocities[indexr2] += timestep*np.transpose((np.transpose(angle_forces_split[1]))/top.getMasses()[indexr2])
            velocities[indexr3] += timestep*np.transpose((np.transpose(angle_forces_split[2]))/top.getMasses()[indexr3])
            
            ################################
            #JUST A TRY
            #NEED TO REPLACE FOR LOOPS BY NUMPY SYNTAX
            for i in range(0, len(top.getElements())):
                for k in range(0,len(top.getElements())):
                       velocities[k] += timestep*np.transpose(np.transpose((lennardjonesforces[i][k]))/top.getMasses()[k])
            
            ################################
            print(xyzarray)
            #write positions in xyz file
            writer.writerow([len(top.getElements())])
            writer.writerow(['t=']+[j])
            k = 0
            for atom in xyzarray:
                writer.writerow([top.getElements()[k]]+[atom[0]]+[atom[1]]+[atom[2]])
                k += 1
            
            #KEEP TEMPERATURE STABLE
            #velocities = rescaleVelocity(velocities_old, velocities)
            ########################
            
            #Temperature
            print("The temperature is: ", getTemperature(top, velocities))
            
            
    #return(positions, velocities)
    return 1

###############################################################################
def calcLennardJonesInteraction(top, xyzarray):
    trajectory = np.array([np.concatenate(xyzarray, axis=None)])
    #calculate distances between all the atoms
    distances = calcDistance(trajectory, len(top.getElements()))
    #print(distances)
    sigmas = top.getAtom_Sigma()
    epsilons = top.getAtom_Epsilon()
    
    sigmas_ij = calcSigmas_ij(top, sigmas)
    epsilons_ij = calcEpsilons_ij(top,epsilons)
    
    U = 4*epsilons_ij*((sigmas_ij/distances)**12-(sigmas_ij/distances)**6)
    
    return U

##############################################################################
def calcLennardJonesForces(top, xyzarray):
    trajectory = np.array([np.concatenate(xyzarray, axis=None)])
    #calculate distances between all the atoms
    distances = calcDistance(trajectory, len(top.getElements()))
    
    distances += np.identity(len(top.getElements()))

    sigmas = top.getAtom_Sigma()
    epsilons = top.getAtom_Epsilon()
    
    sigmas_ij = calcSigmas_ij(top, sigmas)
    epsilons_ij = calcEpsilons_ij(top,epsilons)
    
    U1 = ((24*epsilons_ij)/distances)
    U1 -= np.diag(U1)
    U2 = (sigmas_ij/distances)
    U2 -= np.diag(U2)
    
    U = U1*(2*U2**12-U2**6)
    
    U -= np.diag(U)
    
    for s,q in enumerate(trajectory):
        q = q.reshape(len(top.getElements()), 3) 
        vectors = q - q[:, np.newaxis] 
   
    #we need to divide vectors by its length
    vectors_normed = np.transpose(np.transpose(vectors)/distances)
    
    forces = np.transpose(U*np.transpose(vectors_normed))
    
    return forces
   
##############################################################################
def calcSigmas_ij(top,sigmas):
    trajectory = np.array([np.concatenate(sigmas, axis=None)])
    for s,q in enumerate(trajectory):
        sigmas_ij = 0.5*(q + q[:, np.newaxis] )
    return sigmas_ij;

##############################################################################
def calcEpsilons_ij(top,epsilons):
    trajectory = np.array([np.concatenate(epsilons, axis=None)])
    for s,q in enumerate(trajectory):
        epsilons_ij = np.sqrt((q * q[:, np.newaxis]))
    return epsilons_ij;

###############################################################################
def calcDiheadralPotential(top, xyzarray):
    #STILL TO DO
    return 1

##############################################################################
def getTemperature(top, velocities):
    #STILL TO DO
    k_B = 0.0083144621
    N_f = 3*len(top.getElements())
    
    #kinetic energy of one atom k = 0.5*m+v^2
    kinetic = 0.5*top.getMasses()*np.linalg.norm(velocities,axis=1)**2
    
    E_kin = np.sum(kinetic)
    
    T = (2*E_kin)/(k_B*N_f)
    return T

###############################################################################
def rescaleVelocity(v_old, v_new):
    v_new_normed = np.transpose(np.transpose(v_new)/np.linalg.norm(v_new, axis=1))
    
    v_rescaled = np.transpose(np.transpose(v_new_normed)*np.linalg.norm(v_old, axis=1))
    
    return v_rescaled

#main
##############################################################################
#t, n = ReadTrajectory("Hydrogen.xyz")
   
top=Topology()
top.ReadTopologyFile("water_h2.xml")

xyzarray=np.array([[0.3,14,10.2],
[0.5,13,10.6],
[0.9,14,10.8],
[20,1,20.22],
[20,1,20.5], 
[20, 1, 21],
[30,11,0.3],
[30,10.3,0.7], 
[30, 11, 0.9],
[10.2,20.2,4],
[10.6,20,3.6],
[15,30,30],
[15.5,29.5,30.5],
[19, 5, 14],
[19.5,5,14.2],
[19.3,5,14.7],
[19.7,5,14.9],
[19.9,5,14.1]],dtype=float)

bond_forces = calcBondForce_MultipleAtoms(top, xyzarray)
#print(bond_forces)

angle_forces = calcAngleForce_MultipleAtoms(top, xyzarray)
#print(angle_forces)

numIntegrateVelocityVerlet(top, xyzarray, 0.02, 1000)

#U = calcLennardJonesForces(top, xyzarray)
#print(U)




