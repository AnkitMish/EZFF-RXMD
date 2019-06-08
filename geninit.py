import numpy as np
import sys, pickle

from scipy.io import FortranFile

def getAtomNames(ffieldName, isLG=False):
	
	atomNames = []

	with open(ffieldName, 'r') as ff:
		ff.readline()
		numParams = int(ff.readline().strip().split()[0])

		for i in range(numParams):
			ff.readline()
		
		numAtomNames = int(ff.readline().strip().split()[0])
		ff.readline()
		ff.readline()
		ff.readline()

		for i in range(numAtomNames):
			atomNames.append(ff.readline().strip().split()[0])	
			ff.readline()
			ff.readline()
			ff.readline()

			if (isLG): ff.readline()
			print('%d ----> %s' %(i+1, atomNames[-1]))

	#print(atomNames)
	return atomNames

def getBox(la,lb,lc,angle1,angle2,angle3):

	H  = np.zeros((3,3))
	Hi = np.zeros((3,3))

	lal = angle1 * (np.pi / 180.0)
	lbe = angle2 * (np.pi / 180.0)
	lga = angle3 * (np.pi / 180.0)

	hh1 = lc * (np.cos(lal) - np.cos(lbe)* np.cos(lga))/ np.sin(lga)
	hh2 = lc * pow(1.0 - pow(np.cos(lal),2) - pow(np.cos(lbe),2) - pow(np.cos(lga),2) + 2 * np.cos(lal) * np.cos(lbe) * np.cos(lga), 0.5)/np.sin(lga)

	H[0,0] = la
	H[1,0] = 0.0
	H[2,0] = 0.0
	H[0,1] = lb * np.cos(lga)
	H[1,1] = lb * np.sin(lga)
	H[2,1] = 0.0
	H[0,2] = lc * np.cos(lbe)
	H[1,2] = hh1
	H[2,2] = hh2

	print('---------------Hmatrix------------------')
	print('%12.6f   %12.6f    %12.6f' %(H[0,0], H[0,1], H[0,2]))
	print('%12.6f   %12.6f    %12.6f' %(H[1,0], H[1,1], H[1,2]))
	print('%12.6f   %12.6f    %12.6f' %(H[2,0], H[2,1], H[2,2]))

	Hi = np.linalg.inv(H)
	
	print('---------------Hinv--------------------')
	print('%12.6f   %12.6f    %12.6f' %(Hi[0,0], Hi[0,1], Hi[0,2]))
	print('%12.6f   %12.6f    %12.6f' %(Hi[1,0], Hi[1,1], Hi[1,2]))	
	print('%12.6f   %12.6f    %12.6f' %(Hi[2,0], Hi[2,1], Hi[2,2]))

	return (H, Hi)

#(H,Hi) = getBox(20,20,20,90,90,90)
#for i in range(3):
#	print('%12.6f   %12.6f    %12.6f' %(H[i,0], H[i,1], H[i,2]))
#for i in range(3):
#	print('%12.6f   %12.6f    %12.6f' %(Hi[i,0], Hi[i,1], Hi[i,2]))


def GenerateBinary(xyzFile, vprocs):

	ffieldName = sys.argv[1]
	atomNames = getAtomNames(ffieldName)
	atomLen = len(atomNames)

	natoms = 0
	l1, l2, l3, lalpha, lbeta, lgamma = None, None, None, None, None, None
	ctype0, pos0,  itype0, itype1 = [], None, None, None
	procs = vprocs[0]*vprocs[1]*vprocs[2]
	lnatoms, lnatoms1, lnatoms2 = np.zeros((procs,1)), np.zeros((procs,1)), np.zeros((procs,1))
	rr = np.zeros((3,1))
	lbox = np.zeros((3,1))
	obox = np.zeros((3,1))
	H, Hi = np.zeros((3,3)), np.zeros((3,3))
	# Read xyz coordinate, normalize them and prepare itype0 array
	with open(xyzFile, 'r') as ff:

		# Read number of atoms
		natoms = int(ff.readline().strip().split()[0])

		# Read lattice constant values
		line = ff.readline().strip().split()
		l1, l2, l3, lalpha, lbeta, lgamma = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])

		# Evaluate H and Hinverse matrix		
		H, Hi =  getBox(l1, l2, l3, lalpha, lbeta, lgamma)

		# Initialize atom list, position array and atom ID array
		ctype0 = []
		pos0   = np.zeros((natoms,3), np.float64)
		itype0 = np.zeros((natoms, ), np.float64)
		itype1 = np.zeros((natoms, ), np.float64) 

		for i in range(natoms):
			line = ff.readline().strip().split()
			atype, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
			
			for j in range(atomLen):	
				if atype == atomNames[j]:
					itype0[i] = j+1
					break

			pos0[i] = Hi.dot(np.array([x,y,z]))
			itype1[i] = itype0[i] + pow(10,-13)*(i+1) + pow(10,-14)
			#print(itype1[i])
			#print('%1.14f' %itype1[i])

	# Wrap back coordinates contained in pos0 if they are out of box
	rmin, rmax = np.zeros((3,1)), np.zeros((3,1))
	rmin[0], rmin[1], rmin[2] = np.min(pos0[:,0]), np.min(pos0[:,1]), np.min(pos0[:,2])
	pos0[:,0] = pos0[:,0] - rmin[0]
	pos0[:,1] = pos0[:,1] - rmin[1]
	pos0[:,2] = pos0[:,2] - rmin[2]

	# Wrap back more if necessary
	pos0[:,0] = pos0[:,0]%1.0
	pos0[:,1] = pos0[:,1]%1.0
	pos0[:,2] = pos0[:,2]%1.0
	
	# Shift by a small amount to avoid coordinates at zero
	pos0 = pos0 + pow(10,-9)	

	# Check minimum and maximum values of coordinates in pos0
	rmin[0], rmin[1], rmin[2] = np.min(pos0[:,0]), np.min(pos0[:,1]), np.min(pos0[:,2])
	rmax[0], rmax[1], rmax[2] = np.max(pos0[:,0]), np.max(pos0[:,1]), np.max(pos0[:,2])

	print('rmin ------------> %12.6f  %12.6f  %12.6f' %(rmin[0], rmin[1], rmin[2]))
	print('rmax ------------> %12.6f  %12.6f  %12.6f' %(rmax[0], rmax[1], rmax[2]))

	
	# count how many atoms per MPI domain, get lnatoms()
	for n in range(natoms):
		i = int(pos0[n,0] * vprocs[0])
		j = int(pos0[n,1] * vprocs[1])
		k = int(pos0[n,2] * vprocs[2])

		sID = int(i + j*vprocs[0] + k*vprocs[0]*vprocs[1])
		lnatoms[sID] += int(1)	

	# Get prefix sum
	lnatoms1 = np.cumsum(lnatoms)
	
	# To avoid opening too many files, sort atom data using disk
	lbox = 1/np.array(vprocs)
	file1 = FortranFile('all1.bin', 'w')
	for n in range(natoms):
		i = int(pos0[n,0] * vprocs[0])
		j = int(pos0[n,1] * vprocs[1])
		k = int(pos0[n,2] * vprocs[2])
	
		sID = int(i + j * vprocs[0] + k * vprocs[0] * vprocs[1])
		obox = np.multiply(lbox, np.array([i,j,k]))

		pos0[n] = pos0[n] - obox

		#ii = int(lnatoms1[sID]) + int(lnatoms2[sID])
		#index = ii*32 + 1
		#file1.seek(index)
		file1.write_record(pos0[n,:3], itype1[n], 0.0)
		#print('%1.14f' %itype1[n])
		#file1.write_record(itype1[n])
		#file1.write_record(0.0)
		#print(pos0[n,0].nbytes)
		#print(itype0[n].nbytes)
		#pos0[n,:3].tofile(file1)
		#itype0[n].tofile(file1)
	
		lnatoms2[sID] += 1

	file1.close()

	# error check
	for i in range(procs):
		print(int(lnatoms[i]), int(lnatoms2[i]))
		assert lnatoms[i] == lnatoms2[i], 'Error %d ---> %d,  %d' %(i+1, lnatoms[i], lnatoms2[i])


	# Create Reax binary and write final xyz data to a file for error checking
	file2 = FortranFile('all1.bin', 'r')
	file3 = open('geninit1.xyz', 'w')
	file4 = FortranFile('rxff1.bin', 'w')

	file4.write_record(procs, vprocs)
	file4.write_record(lnatoms)
	file4.write_record(0)
	file4.write_record(l1, l2, l3, lalpha, lbeta, lgamma)

	file3.write('%d\n' %(np.sum(np.array(lnatoms))))
	file3.write('%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n' %(l1,l2,l3,lalpha,lbeta,lgamma))

	vv = np.zeros((3,1))
	rr1 = np.zeros((3,1))
	for myid in range(procs):
		i = myid % vprocs[0]
		j = (myid/vprocs[0])%vprocs[1]	
		k = myid/(vprocs[0]*vprocs[1])
		obox = np.multiply(lbox, np.array([i,j,k]))	
		
		for n in range(int(lnatoms[myid])):
			records = file2.read_record('f8')
			#print(records)
			#dtype = file2.read_record('f8')
			#print('%1.14f' %dtype[0])
			#qq = file2.read_record('f8')
			#print(qq)
			
			file4.write_record(records[:3])
			file4.write_record(vv)
			file4.write_record(records[4])
			file4.write_record(records[3])
			file4.write_record(0.0)
			file4.write_record(0.0)

			rr1 = records[:3] + obox
				
			rr1 = H.dot(rr1)
			dtype = records[3]
			gid = int((dtype - int(dtype))*pow(10,13))
			ity = int(dtype)-1
			file3.write('%s %12.6f %12.6f %12.6f %6d\n' %(atomNames[ity], rr1[0], rr1[1], rr1[2], gid))
			

xyzFile = '000900301.xyz'
GenerateBinary(xyzFile, [1,1,1])

