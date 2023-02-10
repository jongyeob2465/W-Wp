## import
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor, hist #-> hist is deprecated in coffea v0.8.0
#from coffea import processor
import numpy as np
import matplotlib.pyplot as plt
from coffea import lumi_tools
import glob
from coffea.util import load, save
import uproot as root
from matplotlib import rc
from matplotlib import gridspec

Wp1000_50 = np.array ([0.0116046,0.56422,0.0261852,1.30055e-05,5.1505e-08,4.16952e-10,3.4968e-12])
Wp1000_100 = np.array([0.067459,2.18764,0.178531,0.000207972,8.2481e-07,6.6804e-09,6.986e-11])
Wp1000_150 = np.array([0.228731,4.8175,0.51518,0.00105342,4.17612e-06,3.38117e-08,3.54659e-10])
Wp1000_250 = np.array([1.26102,12.8454,1.7484,0.008122,3.21727e-05,2.60589e-07,2.73561e-09])
Wp1000_300 = np.array([2.33629,17.9567,2.59808,0.0168035,6.6812e-05,5.4172e-07,5.6665e-09])

Wp2000_50 = np.array ([0.000133893,0.0044943,0.0197352,0.00102895,7.2067e-08,4.8336e-10,4.76428e-12])
Wp2000_100 = np.array([0.00175723,0.0222595,0.075638,0.0067392,1.15382e-06,7.7358e-09,7.6202e-11])
Wp2000_150 = np.array([0.008468,0.065569,0.163771,0.0186881,5.8405e-06,3.91796e-08,3.85262e-10])
Wp2000_250 = np.array([0.061327,0.301369,0.41267,0.058509,4.49486e-05,3.02286e-07,2.9719e-09])
Wp2000_300 = np.array([0.120562,0.52398,0.55292,0.083439,9.2975e-05,6.2627e-07,6.1669e-09])

Wp3000_50 = np.array ([1.95426e-05,0.000210384,0.00063027,0.00154351,1.54252e-07,6.3462e-10,5.529e-12])
Wp3000_100 = np.array([0.00031218,0.0015962,0.00295763,0.0060482,2.46454e-06,1.01735e-08,8.8313e-11])
Wp3000_150 = np.array([0.00156806,0.0063575,0.0081658,0.0131067,1.24337e-05,5.1489e-08,4.46501e-10])
Wp3000_250 = np.array([0.0116719,0.040087,0.0330963,0.0315777,9.3104e-05,3.96668e-07,3.45265e-09])
Wp3000_300 = np.array([0.0231424,0.076261,0.054122,0.0408398,0.000186345,8.1872e-07,7.1391e-09])

Wp4000_50  = np.array([1.9675e-06,2.61049e-05,4.7599e-05,0.000214378,1.01027e-05,1.01571e-09,6.9662e-12])
Wp4000_100  = np.array([9.7471e-05,0.000323887,0.000302576,0.00087075,6.1102e-05,1.6236e-08,1.11313e-10])
Wp4000_150 = np.array([0.00049141,0.00154348,0.00108776,0.00204639,0.000155353,8.1993e-08,5.6402e-10])
Wp4000_250 = np.array([0.003655,0.0110344,0.006153,0.0061241,0.000412738,6.2505e-07,4.33783e-09]) 
Wp4000_300 = np.array([0.0072515,0.0215552,0.0112283,0.008813,0.00054459,1.27101e-06,8.9688e-09])

Wp5000_50  = np.array([7.5949e-06,6.18e-06,1.44951e-05,1.66897e-05,2.41901e-09,9.8308e-12])
Wp5000_100 = np.array([3.9754e-05,0.000116267,6.3345e-05,8.0711e-05,6.4537e-05,3.8617e-08,1.57221e-10])
Wp5000_150 = np.array([0.00020075,0.00058,0.000284369,0.00026064,0.000135894,1.93305e-07,7.9431e-10])
Wp5000_250 = np.array([0.00149037,0.0042803,0.00194047,0.00125951,0.000295556,1.35778e-06,6.0792e-09])
Wp5000_300 = np.array([0.00295548,0.0084144,0.0037229,0.0021381,0.000358285,2.54809e-06,1.24227e-08])
#    Wp Mass np.array(= 6000 GeV cross section list
Wp5000_30  = np.array([9497.2,51954095088,0.0052842366,2.91816e-07  ,1.68752e-11])
Wp6000_30  = np.array([9497.0,2.51954047434,0.00527332472,6.6101e-07,4.11882e-11])

Wp6000_50  = np.array([4.02567e-06,3.3863e-06,1.62968e-06,1.70545e-06,2.60733e-06,1.3833e-07,1.66768e-11])
Wp6000_100 = np.array([1.91786e-05,5.3933e-05,2.32463e-05,1.49191e-05,1.09895e-05,7.7099e-07,2.67001e-10])
Wp6000_150 = np.array([9.6365e-05,0.000271664,0.000114014,6.2909e-05,2.68614e-05,1.81362e-06,1.34731e-09])
Wp6000_250 = np.array([0.00071745,0.00201062,0.00082965,0.000398558,8.2055e-05,4.17415e-06,9.9767e-09])
Wp6000_300 = np.array([0.0014224,0.003976,0.00161486,0.00073735,0.000115279,5.1893e-06,1.97937e-08])

Sig4000 = np.array([0.000197131,0.000223371,0.000295177,0.000321921,0.00050849,5.8816e-05,1.62243e-08,1.11515e-10])
Sig5000 = np.array([7.9745e-05,7.6431e-05,6.2657e-05,4.0026e-05,3.8197e-05,6.1302e-05,3.85479e-08,1.57113e-10 ])
Sig6000 = np.array([1.9155e-05,5.3917e-05,2.31681e-05,1.46845e-05,1.05071e-05,7.3869e-07,2.6715e-10])

print("now selection start")

mass = [5000,6000]
coupling = [30]
number = [1,2,3,4,5]
event = 1000000
lumi = 138000

for i in range(len(mass)):
	for j in range(len(coupling)):
		for k in range(len(number)):
	
			path = '/x6/cykim/condor/Delphes_cms/condorDelpyOut/Delphes_' + str(mass[i]) + '_' + str(coupling[j]) + '_'+str(number[k])+ '.root'
	
			print("run start")		
			print(path)
			globals()['weight'+str(k)] = lumi * globals()['Wp'+str(mass[i])+'_'+str(coupling[j])]  / event                                                  # weight
			print("weight :",globals()['weight'+str(j)])
			
			SM_MET      = root.open(path)['Delphes']['MissingET']['MissingET.MET'].array()
			SM_Electron = root.open(path)['Delphes']['Electron']['Electron.PT'].array()
			SM_MPHI     = root.open(path)['Delphes']['MissingET']['MissingET.Phi'].array()
			SM_EPHI     = root.open(path)['Delphes']['Electron']['Electron.Phi'].array()
			SM_EETA     = root.open(path)['Delphes']['Electron']['Electron.Eta'].array()
			SM_JNUM     = root.open(path)['Delphes']['Jet']['Jet.PT'].array()
			
			cut0 = len(ak.flatten(SM_MET))

			mymask = ak.num(SM_Electron) > 0
			SM_MET = SM_MET[mymask] 
			SM_Electron = SM_Electron[mymask] 
			SM_MPHI = SM_MPHI[mymask] 
			SM_EPHI = SM_EPHI[mymask] 
			SM_EETA = SM_EETA[mymask]
			SM_JNUM = SM_JNUM[mymask]
			###########################################
			cut1 = len(ak.flatten(SM_MET))

			mask_e_2veto = ak.num(SM_Electron[SM_Electron > 25]) == 1  
			SM_MET = SM_MET[mask_e_2veto] 
			SM_Electron = SM_Electron[mask_e_2veto] 
			SM_MPHI = SM_MPHI[mask_e_2veto] 
			SM_EPHI = SM_EPHI[mask_e_2veto] 
			SM_EETA = SM_EETA[mask_e_2veto]
			SM_JNUM = SM_JNUM[mask_e_2veto]	
			###########################################
			cut2 = len(ak.flatten(SM_MET))	
			
			mask_e_pt = ak.num(SM_Electron[SM_Electron > 50]) == 1  
			SM_MET = SM_MET[mask_e_pt] 
			SM_Electron = SM_Electron[mask_e_pt] 
			SM_MPHI = SM_MPHI[mask_e_pt] 
			SM_EPHI = SM_EPHI[mask_e_pt] 
			SM_EETA = SM_EETA[mask_e_pt]
			SM_JNUM = SM_JNUM[mask_e_pt]
			
			test = ak.num(SM_Electron) > 0                             ## mask by event
			
			SM_MET = SM_MET          [test] 
			SM_Electron = SM_Electron[test] 
			SM_MPHI = SM_MPHI        [test] 
			SM_EPHI = SM_EPHI        [test] 
			SM_EETA = SM_EETA        [test]
			SM_JNUM = SM_JNUM	 [test]
			###########################################
			cut3 = len(ak.flatten(SM_MET))
			
			mask_e_eta = ak.num(SM_EETA[abs(SM_EETA) < 2.5]) == 1
			
			SM_MET = SM_MET[mask_e_eta] 
			SM_Electron = SM_Electron[mask_e_eta] 
			SM_MPHI = SM_MPHI[mask_e_eta] 
			SM_EPHI = SM_EPHI[mask_e_eta] 
			SM_EETA = SM_EETA[mask_e_eta]
			SM_JNUM = SM_JNUM[mask_e_eta]
			
			###########################################
			cut4 = len(ak.flatten(SM_MET))
			
			JNUMBER = ak.num(SM_JNUM > 0)
			
			METMask = ak.flatten(SM_MET) > 50
			SM_MET = SM_MET          [METMask] 
			SM_Electron = SM_Electron[METMask] 
			SM_MPHI = SM_MPHI        [METMask] 
			SM_EPHI = SM_EPHI        [METMask] 
			SM_EETA = SM_EETA        [METMask]
			SM_JNUM = SM_JNUM	 [METMask]
			###########################################
			cut5 = len(ak.flatten(SM_MET))
			
			jetcut = ak.num(SM_JNUM) < 12
			SM_MET = SM_MET          [jetcut] 
			SM_Electron = SM_Electron[jetcut] 
			SM_MPHI = SM_MPHI        [jetcut] 
			SM_EPHI = SM_EPHI        [jetcut] 
			SM_EETA = SM_EETA        [jetcut]
			SM_JNUM = SM_JNUM	 [jetcut]
			###########################################
			cut6 = len(ak.flatten(SM_MET))

			dphi = abs(SM_EPHI[:,0] - SM_MPHI)
			pt_ratio = (SM_Electron[:,0]/SM_MET)	
			mask_dphi = (dphi > 2.5) & (dphi < 3.78)

			np.save("/x6/cykim/condor/OUTPUT/npy/PEPT"   +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_Electron))
			np.save("/x6/cykim/condor/OUTPUT/npy/PEPHI"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_EPHI))
			np.save("/x6/cykim/condor/OUTPUT/npy/PMET"   +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_MET))
			np.save("/x6/cykim/condor/OUTPUT/npy/PEETA"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_EETA))
			np.save("/x6/cykim/condor/OUTPUT/npy/PMPHI"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_MPHI))
			np.save("/x6/cykim/condor/OUTPUT/npy/PDPHI"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(dphi))
			np.save("/x6/cykim/condor/OUTPUT/npy/PRATIO" +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(pt_ratio))
			np.save("/x6/cykim/condor/OUTPUT/npy/PMT"    +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(np.sqrt((1-np.cos(dphi))*2*SM_MET*SM_Electron)))
			
			SM_MET = SM_MET[mask_dphi] 
			SM_Electron = SM_Electron[mask_dphi] 
			SM_MPHI = SM_MPHI[mask_dphi] 
			SM_EPHI = SM_EPHI[mask_dphi] 
			SM_EETA = SM_EETA[mask_dphi]
			###########################################	
			cut7 = len(ak.flatten(SM_MET))
			
			pt_ratio = (SM_Electron[:,0]/SM_MET)
			mask_ptratio = (pt_ratio > 0.4) & (pt_ratio < 1.5)

			SM_MET = SM_MET          [mask_ptratio] 
			SM_Electron = SM_Electron[mask_ptratio] 
			SM_MPHI = SM_MPHI        [mask_ptratio] 
			SM_EPHI = SM_EPHI        [mask_ptratio] 
			SM_EETA = SM_EETA        [mask_ptratio]

			###########################################
			cut8 = len(ak.flatten(SM_MET))
		
			dphi = abs(SM_EPHI[:,0] - SM_MPHI)
			pt_ratio = (SM_Electron[:,0]/SM_MET)	

			print(cut1/cut0*100)
			print(cut2/cut0*100)
			print(cut3/cut0*100)
			print(cut4/cut0*100)
			print(cut5/cut0*100)
			print(cut6/cut0*100)
			print(cut7/cut0*100)
			print(cut8/cut0*100)

			np.save("/x6/cykim/condor/OUTPUT/npy/KEPT"   +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_Electron))
			np.save("/x6/cykim/condor/OUTPUT/npy/KEPHI"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_EPHI))
			np.save("/x6/cykim/condor/OUTPUT/npy/KMET"   +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_MET))
			np.save("/x6/cykim/condor/OUTPUT/npy/KEETA"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_EETA))
			np.save("/x6/cykim/condor/OUTPUT/npy/KMPHI"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(SM_MPHI))
			np.save("/x6/cykim/condor/OUTPUT/npy/KDPHI"  +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(dphi))
			np.save("/x6/cykim/condor/OUTPUT/npy/KRATIO" +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(pt_ratio))
			np.save("/x6/cykim/condor/OUTPUT/npy/KMT"    +str(mass[i])+str(coupling[j])+str(number[k]),ak.to_numpy(np.sqrt((1-np.cos(dphi))*2*SM_MET*SM_Electron)))
