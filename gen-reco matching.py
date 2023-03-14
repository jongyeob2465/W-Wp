import numpy as np
import uproot as up
import awkward as ak
import matplotlib.pyplot as plt

def error(gen,dec): # 오차 define
	return abs(gen-dec)/gen*100

# variable = reconstruction을 위해서 가져올 field 종류, variable2 = gen level 에서 가져올 field 종류
variable = ['Electron.PT', 'Electron.Eta', 'Electron.Phi', 'MissingET.MET', 'MissingET.Phi', 'Jet.PT','Particle.PID','Particle.PT','Particle.Phi']

mask = {
	# dictionary 형태로 mask를 정의

	# pre selection

	'Electron_Pt_first':  lambda data: ak.num(data['Electron.PT'] [data['Electron.PT'] > 25]) == 1, 
	'Electron_Pt_second': lambda data: ak.num(data['Electron.PT'] [data['Electron.PT'] > 50]) == 1,
	'Electron_Eta':       lambda data: ak.num(data['Electron.Eta'][abs(data['Electron.Eta']) < 2.5]) == 1, # 추후 수정할 예정 
	'MET':                lambda data: ak.flatten(data['MissingET.MET']) > 50,
	'Jet_number':         lambda data: ak.num(data['Jet.PT']) < 12,

	# kinematic selection
	'dphi':               lambda data: ak.flatten((np.abs(data['Electron.Phi'][:,0] - data['MissingET.Phi']) > 2.5) & (np.abs(data['Electron.Phi'][:,0]-data['MissingET.Phi']) < 3.78)),
	'PT_ratio':           lambda data: ak.flatten(((data['Electron.PT'][:,0]/data['MissingET.MET']) > 0.4) & ((data['Electron.PT'][:,0]/data['MissingET.MET']) < 1.5))
}

for i in [6000]:              # W' 의 질량에 대한 loop
	for j in [100]:           # W' 의 gL/gR 에 대한 loop (100 -> gL/gW 1.0)
		for k in [1,2,3,4,5]: # 파일 번호에 대한 loop, 1~5까지 총 5개의 파일이 있음. plt cut 기준

			# load root file
			globals()['dec_%s_%s_%s'%(i,j,k)] = up.lazy("/x6/cykim/condor/WP/Delphes_cms/condorDelpyOut/Delphes_%s_%s_%s"%(i,j,k) + ".root",branches=variable)
			globals()['gen_%s_%s_%s'%(i,j,k)] = up.lazy("/x6/cykim/condor/WP/KL_root/KL%s_%s_%s"%(i,j,k) + ".root",branches=variable2)

			print("open root file, mass : %s, coupling : %s, number : %s"%(i,j,k))

			# apply mask
			for m in mask.keys(): # mask를 gen level, detector level에 모두 적용. 모든 컷은 event 에 대해서 적용되므로, detector에 대한 cut을 gen level에 적용 가능.
				if m == 'Jet_number': # 해당 mask를 적용하기 직전에 jet number 저장
					np.save('jetnumber_%s%s%s'%(i,j,k),ak.to_numpy(ak.num(globals()['dec_%s_%s_%s'%(i,j,k)]['Jet.PT'])))
				globals()['gen_%s_%s_%s'%(i,j,k)] =  globals()['gen_%s_%s_%s'%(i,j,k)][mask[m](globals()['dec_%s_%s_%s'%(i,j,k)])] 
				globals()['dec_%s_%s_%s'%(i,j,k)] =  globals()['dec_%s_%s_%s'%(i,j,k)][mask[m](globals()['dec_%s_%s_%s'%(i,j,k)])]

			# reconstruct mass
			globals()['wmt_file_%s_%s_%s'%(i,j,k)] = np.sqrt(2*globals()['dec_%s_%s_%s'%(i,j,k)]['Electron.PT']*globals()['dec_%s_%s_%s'%(i,j,k)]['MissingET.MET']*(1 - np.cos((globals()['dec_%s_%s_%s'%(i,j,k)]['Electron.Phi']-globals()['dec_%s_%s_%s'%(i,j,k)]['MissingET.Phi']))))
