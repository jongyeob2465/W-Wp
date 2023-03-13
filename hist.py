import numpy as np
import matplotlib.pyplot as plt

# x range setting
xmin = -6000
xmax = 6000
xbin = 100

# normalization information (W boson cross section, luminosity, event number)
lumi = 13800 # fb^-1
cross_section = np.array([9682.1, 0.212977, 0.0051716, 0.000135443,9.1129e-06])
event = np.array([200000,20000,20000,20000,1000000])

mytext = ['~200','200~500','500~1000','1000~1500','1500~inf']

for i in range(len(cross_section)):

	# gen level electron pt array
	globals()['a%s'%(i+1)] = np.load('/x6/cykim/condor/work/resolution/python_file/gen_electron.pt_6000_100_%s.npy'%(i+1)).flatten()

	# reco electron pt array
	globals()['A%s'%(i+1)] = np.load('/x6/cykim/condor/work/resolution/python_file/dec_electron.pt_6000_100_%s.npy'%(i+1)).flatten()

	# pt mean
	#globals()['a_mean%s'%(i+1)] = np.mean(globals()['a%s'%(i+1)])
	#globals()['A_mean%s'%(i+1)] = np.mean(globals()['A%s'%(i+1)])

	# pt std
	#globals()['a_std%s'%(i+1)] = np.std(globals()['A%s'%(i+1)])
	#globals()['A_std%s'%(i+1)] = np.std(globals()['a%s'%(i+1)])

	# (gen pt) - (reco pt)
	globals()['error%s'%(i+1)] = (globals()['a%s'%(i+1)] - globals()['A%s'%(i+1)])

	# mean [(gen pt) - (reco pt)]
	#globals()['error_mean%s'%(i+1)] = np.mean(globals()['error%s'%(i+1)])

	plt.hist(globals()['error%s'%(i+1)],bins=np.arange(xmin,xmax+xbin,xbin),label=mytext[i])
	plt.yscale('log')
	plt.legend()
	plt.xlabel('$p^{gen}_{T}$ - $p^{reco}_{T}$ ',fontsize = 18)
	plt.ylabel('Events', fontsize = 18)
	plt.ylim(1,1000000)
	plt.xlim(xmin,xmax)
	plt.savefig('pt_resolution_%s'%(mytext[i]))
	plt.clf()
