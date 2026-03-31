import booz_xform as bx
import matplotlib.pyplot as plt
import numpy as np
from simsopt.mhd.vmec import Vmec 

b = bx.Booz_xform()
b.read_boozmn('boozmn_fixed_000_000000.nc')

bmnc = b.bmnc_b
xm = b.xm_b
xn = b.xn_b
f = np.sqrt(np.sum(bmnc[xn!=0,:]**2,axis=0)/np.sum(bmnc**2,axis=0))
print('Mean QA metric: ',np.mean(f))
print('Mean iota: ',np.mean(b.iota))
s = b.s_b

plt.figure()
plt.plot(s,f)
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('f [quasisymmetry metric]')
plt.tight_layout()
plt.savefig('fqs_plot.png')

plt.figure()
plt.plot(s,b.iota)
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('iota')
plt.tight_layout()
plt.savefig('iota.png')

plt.figure()
bx.surfplot(b, js=-1)
plt.tight_layout()
plt.savefig('modB_plot.png')

vmec = Vmec('wout_fixed_000_000000.nc');
plt.figure()
plt.plot(vmec.s_half_grid,vmec.wout.vp[1::])
plt.xlabel('s [normalized toroidal flux]')
plt.ylabel('Vp [radial derivative of volume]')
plt.tight_layout()
plt.savefig('magwell.png')
plt.show()

print('Volume: ', vmec.wout.volume_p)
print('Aspect ratio: ',vmec.wout.aspect)
print('Minor radius: ',vmec.wout.Aminor_p)
print('Major radius: ',vmec.wout.Rmajor_p)
print('Averaged field strength: ',vmec.wout.volavgB)