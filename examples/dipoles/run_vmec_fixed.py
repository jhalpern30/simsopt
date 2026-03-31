from simsopt.mhd.vmec import Vmec
from simsopt._core import load
from simsopt.geo.surfaceobjectives import ToroidalFlux

path = '../single_stage_scans_no_sparsity_epsilon_constraint/wout_nfp22ginsburg_000_000281_init_dir90/iota_tar0.15/stage02_cw3/mpol6_ntor6/'
surf = load(path + 'surf_opt.json')
bs = load(path + 'bs_opt.json')
tf = ToroidalFlux(surf,bs)

vmec = Vmec('input.fixed')
vmec.boundary = surf
vmec.indata.phiedge = tf.J()
vmec.indata.mpol = surf.mpol
vmec.indata.ntor = surf.ntor
vmec.indata.nfp = surf.nfp
vmec.run()