import booz_xform as bx

mboz = 48 # number of poloidal harmonics for Boozer transformation
nboz = 48 # number of toroidal harmonics for Boozer transformation

b = bx.Booz_xform()
wout_filename = 'wout_fixed_000_000000.nc'
b.read_wout(wout_filename)
b.mboz = mboz
b.nboz = nboz
b.run()
b.write_boozmn('boozmn_fixed_000_000000.nc')