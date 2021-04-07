from simsopt.geo.magneticfieldclasses import ToroidalField, ScalarPotentialRZMagneticField, CircularCoil, Dommaschk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.magneticfield import MagneticFieldSum
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.biotsavart import BiotSavart

import numpy as np
import unittest

class Testing(unittest.TestCase):

    # def test_toroidal_field(self):
    #     R0test    = 1.3
    #     B0test    = 0.8
    #     pointVar  = 1e-2
    #     npoints   = 20
    #     # point locations
    #     points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
    #     points   += pointVar * (np.random.rand(*points.shape)-0.5)
    #     # Bfield from class
    #     Bfield    = ToroidalField(R0test,B0test)
    #     Bfield.set_points(points)
    #     B1        = Bfield.B()
    #     dB1_by_dX = Bfield.dB_by_dX()
    #     # Bfield analytical
    #     B2        = np.array([(B0test*R0test/(point[0]**2+point[1]**2))*np.array([-point[1], point[0], 0.]) for point in points])
    #     dB2_by_dX = np.array([(B0test*R0test/((point[0]**2+point[1]**2)**2))*np.array([[2*point[0]*point[1], point[1]**2-point[0]**2, 0],[point[1]**2-point[0]**2, -2*point[0]*point[1], 0],[0,0,0]]) for point in points])
    #     # Verify
    #     assert np.allclose(B1, B2)
    #     assert np.allclose(dB1_by_dX, dB2_by_dX)
    #     # Verify that divergence is zero
    #     assert (dB1_by_dX[:,0,0]+dB1_by_dX[:,1,1]+dB1_by_dX[:,2,2]==np.zeros((npoints))).all()
    #     assert (dB2_by_dX[:,0,0]+dB2_by_dX[:,1,1]+dB2_by_dX[:,2,2]==np.zeros((npoints))).all()
    #     # Verify that, as a vacuum field, grad B=grad grad phi so that grad_i B_j = grad_j B_i
    #     transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
    #     transpGradB2 = [dBdx.T for dBdx in dB1_by_dX]
    #     assert np.allclose(dB1_by_dX,transpGradB1)
    #     assert np.allclose(dB2_by_dX,transpGradB2)
    #     # Verify values of the vector potential
    #     Afield1   = Bfield.A()
    #     newA1     = np.array([[B0test*R0test*point[0]*point[2]/(point[0]**2+point[1]**2),B0test*R0test*point[1]*point[2]/(point[0]**2+point[1]**2),0] for point in points])
    #     assert np.allclose(Afield1,newA1)
    #     # Verify that curl of magnetic vector potential is the toroidal magnetic field
    #     dA1_by_dX = Bfield.dA_by_dX()
    #     newB1     = np.array([[dA1bydX[1,2]-dA1bydX[2,1],dA1bydX[2,0]-dA1bydX[0,2],dA1bydX[0,1]-dA1bydX[1,0]] for dA1bydX in dA1_by_dX])
    #     assert np.allclose(B1,newB1)
    #     # Verify symmetry of the Hessians
    #     GradGradB1       = Bfield.d2B_by_dXdX()
    #     GradGradA1       = Bfield.d2A_by_dXdX()
    #     transpGradGradB1 = np.array([[gradgradB1.T for gradgradB1 in gradgradB]for gradgradB in GradGradB1])
    #     transpGradGradA1 = np.array([[gradgradA1.T for gradgradA1 in gradgradA]for gradgradA in GradGradA1])
    #     assert np.allclose(GradGradB1,transpGradGradB1)
    #     assert np.allclose(GradGradA1,transpGradGradA1)

    # def test_sum_Bfields(self):
    #     pointVar  = 1e-1
    #     npoints   = 20
    #     points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
    #     points   += pointVar * (np.random.rand(*points.shape)-0.5)
    #     # Set up helical field
    #     coils     = [CurveHelical(101, 2, 5, 2, 1., 0.3) for i in range(2)]
    #     coils[0].set_dofs(np.concatenate(([np.pi/2,0],[0,0])))
    #     coils[1].set_dofs(np.concatenate(([0    ,0],[0,0])))
    #     currents  = [-2.1e5,2.1e5]
    #     Bhelical  = BiotSavart(coils, currents)
    #     # Set up toroidal fields
    #     Btoroidal1 = ToroidalField(1.,1.)
    #     Btoroidal2 = ToroidalField(1.2,0.1)
    #     # Set up sum of the three in two different ways
    #     Btotal1 = MagneticFieldSum([Bhelical,Btoroidal1,Btoroidal2])
    #     Btotal2 = Bhelical+Btoroidal1+Btoroidal2
    #     # Evaluate at a given point
    #     Bhelical.set_points(points)
    #     Btoroidal1.set_points(points)
    #     Btoroidal2.set_points(points)
    #     Btotal1.set_points(points)
    #     Btotal2.set_points(points)
    #     # Verify
    #     assert np.allclose(Btotal1.B(),Btotal2.B())
    #     assert np.allclose(Bhelical.B()+Btoroidal1.B()+Btoroidal2.B(),Btotal1.B())
    #     assert np.allclose(Btotal1.dB_by_dX(),Btotal2.dB_by_dX())
    #     assert np.allclose(Bhelical.dB_by_dX()+Btoroidal1.dB_by_dX()+Btoroidal2.dB_by_dX(),Btotal1.dB_by_dX())

    # def test_scalarpotential_Bfield(self):
    #     # Set up magnetic field scalar potential
    #     PhiStr = "0.1*phi+0.2*R*Z+0.3*Z*phi+0.4*R**2+0.5*Z**2"
    #     # Define set of points
    #     pointVar  = 1e-1
    #     npoints   = 20
    #     points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
    #     points   += pointVar * (np.random.rand(*points.shape)-0.5)
    #     # Set up scalar potential B
    #     Bscalar = ScalarPotentialRZMagneticField(PhiStr)
    #     Bscalar.set_points(points)
    #     B1        = np.array(Bscalar.B())
    #     dB1_by_dX = np.array(Bscalar.dB_by_dX())
    #     # Analytical Formula for B
    #     rphiz     = [[np.sqrt(np.power(point[0],2) + np.power(point[1],2)),np.arctan2(point[1],point[0]),point[2]] for point in points]
    #     B2        = np.array([[0.2*point[2]+0.8*point[0],(0.1+0.3*point[2])/point[0],0.2*point[0]+0.3*point[1]+point[2]] for point in rphiz])
    #     dB2_by_dX = np.array([
    #         [[0.8*np.cos(point[1]),-(np.cos(point[1])/point[0]**2)*(0.1+0.3*point[2]),0.2*np.cos(point[1])-0.3*np.sin(point[1])/point[0]],
    #          [0.8*np.sin(point[1]),-(np.sin(point[1])/point[0]**2)*(0.1+0.3*point[2]),0.2*np.sin(point[1])+0.3*np.cos(point[1])/point[0]],
    #          [0.2, 0.3/point[0], 1]] for point in rphiz])
    #     # Verify
    #     assert np.allclose(B1,B2)
    #     assert np.allclose(dB1_by_dX,dB2_by_dX)

    # def test_circularcoil_Bfield(self):
    #     current = 1.2e7
    #     radius  = 1.12345
    #     center    = [0.12345,0.6789,1.23456]
    #     pointVar  = 1e-1
    #     npoints   = 20
    #     ## verify the field at the center of a coil in the xy plane
    #     Bfield  = CircularCoil(I=current, r0=radius)
    #     points  = np.array([[1e-10,0,0.]])
    #     Bfield.set_points(points)
    #     assert np.allclose(Bfield.B(),[[0,0,current/1e7*2*np.pi/radius]])
    #     # Verify that divergence is zero
    #     dB1_by_dX=Bfield.dB_by_dX()
    #     assert np.allclose(dB1_by_dX[:,0,0]+dB1_by_dX[:,1,1]+dB1_by_dX[:,2,2],np.zeros((npoints)))
    #     # Verify that, as a vacuum field, grad B=grad grad phi so that grad_i B_j = grad_j B_i
    #     transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
    #     assert np.allclose(dB1_by_dX,transpGradB1)
    #     ### compare to biosavart(circular_coil)
    #     ## at these points
    #     points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
    #     points   += pointVar * (np.random.rand(*points.shape)-0.5)
    #     ## verify with a x^2+z^2=radius^2 circular coil
    #     normal    = [np.pi/2,0]
    #     coils     = [CurveXYZFourier(300, 1)]
    #     coils[0].set_dofs([center[0],radius,0.,center[1],0.,0.,center[2],0.,radius])
    #     Bcircular = BiotSavart(coils, [current])
    #     Bfield    = CircularCoil(I=current, r0=radius, normal=normal, center=center)
    #     Bfield.set_points(points)
    #     Bcircular.set_points(points)
    #     dB1_by_dX=Bfield.dB_by_dX()
    #     transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
    #     assert np.allclose(Bfield.B(),Bcircular.B())
    #     assert np.allclose(Bfield.dB_by_dX(),Bcircular.dB_by_dX())
    #     assert np.allclose(dB1_by_dX[:,0,0]+dB1_by_dX[:,1,1]+dB1_by_dX[:,2,2],np.zeros((npoints)))
    #     assert np.allclose(dB1_by_dX,transpGradB1)
    #     ## verify with a y^2+z^2=radius^2 circular coil
    #     normal    = [np.pi/2,-np.pi/2]
    #     coils     = [CurveXYZFourier(300, 1)]
    #     coils[0].set_dofs([center[0],0,0.,center[1],radius,0.,center[2],0.,radius])
    #     Bcircular = BiotSavart(coils, [current])
    #     Bfield    = CircularCoil(I=current, r0=radius, normal=normal, center=center)
    #     Bfield.set_points(points)
    #     Bcircular.set_points(points)
    #     dB1_by_dX=Bfield.dB_by_dX()
    #     transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
    #     assert np.allclose(Bfield.B(),Bcircular.B())
    #     assert np.allclose(Bfield.dB_by_dX(),Bcircular.dB_by_dX())
    #     assert np.allclose(dB1_by_dX[:,0,0]+dB1_by_dX[:,1,1]+dB1_by_dX[:,2,2],np.zeros((npoints))) # divergence
    #     assert np.allclose(dB1_by_dX,transpGradB1) # symmetry of the gradient
    #     ## verify with a x^2+y^2=radius^2 circular coil
    #     center     = [0,0,0]
    #     normal     = [0,0]
    #     coils      = [CurveXYZFourier(300, 1)]
    #     coils[0].set_dofs([center[0],0,radius,center[1],radius,0.,center[2],0.,0.])
    #     Bcircular  = BiotSavart(coils, [current])
    #     coils2     = [CurveRZFourier(300, 1, 1, True)]
    #     coils2[0].set_dofs([radius,0,0])
    #     Bcircular2 = BiotSavart(coils, [current])
    #     Bfield     = CircularCoil(I=current, r0=radius, normal=normal, center=center)
    #     Bfield.set_points(points)
    #     Bcircular.set_points(points)
    #     Bcircular2.set_points(points)
    #     dB1_by_dX=Bfield.dB_by_dX()
    #     transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
    #     assert np.allclose(Bfield.B(),Bcircular.B())
    #     assert np.allclose(Bfield.B(),Bcircular2.B())
    #     assert np.allclose(Bfield.dB_by_dX(),Bcircular.dB_by_dX())
    #     assert np.allclose(Bfield.dB_by_dX(),Bcircular2.dB_by_dX())
    #     assert np.allclose(dB1_by_dX[:,0,0]+dB1_by_dX[:,1,1]+dB1_by_dX[:,2,2],np.zeros((npoints))) # divergence
    #     assert np.allclose(dB1_by_dX,transpGradB1) # symmetry of the gradient

    # def test_helicalcoil_Bfield(self):
    #     point = [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]]
    #     field = [[-0.00101961,0.20767292,-0.00224908]]
    #     derivative = [[[0.47545098,0.01847397,1.10223595],[0.01847426,-2.66700072,0.01849548],[1.10237535,0.01847085,2.19154973]]]
    #     coils     = [CurveHelical(100, 2, 5, 2, 1., 0.3) for i in range(2)]
    #     coils[0].set_dofs(np.concatenate(([0,0],[0,0])))
    #     coils[1].set_dofs(np.concatenate(([np.pi/2,0],[0,0])))
    #     currents  = [-3.07e5,3.07e5]
    #     Bhelical  = BiotSavart(coils, currents)
    #     Bhelical.set_points(point)
    #     assert np.allclose(Bhelical.B(),field)
    #     assert np.allclose(Bhelical.dB_by_dX(),derivative)

    def test_Dommaschk(self):
        a       = np.zeros((16,4))
        b       = np.zeros((16,4))
        c       = np.zeros((16,4))
        d       = np.zeros((16,4))
        b[10,2] =-2.18
        c[10,2] =-2.18
        a[15,3] = 25.8
        d[15,3] =-25.8
        Bfield  = Dommaschk(a=a, b=b, c=c, d=d)
        # verify CD and CN
        assert  np.allclose(Bfield.CD(1.12341,0,0),1)
        assert  np.allclose(Bfield.CN(1.12341,0,0),np.log(1.12341))
        test1   = np.sum([[Bfield.CD(0.9923456,m,k) for m in range(6)] for k in range(6)])
        test2   = np.sum([[Bfield.CN(0.9923456,m,k) for m in range(6)] for k in range(6)])
        assert  np.allclose(test1,6.03205)
        assert  np.allclose(test2,-0.0463384)
        # verify ImnD and ImnN
        assert  np.allclose(Bfield.ImnD(0,0,1.12341,0.2),1)
        assert  np.allclose(Bfield.ImnN(0,0,1.12341,0.2),np.log(1.12341))
        test3   = np.sum([[Bfield.ImnD(m,n,0.9923456,0.231231) for m in range(6)] for n in range(6)])
        test4   = np.sum([[Bfield.ImnN(m,n,0.9923456,0.231231) for m in range(6)] for n in range(6)])
        assert  np.allclose(test3,7.6012)
        assert  np.allclose(test4,-0.0583926)
        # verify Vml
        assert  np.allclose(Bfield.Vml(10,2,0.9923456,0.83216793,0.231231),-0.0538655)
        assert  np.allclose(Bfield.Vml(15,3,0.9923456,0.83216793,0.231231),0.0527084)
        # verify V
        assert  np.allclose(Bfield.V(0.9923456,0.83216793,0.231231),0.831011)
        # verify B
        # calling the function the first time takes a compilation time of around 30 seconds
        Bfield.set_points([[0.9231, 0.8423, -0.1123]])
        assert  np.allclose(Bfield.B(),[[-1.72696, 3.26173, -2.22013]])
        # verify gradB
        # calling the function the first time takes a compilation time of around 2 minutes
        # gradB       = np.array(Bfield.dB_by_dX())
        # transpGradB = np.array([dBdx.T for dBdx in gradB])
        # assert np.allclose(gradB,np.array([[-59.9602, 8.96793, -24.8844],[8.96793, 49.0327, -18.4131],[-24.8844, -18.4131, 10.9275]]))
        # assert np.allclose(gradB,transpGradB)
        # verify gradgradB
        # calling the function the first time takes a compilation time of around (many) minutes
        # GradGradB1   = np.array(Bfield.d2B_by_dXdX())
        # transpGradGradB1 = np.array([[gradgradB1.T for gradgradB1 in gradgradB]for gradgradB in GradGradB1])
        # assert np.allclose(GradGradB1,transpGradGradB1)
        # assert np.allclose(GradGradB1,[[[-930.936, -396.415, 129.123],[-396.415, 430.684, -375.609],[129.123, -375.609, 500.252]],[[-396.415,430.684,-375.609],[430.684,691.762,-165.373],[-375.609,-165.373,-295.348]],[[129.123,-375.609,500.252],[-375.609,-165.373,-295.348],[500.252,-295.348,36.2501]]])

if __name__ == "__main__":
    unittest.main()