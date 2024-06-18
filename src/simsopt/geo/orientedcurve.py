import jax.numpy as jnp
from math import pi, sin, cos
import numpy as np
from .curve import JaxCurve
from simsopt._core.optimizable import Optimizable

__all__ = ['OrientedCurveXYZFourier', 'OrientedCurveRTPFourier','OrientedCurveCylindricalFourier']

def shift_pure( v, xyz ):
    for ii in range(0,3):
        v = v.at[:,ii].add(xyz[ii])
    return v

#Shifts a set of vectors by a specified amount in 3D space
def rotate_pure( v, ypr ):        
    yaw = ypr[0]
    pitch = ypr[1]
    roll = ypr[2]

    Myaw = jnp.asarray(
        [[jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]]
    )
    Mpitch = jnp.asarray(
        [[jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]]
    )
    Mroll = jnp.asarray(
        [[1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]]
    )

    return v @ Myaw @ Mpitch @ Mroll

# Rohan's code
def centercurve_pure(dofs, quadpoints, order, rotation_center = None):
    xyz = dofs[0:3]
    ypr = dofs[3:6]
    fmn = dofs[6:]
    print("Rotation center:", rotation_center)
    
    k = len(fmn)//3
    coeffs = [fmn[:k], fmn[k:(2*k)], fmn[(2*k):]]
    #coeffs is a list of a threee arrays containing coefficients of each dimension for x,y,z, used to scale cos and sin terms
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(0,3):
    #iterates over xyz coordinates
        for j in range(0, order):
            gamma = gamma.at[:, i].add(coeffs[i][2 * j    ] * jnp.sin(2 * pi * (j+1) * points))
            gamma = gamma.at[:, i].add(coeffs[i][2 * j + 1] * jnp.cos(2 * pi * (j+1) * points))
            #updates gamma for each dimension in the fourier series
    shifted_gamma = shift_pure(gamma, -rotation_center)
    # Apply the rotation around the origin
    rotated_gamma = rotate_pure(shifted_gamma, ypr)
    # Now, shift all points except the rotation center back to their original position
    final_gamma = shift_pure(rotated_gamma, rotation_center +xyz)
    return final_gamma
    #gives a point as input which can be gotten through the gamma function, and then the output should be the curve rotated around
    #this point and not zero
    #this is a rotate then shift
    #in order to position all the curves at zero, may need to shift, rotate, then shift
    #need to modify the class so that one can rotate around any point, then constraint so that the point can be set at any point in
    #cartesian space
    #have to define how to rotate around the first gamma point, then make a penalty function where it has to curl around that point
    

class OrientedCurveXYZFourier( JaxCurve ):
    """
    OrientedCurveXYZFourier is a translated and rotated
    JaxCurveXYZFourier Curve.
    """
    def __init__(self, quadpoints, order, dofs=None, rotation_center = np.zeros(3)):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        self.rotation_center=rotation_center
        self.order = order
        pure = lambda dofs, points: centercurve_pure(dofs, points, self.order, rotation_center)
        #we go from dof to real space coordinate here, by modifying centercurvepure.
        #need to to provide an addition attribute that the user can set which is the point, which needs to be passed to the
        #centercurve pure function which also being passed in as gamma
        self.coefficients = [np.zeros((3,)), np.zeros((3,)), np.zeros((2*order,)), np.zeros((2*order,)), np.zeros((2*order,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=OrientedCurveXYZFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=OrientedCurveXYZFourier.set_dofs_impl,
                             names=self._make_names())
            self.update_rotation_center(self.rotation_center)
    #def update_rotation_center(self, x0):
        #"""Updates the rotation center based on the current curve"""
        #self.rotation_center = x0
        #curve_points = self.gamma()
        #if len(curve_points) > 0:
            #self.rotation_center = curve_points[0]
        #else:
        #    raise ValueError("Curve points are empty, cannot set rotation center")
#I think the issue may be that this isn't being assigned to the class, so that the first value is always chosen as
#the point of rotation
    
    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3 + 3*(2*self.order)
   
    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)
    
    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:3]
        self.coefficients[1][:] = dofs[3:6]
        counter = 6
        for i in range(0,3):
            for j in range(0, self.order):
                self.coefficients[i+2][2*j] = dofs[counter]
                counter += 1
                self.coefficients[i+2][2*j+1] = dofs[counter]
                counter += 1
    
    def update_rotation_center(self,rotation_center):
        new_rotation_center = jnp.array(new_rotation_center)
        gamma_points = self.gamma()
        lowest_z = jnp.min(gamma_points[:, 2])
        lowest_point = gamma_points[jnp.argmin(gamma_points[:, 2])]
        shift_needed = new_rotation_center - lowest_point[:3]  # Focus on x, y, z coordinates
        current_dofs = self.get_dofs()
        current_dofs[:3] -= shift_needed  # Adjust this logic based on actual DoFs' role and representation
        new_dofs = jnp.concatenate([translation_dofs, current_dofs[3:]])
        self.set_dofs(new_dofs)
        self.rotation_center = new_rotation_center
    
    def _make_names(self):
        xyc_name = ['x', 'y', 'z']
        ypr_name = ['yaw', 'pitch', 'roll']
        dofs_name = []
        for c in ['x', 'y', 'z']:
            for j in range(0, self.order):
                dofs_name += [f'{c}s({j+1})', f'{c}c({j+1})']
        return xyc_name + ypr_name + dofs_name
    @classmethod
    def convert_xyz_to_oriented(cls, curve_xyz_fourier, quadpoints=None,rotation_center=None):
        """
        Converts a CurveXYZFourier object to an OrientedCurveXYZFourier object.
        Parameters:
        curve_xyz_fourier: Instance of CurveXYZFourier
        quadpoints: Quadrature points for the curve
        order: Fourier series order
        Returns:
        oriented_curve: Instance of OrientedCurveXYZFourier
        """

        if quadpoints is None:
            quadpoints = curve_xyz_fourier.quadpoints

        fixed_point = curve_xyz_fourier.gamma()[0]

        curve_dofs = curve_xyz_fourier.get_dofs()
        o = curve_xyz_fourier.order
        translation = curve_dofs[[0, 2*o+1,2*(2*o+1)]]
        curve_dofs = np.delete(curve_dofs,[0, 2*o+1,2*(2*o+1)])

        # Initialize translation and rotation to zero

        rotation = np.zeros(3)

        # Combine the translation, rotation, and curve DOFs
        oriented_dofs = np.concatenate([translation, rotation, curve_dofs])

        oriented_curve = cls(quadpoints, curve_xyz_fourier.order, rotation_center=rotation_center)

        oriented_curve.set_dofs(oriented_dofs)
        #have to loop through all the dofs of oriented curve and give them the value i want them to hold, which has a
        #1to1 correspondence with xyzfourier dof

        #c.set('xo',...)

        return oriented_curve








def xyz_to_rtpz(gamma, R0):
    gamma_rtpz = jnp.empty(gamma.shape)
    # calculate r from x and y

    #check this operation and ensure it actually works
    gamma_rtpz = gamma_rtpz.at[:,0].set(jnp.sqrt(gamma[:,0]**2 + gamma[:,1]**2) - R0)

    # calculate theta
   #gamma_rtpz = gamma_rtpz.at[:,1].set(jnp.arctan2(gamma[:,2], jnp.sqrt(gamma[:,0]**2 + gamma[:,1]**2) - R0))
    # calculate phi
    gamma_rtpz = gamma_rtpz.at[:,1].set(jnp.arctan2(gamma[:,1],gamma[:,0]))
    # z coordinate remains the same
    gamma_rtpz = gamma_rtpz.at[:,2].set(gamma[:,2])

    return gamma_rtpz


def rtpz_to_xyz(gamma, R0):
    gamma_xyz = jnp.empty(gamma.shape)
    # calculate x from r, theta, phi
    #print("Shape of gamma_xyz:", gamma_xyz.shape)
    #print("Shape of R0:", R0.shape if hasattr(R0, 'shape') else 'Scalar')
    #print("Shape of gamma[:,0]:", gamma[:,0].shape)
    #print("Shape of jnp.cos(gamma[:,2]):", jnp.cos(gamma[:,2]).shape)

    gamma_xyz = gamma_xyz.at[:,0].set((R0 + gamma[:,0] * jnp.cos(gamma[:,1]))*jnp.cos(gamma[:,2]))
    # calculate y
    gamma_xyz = gamma_xyz.at[:,1].set((R0 + gamma[:,0] * jnp.cos(gamma[:,1])) * jnp.sin(gamma[:,2]))   
     # calculate z
    gamma_xyz = gamma_xyz.at[:,2].set(gamma[:,2] *jnp.sin(gamma[:,1]))

    return gamma_xyz

def r_shift(gamma, r_shift):
    gamma = gamma.at[:, 0].add(r_shift)
    return gamma

def theta_shift(gamma, theta_shift):
    gamma = gamma.at[:, 1].add(theta_shift)
    return gamma

def phi_shift(gamma, phi_shift):
    gamma = gamma.at[:, 2].add(phi_shift)
    return gamma


def z_shift(gamma, z_shift):
    gamma = gamma.at[:, 3].add(z_shift)
    return gamma

def rtpz_centercurve_pure(dofs, quadpoints, order, rotation_center=None, Rcenter=0):
    R0aZ0 = dofs[0:3]
    tp = dofs[3:6]
    rotation_center = dofs[6:9]
    fmn = dofs[9:]
    k = len(fmn)//3
    coeffs = [fmn[:k], fmn[k:(2*k)], fmn[(2*k):]]
    #coeffs is a list of a threee arrays containing coefficients of each dimension for x,y,z, used to scale cos and sin terms
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    print("points:",points.shape)
    print("gamma:",1 )

    for i in range(0,3):
    #iterates over xyz coordinates
        for j in range(0, order):
            gamma = gamma.at[:, i].add(coeffs[i][2 * j ] * jnp.sin(2 * pi * (j+1) * points))
            gamma = gamma.at[:, i].add(coeffs[i][2 * j + 1] * jnp.cos(2 * pi * (j+1) * points))

    gamma= xyz_to_rtpz(gamma, Rcenter)
    print("Gamma from xyz_to:", gamma)
    print(2)

    # shifted_gamma = shift_pure(gamma, -rotation_center)
    # rotated_gamma = rotate_pure(shifted_gamma, tp)
    # final_gamma = shift_pure(rotated_gamma, rotation_center + R0aZ0[:3])

    return rtpz_to_xyz(gamma, R0aZ0[0])

class OrientedCurveCylindricalFourier(JaxCurve):
    """
    OrientedCurveCylindricalFourier is a translated and rotated Curve in r, theta, phi, and z coordinates.
    """
    def __init__(self, quadpoints, order, dofs=None, rotation_center=np.zeros(3)):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        self.rotation_center = rotation_center
        self.order = order
        #self.Rcenter=1
        pure = lambda dofs, points: rtpz_centercurve_pure(dofs, points, self.order, self.rotation_center)#, self.Rcenter)

        self.coefficients = [np.zeros((3,)), np.zeros((3,)),rotation_center, np.zeros((2*order,)), np.zeros((2*order,)), np.zeros((2*order,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=OrientedCurveCylindricalFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=OrientedCurveCylindricalFourier.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3+ 3 + 3 * (2 * self.order)

    def get_dofs(self):
        return np.concatenate([self.coefficients[0], self.coefficients[1], self.coefficients[2], *self.coefficients[3:]])


    def set_dofs_impl(self, dofs):
        """
        This function sets the degrees of freedom (DoFs) for this object.
         """
        self.coefficients[0][:] = dofs[0:3]  # R0, phi, Z0
        self.coefficients[1][:] = dofs[3:6]  # theta, constant_phi,
        self.coefficients[2][:] = dofs[6:9] 

        self.rotation_center = dofs[6:9]
        counter = 9

        for i in range(0, 3):  # Adjusted for the additional z component
            for j in range(0, self.order):
                self.coefficients[i+3][2*j] = dofs[counter]
                counter += 1
                self.coefficients[i+3][2*j+1] = dofs[counter]
                counter += 1


    def update_rotation_center(self, new_rotation_center):
        """
        Updates the rotation center based on the given new_rotation_center.
        """
        new_rotation_center = np.array(new_rotation_center)
        gamma_points = self.gamma()

        # Find the point with the lowest z-coordinate
        lowest_z = np.min(gamma_points[:, 2])
        lowest_point = gamma_points[np.argmin(gamma_points[:, 2])]

        # Calculate the shift needed to move the lowest point to the new rotation center
        shift_needed = new_rotation_center - lowest_point[:3]

        # Update the DoFs to apply the shift
        current_dofs = self.get_dofs()
        current_dofs[:3] -= shift_needed
        self.set_dofs(current_dofs)

        # Update the rotation center attribute
        self.rotation_center = new_rotation_center


    def _make_names(self):
        """
        Generates names for the degrees of freedom (DoFs) for this object.
        """

        rtpz_name = ['R0', 'phi', 'Z0']
        angle_name = ['r_rotation', 'phi_rotation', 'z_rotation']
        rotation_center_name = ['rx', 'ry', 'rz']

        dofs_name = []

        #
        for c in ['x', 'y', 'z']:
            for j in range(0, self.order):
                dofs_name += [f'{c}s({j+1})', f'{c}c({j+1})']
        print(dofs_name)
        return rtpz_name + angle_name + rotation_center_name + dofs_name

    @classmethod
    def convert_xyz_to_oriented(cls, curve_xyz_fourier, quadpoints=None, rotation_center=None):
        """
        Converts a CurveXYZFourier object to an OrientedCurveCylindricalFourier object.
        Parameters:
        curve_xyz_fourier: Instance of CurveXYZFourier
        quadpoints: Quadrature points for the curve
        rotation_center: Center of rotation in cylindrical coordinates (r, phi, z)
        Returns:
        oriented_curve: Instance of OrientedCurveCylindricalFourier
        """

        if quadpoints is None:
            quadpoints = curve_xyz_fourier.quadpoints

        fixed_point = curve_xyz_fourier.gamma()[0]


        curve_dofs = curve_xyz_fourier.get_dofs()
        o = curve_xyz_fourier.order

        translation_xyz = curve_dofs[[0, 2*o+1,2*(2*o+1)]]

        R0 = np.sqrt(translation_xyz[0]**2+translation_xyz[1]**2)
        phi = np.arctan2(translation_xyz[1],translation_xyz[0])
        Z = translation_xyz[2]
        curve_dofs = np.delete(curve_dofs,[0, 2*o+1,2*(2*o+1)])


        translation_cylindrical= [R0,phi,Z]

        rotation = np.zeros(3)  # Cylindrical has 2 rotation DoFs (phi, z)
        rotation_center= np.zeros(3)

        # Combine the translation, rotation, and curve DoFs
        oriented_dofs = np.concatenate([translation_cylindrical, rotation,rotation_center, curve_dofs])        

        oriented_curve = cls(quadpoints, curve_xyz_fourier.order, rotation_center=rotation_center)

        oriented_curve.set_dofs(oriented_dofs)

        return oriented_curve

#end of Rohan's code

def xyz_to_rtp(gamma,R0):
    gamma_rtp = jnp.empty(gamma.shape)
    # calculate r from x, y, z, and R0
    gamma_rtp = gamma_rtp.at[:,0].set(jnp.sqrt(gamma[:,2]**2 + (jnp.sqrt(gamma[:,0]**2 + gamma[:,1]**2) - R0)**2) )
    # calculate theta
    gamma_rtp = gamma_rtp.at[:,1].set(jnp.arctan2(gamma[:,2], jnp.sqrt(gamma[:,0]**2 + gamma[:,1]**2) - R0))
    # calculate phi
    gamma_rtp = gamma_rtp.at[:,2].set(jnp.arctan2(gamma[:,1],gamma[:,0]))

    return gamma_rtp

def rtp_to_xyz(gamma, R0):
    gamma_xyz = jnp.empty(gamma.shape)
    # calculate x from r, theta, phi
    gamma_xyz = gamma_xyz.at[:,0].set((R0 + gamma[:,0] * jnp.cos(gamma[:,1])) * jnp.cos(gamma[:,2]))
    # calculate y
    gamma_xyz = gamma_xyz.at[:,1].set((R0 + gamma[:,0] * jnp.cos(gamma[:,1])) * jnp.sin(gamma[:,2]))
    # calculate z
    gamma_xyz = gamma_xyz.at[:,2].set(gamma[:,0] * jnp.sin(gamma[:,1]))

    return gamma_xyz

def R0_shift(gamma, R0):
    gamma = gamma.at[:,0].add(R0)
    return gamma
    
def theta_shift(gamma, theta):
    gamma = gamma.at[:,1].add(theta)
    return gamma

def phi_shift(gamma, phi):
    gamma = gamma.at[:,2].add(phi)
    return gamma
    
def rtp_shift_and_rotate_pure( v, R0aZ0, tp):
    R0 = R0aZ0[0]
    a = R0aZ0[1]
    Z0 = R0aZ0[2] # not currently built in
    theta = tp[0]
    phi = tp[1]

    return rtp_to_xyz(theta_shift(phi_shift(xyz_to_rtp(R0_shift(v,R0+a),R0), phi), theta), R0)

def rtp_centercurve_pure(dofs, quadpoints, order):
    R0aZ0 = dofs[0:3]
    tp = dofs[3:5]
    fmn = dofs[5:]

    k = len(fmn)//3
    coeffs = [fmn[:k], fmn[k:(2*k)], fmn[(2*k):]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(0,3):
        for j in range(0, order):
            gamma = gamma.at[:, i].add(coeffs[i][2 * j    ] * jnp.sin(2 * pi * (j+1) * points))
            gamma = gamma.at[:, i].add(coeffs[i][2 * j + 1] * jnp.cos(2 * pi * (j+1) * points))

    return rtp_shift_and_rotate_pure( gamma, R0aZ0 , tp )
    
    
class OrientedCurveRTPFourier( JaxCurve ):
    """
    OrientedCurveRTPFourier is a translated and rotated Curve in r, theta, phi coordinates.
    """
    def __init__(self, quadpoints, order, dofs=None ):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        self.order = order
        pure = lambda dofs, points: rtp_centercurve_pure(dofs, points, self.order)

        self.coefficients = [np.zeros((3,)), np.zeros((2,)), np.zeros((2*order,)), np.zeros((2*order,)), np.zeros((2*order,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=OrientedCurveRTPFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             external_dof_setter=OrientedCurveRTPFourier.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 2 + 3*(2*self.order)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)
    
    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:3]
        self.coefficients[1][:] = dofs[3:5]

        counter = 5
        for i in range(0,3):
            for j in range(0, self.order):
                self.coefficients[i+2][2*j] = dofs[counter]
                counter += 1
                self.coefficients[i+2][2*j+1] = dofs[counter]
                counter += 1

        

    def _make_names(self):
        xyc_name = ['R0', 'a', 'Z0']
        ypr_name = ['theta', 'phi']
        dofs_name = []
        for c in ['x', 'y', 'z']:
            for j in range(0, self.order):
                dofs_name += [f'{c}s({j+1})', f'{c}c({j+1})']
        return xyc_name + ypr_name + dofs_name



