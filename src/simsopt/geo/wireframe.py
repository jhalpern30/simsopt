"""
wireframe.py

Definitions for the ToroidalWireframe class
"""

import numpy as np
import collections
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

__all__ = ['ToroidalWireframe']

class ToroidalWireframe(object):
    """
    ``ToroidalWireframe`` is a wireframe grid whose nodes are placed on a
    toroidal surface as a 2D grid with regular spacing in the poloidal and
    toroidal dimensions.

    Currently only supports surfaces that exhibit stellarator symmetry.

    Parameters
    ----------
        surface: SurfaceRZFourier class instance
            Toroidal surface on which on which the nodes will be placed.
        nPhi: integer
            Number of wireframe nodes per half-period in the toroidal dimension.
            Must be even; if an odd number is provided, it will be incremented
            by one.
        nTheta: integer
            Number of wireframe nodes in the poloidal dimension. Must be even;
            if an odd number is provided, it will be incremented by one.
    """

    def __init__(self, surface, nPhi, nTheta):

        if not isinstance(surface, SurfaceRZFourier):
            raise ValueError('Surface must be a SurfaceRZFourier object')

        if not surface.stellsym:
            raise ValueError('Surfaces without stellarator symmetry are not ' \
                             + 'currently supported in the ToroidalWireframe ' \
                             + 'class')

        if not isinstance(nTheta, int) or not isinstance(nPhi, int):
            raise ValueError('nTheta and nPhi must be integers')

        if nTheta % 2 or nPhi % 2:
            raise ValueError('nPhi and nTheta must be even.')
        self.nTheta = nTheta 
        self.nPhi = nPhi

        # Make copy of surface with quadrature points according to nTheta, nPhi
        qpoints_phi = list(np.linspace(0, 0.5/surface.nfp, nPhi+1))
        qpoints_theta = list(np.linspace(0, 1., nTheta, endpoint=False))
        self.nfp = surface.nfp
        self.surface = SurfaceRZFourier(nfp=surface.nfp, stellsym=True, \
                                        mpol=surface.mpol, ntor=surface.ntor, \
                                        quadpoints_phi=qpoints_phi, \
                                        quadpoints_theta=qpoints_theta, \
                                        dofs=surface.dofs)

        # Determine the locations of the node points within a half period
        nodes_surf = self.surface.gamma()
        self.nNodes = np.prod(nodes_surf.shape[:2])
        nodes_hp = np.ascontiguousarray(np.zeros((self.nNodes, 3)))
        nodes_hp[:, 0] = nodes_surf[:, :, 0].reshape((-1))
        nodes_hp[:, 1] = nodes_surf[:, :, 1].reshape((-1))
        nodes_hp[:, 2] = nodes_surf[:, :, 2].reshape((-1))
        self.node_inds = np.arange(self.nNodes).reshape(nodes_surf.shape[:2])

        # Generate list of sets of nodes for each half period
        self.nodes = [[]]*self.nfp*2
        self.seg_signs = [[]]*self.nfp*2
        self.nodes[0] = nodes_hp
        self.seg_signs[0] = 1.0
        self.nodes[1] = np.ascontiguousarray(np.zeros((self.nNodes, 3)))
        self.nodes[1][:, 0] = self.nodes[0][:, 0]
        self.nodes[1][:, 1] = -self.nodes[0][:, 1]
        self.nodes[1][:, 2] = -self.nodes[0][:, 2]
        self.seg_signs[1] = -1.0
        for i in range(1, self.nfp):

            phi_rot = 2.0*i*np.pi/self.nfp

            self.nodes[2*i]   = np.ascontiguousarray(np.zeros((self.nNodes, 3)))
            self.nodes[2*i+1] = np.ascontiguousarray(np.zeros((self.nNodes, 3)))

            self.nodes[2*i][:, 0] = np.cos(phi_rot)*self.nodes[0][:, 0] - \
                                    np.sin(phi_rot)*self.nodes[0][:, 1]
            self.nodes[2*i][:, 1] = np.sin(phi_rot)*self.nodes[0][:, 0] + \
                                    np.cos(phi_rot)*self.nodes[0][:, 1]
            self.nodes[2*i][:, 2] = self.nodes[0][:, 2]

            self.nodes[2*i+1][:, 0] = np.cos(phi_rot)*self.nodes[1][:, 0] - \
                                      np.sin(phi_rot)*self.nodes[1][:, 1]
            self.nodes[2*i+1][:, 1] = np.sin(phi_rot)*self.nodes[1][:, 0] + \
                                      np.cos(phi_rot)*self.nodes[1][:, 1]
            self.nodes[2*i+1][:, 2] = self.nodes[1][:, 2]

            # Positive current direction reverses in reflected half-periods
            self.seg_signs[2*i] = 1.0
            self.seg_signs[2*i+1] = -1.0

        # Define the segments according to the pairs of nodes connecting them
        self.nTorSegments = nTheta*nPhi
        self.nPolSegments = nTheta*nPhi
        self.nSegments = self.nTorSegments + self.nPolSegments

        # Toroidal segments 
        segments_tor = np.zeros((self.nTorSegments, 2))
        segments_tor[:,0] = self.node_inds[:-1, :].reshape((self.nTorSegments))
        segments_tor[:,1] = self.node_inds[1:,  :].reshape((self.nTorSegments))

        # Map nodes to index in the segment array of segment originating 
        # from the respective node
        self.torSegmentKey = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        self.torSegmentKey[:-1,:] = \
            np.arange(self.nTorSegments).reshape((nPhi, nTheta))

        # Poloidal segments (on symmetry planes, only include segments for z>0)
        segments_pol = np.zeros((self.nPolSegments, 2))
        self.polSegmentKey = -np.ones(nodes_surf.shape[:2]).astype(np.int64)
        HalfNTheta = int(nTheta/2)

        segments_pol[:HalfNTheta, 0] = self.node_inds[0, :HalfNTheta]
        segments_pol[:HalfNTheta, 1] = self.node_inds[0, 1:HalfNTheta+1]
        self.polSegmentKey[0, :HalfNTheta] = np.arange(HalfNTheta) + self.nTorSegments
        for i in range(1, nPhi):
            polInd0 = HalfNTheta + (i-1)*nTheta
            polInd1 = polInd0 + nTheta
            segments_pol[polInd0:polInd1, 0] = self.node_inds[i, :]
            segments_pol[polInd0:polInd1-1, 1] = self.node_inds[i, 1:]
            segments_pol[polInd1-1, 1] = self.node_inds[i, 0]
            self.polSegmentKey[i, :] = np.arange(polInd0, polInd1) + self.nTorSegments

        segments_pol[-HalfNTheta:, 0] = self.node_inds[-1, :HalfNTheta]
        segments_pol[-HalfNTheta:, 1] = self.node_inds[-1, 1:HalfNTheta+1]
        self.polSegmentKey[-1, :HalfNTheta] = \
            np.arange(self.nPolSegments-HalfNTheta, self.nPolSegments) + self.nTorSegments

        # Join the toroidal and poloidal segments into a single array
        self.segments = \
            np.ascontiguousarray(np.zeros((self.nSegments, 2)).astype(np.int64))
        self.segments[:self.nTorSegments, :] = segments_tor[:, :]
        self.segments[self.nTorSegments:, :] = segments_pol[:, :]

        # Initialize currents to zero
        self.currents = np.ascontiguousarray(np.zeros((self.nSegments)))

        #self.nConstraints = self.nTorSegments - 2

        # Create a matrix listing which segments are connected to each node
        self.determine_connected_segments()

        # Add constraints to enforce continuity at each node
        self.initialize_constraints()
        self.add_continuity_constraints()

    def determine_connected_segments(self):
        """
        Determine which segments are connected to each node.
        """

        self.connected_segments = \
            np.ascontiguousarray(np.zeros((self.nNodes, 4)).astype(np.int64))

        halfNTheta = int(self.nTheta)

        for i in range(self.nPhi+1):
            for j in range(self.nTheta):

                if i == 0:
                    ind_tor_in  = \
                        self.torSegmentKey[i, (self.nTheta-j) % self.nTheta]
                    ind_tor_out = self.torSegmentKey[i, j]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, j]
                        ind_pol_out = self.polSegmentKey[i, j]
                    elif j < halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j]
                    elif j == halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, self.nTheta-j]
                        ind_pol_out = self.polSegmentKey[i, self.nTheta-j-1]

                elif i > 0 and i < self.nPhi:
                    ind_tor_in  = self.torSegmentKey[i-1, j]
                    ind_tor_out = self.torSegmentKey[i, j]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, self.nTheta-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                    ind_pol_out = self.polSegmentKey[i, j]

                else:
                    ind_tor_in  = self.torSegmentKey[i-1, j]
                    ind_tor_out = \
                        self.torSegmentKey[i-1, (self.nTheta-j) % self.nTheta]
                    if j == 0:
                        ind_pol_in  = self.polSegmentKey[i, 0]
                        ind_pol_out = self.polSegmentKey[i, 0]
                    elif j < halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j]
                    elif j == halfNTheta:
                        ind_pol_in  = self.polSegmentKey[i, j-1]
                        ind_pol_out = self.polSegmentKey[i, j-1]
                    else:
                        ind_pol_in  = self.polSegmentKey[i, self.nTheta-j]
                        ind_pol_out = self.polSegmentKey[i, self.nTheta-j-1]

                self.connected_segments[self.node_inds[i,j]][:] = \
                    [ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out]

    def initialize_constraints(self):

        self.constraints = collections.OrderedDict()

    def add_constraint(self, name, constraint_type, matrix_row, constant):
        """
        Add a linear equality constraint on the currents in the segments
        in the wireframe of the form 

            matrix_row * x = constant, 

            where:
                x is the array of currents in each segment
                matrix_row is a 1d array of coefficients for each segment
                constant is the constant appearing on the right-hand side

        Parameters
        ----------
            name: string
                Unique name for the constraint
            constraint_type: string
                Type of constraint 
            matrix_row: 1d double array
                Array of coefficients as described above
            constant: double
                Constant on the right-hand side of the equation above
        """

        if name in self.constraints.keys():
            raise ValueError('Constraint %s already exists' % (name))

        if matrix_row.size != self.nSegments:
            raise ValueError('matrix_row must have one element for every ' \
                             + 'segment in the wireframe')

        self.constraints[name] = \
            {'type': constraint_type, \
             'matrix_row': matrix_row, \
             'constant': constant}

    def remove_constraint(self, name):

        if isinstance(name, str):
            del self.constraints[name]
        else:
            for item in name:
                del self.constraints[item]

    def add_poloidal_current_constraint(self, current):
        """
        Add constraint to require the total poloidal current through the 
        inboard midplane to be a certain value (effectively sets the toroidal
        magnetic field). 

        Parameters
        ----------
            current: double
                Total poloidal current; i.e. the sum of the currents in all 
                poloidal segments passing through the inboard midplane.
                A positive poloidal current thereby creates a toroidal field 
                in the negative toroidal direction (clockwise when viewed from 
                above).
        """

        pol_current_per_segment = current/(2.0*self.nfp*self.nPhi)
        pol_current_sum = pol_current_per_segment * self.nPhi * 2

        halfNTheta = int(self.nTheta/2)
        seg_ind0 = self.nTorSegments + halfNTheta - 1
        seg_ind1a = seg_ind0 + halfNTheta
        seg_ind2a = self.nSegments
        seg_ind1b = seg_ind1a + 1
        seg_ind2b = self.nSegments - self.nTheta + 1

        matrix_row = np.zeros((1, self.nSegments))
        matrix_row[0,seg_ind0] = 1
        matrix_row[0,seg_ind1a:seg_ind2a:self.nTheta] = 1
        matrix_row[0,seg_ind1b:seg_ind2b:self.nTheta] = 1

        self.add_constraint('poloidal_current', 'poloidal_current', \
                            matrix_row, pol_current_sum)

    def remove_poloidal_current_constraint(self):

        self.remove_constraint('poloidal_current')

    def set_poloidal_current(self, current):
        """
        Set the constraint requiring the total poloidal current through the 
        inboard midplane to be a certain value (effectively sets the toroidal
        magnetic field). 

        This method will replace an existing poloidal current constraint and
        create one if one does not exist.

        Parameters
        ----------
            current: double
                Total poloidal current; i.e. the sum of the currents in all 
                poloidal segments passing through the inboard midplane.
                A positive poloidal current thereby creates a toroidal field 
                in the negative toroidal direction (clockwise when viewed from 
                above).
        """

        if 'poloidal_current' in self.constraints:
            self.remove_constraint('poloidal_current')

        self.add_poloidal_current_constraint(current)

    def add_toroidal_current_constraint(self, current):
        """
        Add constraint to require the total toroidal current through a poloidal
        cross-section to be a certain value (effectively requires a helical
        current distribution when combined with a poloidal current constraint).
 
        Parameters
        ----------
            current: double
                Total toroidal current; i.e. the sum of the currents in all 
                toroidal segments passing through a symmetry plane.
                A positive toroidal current thereby creates a dipole moment
                in the positive "z" direction.
        """

        matrix_row = np.zeros((1, self.nSegments))
        matrix_row[0,:self.nTheta] = 1

        self.add_constraint('toroidal_current', 'toroidal_current', \
                            matrix_row, current)

    def remove_toroidal_current_constraint(self):

        self.remove_constraint('toroidal_current')

    def set_toroidal_current(self, current):
        """
        Set the constraint requiring the total toroidal current through a 
        poloidal cross-section to be a certain value (effectively requires a 
        helical current distribution when combined with a poloidal current 
        constraint).

        This method will replace an existing toroidal current constraint and
        create one if one does not exist.

        Parameters
        ----------
            current: double
                Total toroidal current; i.e. the sum of the currents in all 
                toroidal segments passing through a symmetry plane.
                A positive toroidal current thereby creates a dipole moment
                in the positive "z" direction.
        """

        if 'toroidal_current' in self.constraints:
            self.remove_constraint('toroidal_current')

        self.add_toroidal_current_constraint(current)

    def add_segment_constraints(self, segments):
        """
        Adds a constraint or constraints requiring the current to be zero in
        one or more given segments.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments to be constrained
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.nSegments):
            raise ValueError('Segment indices must be positive and less than ' \
                             + ' the number of segments in the wireframe')

        for i in range(len(segments)):

            matrix_row = np.zeros((1, self.nSegments))
            matrix_row[0,segments[i]] = 1

            self.add_constraint('segment_%d' % (segments[i]), 'segment', \
                                matrix_row, 0)

    def remove_segment_constraints(self, segments):
        """
        Removes constraints restricting the currents in given segment(s) to be
        zero.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments for which constraints are to
                be removed
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.nSegments):
            raise ValueError('Segment indices must be positive and less than ' \
                             ' the number of segments in the wireframe')

        for i in range(len(segments)):
    
            self.remove_constraint('segment_%d' % (segments[i]))

    def set_segments_constrained(self, segments):
        """
        Ensures that one or more given segments are constrained to have zero
        current.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments to be constrained
        """

        # Free existing constrained segments to avoid conflicts
        self.set_segments_free(segments)

        self.add_segment_constraints(segments)

    def set_segments_free(self, segments):
        """
        Ensures that one or more given segments are unconstrained.

        Parameters
        ----------
            segments: integer or array/list of integers
                Index of the segmenet or segments to be unconstrained
        """

        if np.isscalar(segments):
            segments = np.array([segments])
        else:
            segments = np.array(segments)

        if np.any(segments < 0) or np.any(segments >= self.nSegments):
            raise ValueError('Segment indices must be positive and less than ' \
                             ' the number of segments in the wireframe')

        for i in range(len(segments)):
            if 'segment_%d' % (i) in self.constraints:
                self.remove_constraint('segment_%d' % (segments[i]))

    def free_all_segments(self):
        """
        Remove any existing constraints that restrict individual segments to
        carry zero current.
        """

        for constr in self.constraints:
            if self.constraints[constr]['type'] == 'segment':
                self.remove_constraint(constr)

    def constrained_segments(self):
        """
        Returns the IDs of the segments that are currently constrained to have
        zero current.
        """  

        constr_keys = [key for key in self.constraints.keys() \
                       if self.constraints[key]['type'] == 'segment']

        return [int(key.split('_')[1]) for key in constr_keys]

    def add_continuity_constraints(self):
        """
        Add constraints to ensure current continuity at each node. This is
        called automatically on initialization and doesn't normally need to
        be called by the user.
        """

        for i in range(self.nPhi+1):
            for j in range(self.nTheta):

                if i == 0:
                    if j == 0 or j >= self.nTheta/2:
                        # Constraint automatically satisfied due to symmetry
                        continue

                elif i == self.nPhi:
                    if j == 0 or j >= self.nTheta/2:
                        # Constraint automatically satisfied due to symmetry
                        continue

                ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out = \
                    list(self.connected_segments[self.node_inds[i,j]])

                self.add_continuity_constraint(self.node_inds[i,j], \
                    ind_tor_in, ind_pol_in, ind_tor_out, ind_pol_out)

    def add_continuity_constraint(self, node_ind, ind_tor_in, ind_pol_in, \
                                  ind_tor_out, ind_pol_out):

        name = 'continuity_node_%d' % (node_ind)

        matrix_row = np.zeros((1, self.nSegments))
        matrix_row[0, [ind_tor_in,  ind_pol_in ]] = -1
        matrix_row[0, [ind_tor_out, ind_pol_out]] = 1
        
        self.add_constraint(name, 'continuity', matrix_row, 0.0)
        

    def constraint_matrices(self, remove_redundancies=True):
        """
        Return the matrices for the system of equations that define the linear
        equality constraints for the wireframe segment currents. The equations
        have the form
            B*x = d,
        where x is a column vector with the segment currents, B is a matrix of
        coefficients for the segment currents in each equation, and d is a
        column vector of constant terms in each equation.

        The matrices are initially constructed to have one row (equation) per
        constraint. However, if some of the constraints may be redundant.
        By default, this function will check the constraints for certain 
        redundancies and remove rows from the matrix that are found to be 
        redundant. This is necessary, e.g. for some constrained linear 
        least-squares solvers in which the constraint matrix must be full-rank. 
        However, the user may also opt out of row reduction such that the 
        output matrices contain every constraint equation explicitly.

        Note: the function does not put the output matrix into reduced row-
        echelon form or otherwise guarantee that it will be full-rank. Rather, 
        it only checks for cases in which all four segments connected to a node
        are constrained to have zero current, in which case the node's
        continuity constraint is redundant. If there are other redundancies,
        e.g. due to arbitrary constraints introduced by the user that aren't
        of the usual constraint types, these may not be removed.

        Parameters
        ----------
            remove_redundancies: boolean (optional)
                If true (the default option), rows in the matrix found to be
                redundant will be removed. If false, no checks for redundancy
                will be performed and all constraints will be represented in the
                output matrices.

        Returns
        -------
            constraints_B: 2d double array
                The matrix B in the constraint equation
            constraints_d: 1d double array (column vector)
                The column vector on the right-hand side of the constraint 
                equation
        """

       
        # If matrix is not full rank, look for redundant continuity constraints
        if remove_redundancies: 

            inactive_nodes = self.find_inactive_nodes()
            print('    Number of inactive nodes found: %d' % (len(inactive_nodes)))
            inactive_node_names = ['continuity_node_%d' % (i) \
                                   for i in inactive_nodes]

            constraints_B = np.ascontiguousarray( \
                np.concatenate([self.constraints[key]['matrix_row'] \
                                for key in self.constraints.keys() \
                                if key not in inactive_node_names], axis=0))

            constraints_d = np.ascontiguousarray( \
                np.zeros((constraints_B.shape[0], 1)))

            constraints_d[:] = [[self.constraints[key]['constant']] \
                 for key in self.constraints.keys() \
                 if key not in inactive_node_names]

        else:

            constraints_B = np.ascontiguousarray( \
                np.concatenate([constr['matrix_row'] for constr \
                                in self.constraints.values()], axis=0))

            constraints_d = np.ascontiguousarray(\
                            np.zeros((len(self.constraints), 1)))
            constraints_d[:] = \
                [[constr['constant']] for constr in self.constraints.values()]

        return constraints_B, constraints_d

    def find_inactive_nodes(self):
        """
        Determines which nodes have no current flowing through them according
        to existing segment constraints (i.e. constraints that require 
        individual segments to have zero current).
        """

        # Tally how many inactive segments each node is connected to
        node_sum = np.zeros((self.nNodes))

        for seg_ind in self.constrained_segments():
            connected_nodes = np.sum(self.connected_segments == seg_ind, axis=1)
            node_sum[connected_nodes > 0] += 1

        # If all four connected segments are constrained, the continuity 
        # constraint is redundant
        return np.where(node_sum >= 4)[0]
            
    def make_plot_3d(self, ax=None):
        """
        Make a plot of the wireframe grid, including nodes and segments.
        """

        import matplotlib.pylab as pl
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pl_segments = np.zeros((2*self.nfp*self.nSegments, 2, 3))
        pl_currents = np.zeros((2*self.nfp*self.nSegments))
   
        for i in range(2*self.nfp):
            ind0 = i*self.nSegments
            ind1 = (i+1)*self.nSegments
            pl_segments[ind0:ind1,:,:] = self.nodes[i][:,:][self.segments[:,:]]
            pl_currents[ind0:ind1] = self.currents[:]*1e-6

        lc = Line3DCollection(pl_segments)
        lc.set_array(pl_currents)
        lc.set_clim(np.max(np.abs(self.currents*1e-6))*np.array([-1, 1]))
        lc.set_cmap('coolwarm')

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(projection='3d')

            ax.set_xlim([np.min(pl_segments[:,:,0], axis=(0,1)),
                         np.max(pl_segments[:,:,0], axis=(0,1))])
            ax.set_ylim([np.min(pl_segments[:,:,1], axis=(0,1)),
                         np.max(pl_segments[:,:,1], axis=(0,1))])
            ax.set_zlim([np.min(pl_segments[:,:,2], axis=(0,1)),
                         np.max(pl_segments[:,:,2], axis=(0,1))])

            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            cb = pl.colorbar(lc)
            cb.set_label('Current (MA)')

        ax.add_collection(lc)

        return(ax)

    def make_plot_2d(self, extent='field period', quantity='currents', ax=None):
        """
        Make a 2d plot of the segments in the wireframe grid.

        Parameters
        ----------
            extent: string (optional)
                Portion of the torus to be plotted. Options are 'half period',
                'field period' (default), and 'torus'.
            quantity: string (optional)
                Quantity to be represented in the color of each segment.
                Options are 'currents' (default) and 'constrained segments'.
            ax: instance of the matplotlib.pyplot.Axis class (optional)
                Axis on which to generate the plot. If None, a new plot will
                be created.

        Returns
        -------
            ax: instance of the matplotlib.pyplot.Axis class
                Axis instance on which the plot was created.
        """

        import matplotlib.pyplot as pl
        from matplotlib.collections import LineCollection

        if extent=='half period':
            nHalfPeriods = 1
        elif extent=='field period':
            nHalfPeriods = 2
        elif extent=='torus':
            nHalfPeriods = self.nfp * 2
        else:
            raise ValueError('extent must be \'half period\', ' \
                             + '\'field period\', or \'torus\'')

        pl_segments = np.zeros((nHalfPeriods*self.nSegments, 2, 2))
        pl_quantity = np.zeros((nHalfPeriods*self.nSegments))
   
        for i in range(nHalfPeriods):
            ind0 = i*self.nSegments
            ind1 = (i+1)*self.nSegments
            if i % 2 == 0:
                pl_segments[ind0:ind1,0,0] = \
                    np.floor(self.segments[:,0]/self.nTheta)
                pl_segments[ind0:ind1,0,1] = self.segments[:,0] % self.nTheta
                pl_segments[ind0:ind1,1,0] = \
                    np.floor(self.segments[:,1]/self.nTheta)
                pl_segments[ind0:ind1,1,1] = self.segments[:,1] % self.nTheta

                loop_segs = np.where( \
                    np.logical_and(pl_segments[ind0:ind1,0,1] == self.nTheta-1,\
                                   pl_segments[ind0:ind1,1,1] == 0))
                pl_segments[ind0+loop_segs[0],1,1] = self.nTheta

            else:
                pl_segments[ind0:ind1,0,0] = \
                    2*i*self.nPhi - np.floor(self.segments[:,0]/self.nTheta)
                pl_segments[ind0:ind1,0,1] = \
                    self.nTheta - (self.segments[:,0] % self.nTheta)
                pl_segments[ind0:ind1,1,0] = \
                    2*i*self.nPhi - np.floor(self.segments[:,1]/self.nTheta)
                pl_segments[ind0:ind1,1,1] = \
                    self.nTheta - (self.segments[:,1] % self.nTheta)

                loop_segs = np.where( \
                    np.logical_and(pl_segments[ind0:ind1,0,1] == 1, \
                                   pl_segments[ind0:ind1,1,1] == self.nTheta))
                pl_segments[ind0+loop_segs[0],1,1] = 0

            if quantity=='currents':
                pl_quantity[ind0:ind1] = self.currents[:]*1e-6
            elif quantity=='constrained segments':
                pl_quantity[ind0:ind1][self.constrained_segments()] = 1
            else:
                raise ValueError('Unrecognized quantity for plotting')

        lc = LineCollection(pl_segments)
        lc.set_array(pl_quantity)
        if quantity=='currents':
            lc.set_clim(np.max(np.abs(self.currents*1e-6))*np.array([-1, 1]))
        elif quantity=='constrained segments':
            lc.set_clim([-1, 1])
        lc.set_cmap('coolwarm')

        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot()

        ax.set_xlim((-1, nHalfPeriods*self.nPhi + 1))
        ax.set_ylim((-1, self.nTheta + 1))

        ax.set_xlabel('Toroidal index')
        ax.set_ylabel('Poloidal index')
        cb = pl.colorbar(lc)
        if quantity=='currents':
            cb.set_label('Current (MA)')
        elif quantity=='constrained segments':
            cb.set_label('1 = constrained; 0 = free')


        ax.add_collection(lc)


