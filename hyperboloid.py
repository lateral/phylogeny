import numpy as np

def tangent_norm(tangent):
    """
    Return the norm of the given tangent vector to the hyperboloid.
    """
    return np.sqrt(max(minkowski_dot(tangent, tangent), 0))

def minkowski_dot(u, v):
    """
    `u` and `v` are vectors in Minkowski space.
    """
    rank = u.shape[-1] - 1
    euc_dp = u[:rank].dot(v[:rank])
    return euc_dp - u[rank] * v[rank]

def minkowski_dot_matrix(vecs_a, vecs_b):
    """
    Return the matrix giving the Minkowski dot product of every vector in vecs_a with every vector in vecs_b.
    """
    rank = vecs_a.shape[1] - 1
    euc_dps = vecs_a[:,:rank].dot(vecs_b[:,:rank].T)
    timelike = vecs_a[:,rank][:,np.newaxis].dot(vecs_b[:,rank][:,np.newaxis].T)
    return euc_dps - timelike


class Hyperboloid:

    def __init__(self, rho, local_dimension):
        """
        The hyperboloid with radius -1 * rho ** 2 (and curvature 1 / rho ** 2).
        """
        self.local_dimension = local_dimension
        self.rho = rho

    def exponential(self, base, tangent):
        """
        Compute the exponential of `tangent` from the point `base`.
        """
        tangent = tangent.copy()
        norm = tangent_norm(tangent)
        if norm == 0:
            return base
        tangent /= norm
        arg = norm / self.rho
        return np.cosh(arg) * base + self.rho * np.sinh(arg) * tangent

    def logarithm(self, base, other_pt):
        d = self.distance(base, other_pt)
        if d == 0:
            return np.zeros_like(self.basepoint())
        mdp = minkowski_dot(base, other_pt)
        denom = np.sqrt((mdp / self.rho) ** 2 - self.rho ** 2)
        factor = d / denom
        return factor * (other_pt + self.rho ** -2 * mdp * base)

    def geodesic_parallel_transport(self, base, direction, tangent):
        """
        Parallel transport `tangent`, a tangent vector at point `base`, along the
        geodesic in the direction `direction` (another tangent vector at point
        `base`, not necessarily unit length)
        """
        norm_direction = np.sqrt(minkowski_dot(direction, direction))
        unit_direction = direction / norm_direction
        parallel_component = minkowski_dot(tangent, unit_direction)
        unit_direction_transported = (self.rho ** -1) * np.sinh((self.rho ** -1) * norm_direction) * base + np.cosh((self.rho ** -1) * norm_direction) * unit_direction
        return parallel_component * unit_direction_transported + tangent - parallel_component * unit_direction 

    def to_poincare_ball_point(self, hyperboloid_pt):
        """
        Project the point of the rho-hyperboloid onto the (unit) Poincare ball.
        Post: len(result) == self.local_dimension
        """
        N = self.local_dimension
        rho_disc_pt = hyperboloid_pt[:N] / (hyperboloid_pt[N] / self.rho + 1)
        unit_disc_pt = rho_disc_pt / self.rho
        return unit_disc_pt

    def sample_uniform_on_disc(self, max_distance):
        # FIXME how to test coverage?
        """
        Return a sample drawn uniformly at random from the disc of
        radius `max_distance` on the hyperboloid, centred around the basepoint.
        NOTE: formula is specific to the case of local-dimension 2.
        """
        assert self.local_dimension == 2
        rank = self.local_dimension
        tangent = np.random.randn(rank + 1)
        tangent[rank] = 0
        tangent /= tangent_norm(tangent)
        # we use inversion sampling: invert the CDF, apply result to uniform random samples from [0,1]
        p = np.random.uniform()
        tangent *= self.rho * np.arccosh(1 + p * (np.cosh(max_distance / self.rho) - 1))
        return self.exponential(self.basepoint(), tangent)

    def distance(self, pt0, pt1):
        """
        Return the distances between two points on the hyperboloid.
        """
        mdp = minkowski_dot(pt0, pt1)
        arccosh_arg = -1 * mdp / (self.rho ** 2)
        if arccosh_arg < 1:
            arccosh_arg = 1
        return self.rho * np.arccosh(arccosh_arg)

    def distance_gradient(self, movable, immovable):
        """
        Return the gradient of the distance between the two points provided, with respect to the first point.
        Note: returned vector is NOT yet projected onto the
        tangent space of the first point.
        """
        mdp = minkowski_dot(movable, immovable)
        d_dist_wrt_mdp = -1 / np.sqrt((mdp / self.rho) ** 2 - self.rho ** 2)
        return d_dist_wrt_mdp * immovable  # immovable is the gradient of the mdp

    def pairwise_distances(self, pts_a, pts_b=None):
        """
        Return the pairwise distances between the provided hyperboloid points.
        Pre: self.contains(pt) for pt in pts_a and (pts_b is None or (self.contains(pt) for pt in pts_b)).
        Post: res.shape = (len(pts_a), len(pts_b)).
        """
        if pts_b is None:
            pts_b = pts_a
        mdps = minkowski_dot_matrix(pts_a, pts_b)
        arccosh_arg = -1 * mdps / (self.rho ** 2)
        arccosh_arg[arccosh_arg < 1] = 1  # in case of numerical instability
        dists = self.rho * np.arccosh(arccosh_arg)
        return dists

    def isoceles_leg_length(self, angle, opposite_length):
        """
        Return the length of the two equal legs of an isoceles triangle such
        that they meet at an angle of angle and the opposing side length is
        `opposite_length`.
        """
        return self.rho * np.arccosh(np.sqrt((np.cosh(opposite_length / self.rho) - np.cos(angle)) / (1 - np.cos(angle))))

    def generate_equilateral_triangle(self, radius):
        """
        Return the vertices of an equilateral triangle whose vertices are the given distance from the origin.
        """
        assert self.local_dimension == 2
        rank = self.local_dimension
        tangents = [np.array([np.cos(theta), np.sin(theta), 0]) for theta in np.linspace(0, 2 * np.pi, 4)[0:3]]
        basept = self.basepoint()
        vertices = [self.exponential(basept, radius * tangent) for tangent in tangents]
        return np.array(vertices)

    def contains(self, point, precision=7):
        """
        Return True if the point provided lies on this
        hyperboloid, considering dimension and the defining
        constraint with precision up the specified number of
        places; else return False.
        """
        if len(point) != self.local_dimension + 1:
            return False
        mdp = minkowski_dot(point, point)
        atol = 10 ** -1 * precision
        return np.isclose(mdp, -1 * self.rho ** 2, atol=atol)

    def basepoint(self):
        """
        Return the basepoint of the hyperboloid.
        Post:
        res.shape == (self.local_dimension + 1,) and
        self.contains(res) and
        res[i] == 0 for i < self.local_dimension
        """
        return self.rho * np.eye(self.local_dimension + 1)[self.local_dimension,:]

    def basepoint_tangent(self, i):
        """
        Return the ith basis vector of the tangent space of the basepoint of the hyperboloid.
        Pre: i in range(self.local_dimension)
        Post:
        res.shape == (self.local_dimension + 1,) and
        tangent_norm(res) == 1 and
        minkowski_dot(res, self.basepoint()) == 0
        """
        return np.eye(self.local_dimension + 1)[i,:]

    def ensure_on_hyperboloid(self, pt):
        """
        Given a time-like vector `pt`, rescale it such that it
        lies on the hyperboloid.
        """
        factor = np.sqrt(-1 * minkowski_dot(pt, pt))
        return (self.rho / factor) * pt

    def project_onto_tangent_space(self, hyperboloid_point, minkowski_tangent):
        scalar = minkowski_dot(hyperboloid_point, minkowski_tangent) / (self.rho ** 2)
        return minkowski_tangent + scalar * hyperboloid_point

    def initialisation_point(self, stddev):
        """
        Sample a tangent vector at the basepoint from a normal
        distribution and return its exponential.
        Post: res.shape == (self.local_dimension + 1,) and
        self.contains(res)
        """
        tangent = np.zeros((self.local_dimension + 1,), dtype=np.float64)
        tangent[:self.local_dimension] = stddev * np.random.randn(self.local_dimension)
        return self.ensure_on_hyperboloid(self.exponential(self.basepoint(), tangent))

    def initialisation_points(self, stddev, num_points):
        """
        Return the 2d numpy array obtained via `num_points`
        calls to self.initialisation_point(stddev), where axis
        0 enumerates the calls.
        """
        return np.array([self.initialisation_point(stddev) for _ in range(num_points)])

    def midpoint(self, pt0, pt1):
        """
        Return the mid-point on the geodesic segment connecting pt0 to pt1.
        """
        return self.ensure_on_hyperboloid(pt0 + pt1)

    def random_unit_tangent(self, point, orthogonal_to=None):
        """
        Returns a unit-length tangent vector at `point` which is orthogonal to the tangent `orthogonal_to`.
        `point` is on the hyperboloid.
        `orthogonal_to`, if specified, is a unit vector in the tangent space of `point`.
        """
        if orthogonal_to is None:
            orthogonal_to = np.zeros_like(point)
        trandom = self.project_onto_tangent_space(point, 0.01 * np.random.randn(self.local_dimension + 1))
        trandom = trandom - minkowski_dot(orthogonal_to, trandom) * orthogonal_to
        return trandom / tangent_norm(trandom)

    def embed_tree(self, tree):
        """
        Embeds the tree using a generalisation of Sarkar's construction.
        The root of the tree is embedded at the basepoint of the hyperboloid.
        For any node, the logarithms of the child nodes and the parent node,
        when normalised, are uniformly spaced around a unit circle lying in a
        random 2 dimensional subspace of the tangent space.
        `tree` is a rooted tree.
        Pre: all edges have length > 0.
        Returns a dictionary mapping each node of tree to a point on this hyperboloid.
        """
        vectors = dict()
        for node in tree.preorder_internal_node_iter():
            child_edges = node.incident_edges()[:-1]
            parent_edge = node.incident_edges()[-1]
            
            if node != tree.seed_node:  # seed_node means root node
                # generic case
                tparent = self.logarithm(vectors[node], vectors[node.parent_node])
                tangent1 = tparent / tangent_norm(tparent)  # make unit vector for calculations
                tangent2 = self.random_unit_tangent(vectors[node], orthogonal_to=tangent1)
                degree = len(child_edges) + 1
            else:
                # case of the root node, different since there is no parent tangent to be aware of
                # place at the basepoint of the hyperboloid
                vectors[node] = self.basepoint()
                tangent1 = self.basepoint_tangent(0)
                tangent2 = self.basepoint_tangent(1)
                degree = len(child_edges) # the root node has a dummy incident edge
                
            # orthogonal basis for the plane is tangent1, tangent2
            for i, child_edge in enumerate(child_edges):
                child = child_edge.head_node
                angle = ((i + 1) / degree) * (2 * np.pi)  # angle 0 is reserved for the parent, in case of non-root node
                tchild = child_edge.length * (np.cos(angle) * tangent1 + np.sin(angle) * tangent2)
                vectors[child] = self.exponential(vectors[node], tchild)
        return vectors
