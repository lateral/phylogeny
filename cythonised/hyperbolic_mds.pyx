#cython: boundscheck=True, wraparound=False, language_level=3, initializedcheck=True
import numpy as np
from scipy.optimize import fminbound  # fminbound = brent's method but within specified range


cdef extern from "math.h":
    double sqrt(double)
    double log(double)
    double exp(double)
    double cosh(double)
    double sinh(double)
    double acosh(double)

np.seterr(all='raise')  # floating point errors will be raised as exceptions

cdef double MIN_GRAD_NORM = 1e-12  # minimum gradient norm for normalisation to be attempted


cdef class MDS:
    cdef:
        public double[:,::1] points
        public double[:,::1] gradients
        public double[::1] partial_costs
        public int rounds_done
        public int local_dim
        public int num_points
        public double lr
        double[::1] scratch_pt  # a provisional point used during line search
        double[::1] scratch_tangent  # a provisional tangent used during line search
        double[:,::1] deltas
        double[:,::1] weightings
        double rho
        double max_step_size
        double xtol

    #TODO really need to take rank and number of points at init, otherwise can't properly initialise arrays
    def __init__(self, rho, lr, max_step_size=0.25, xtol=1e-7):
        """
        If `rho` > 0, it is interpreted as the "radius" of the hyperboloid.
        `rho` = np.inf means use Euclidean space.
        `max_step_size` bounds the maximal movement of a single point in a single round.
        `lr` > 0 means use gradient descent with this learning rate; pass 0. to
        use Brent's method (bounded).
        `xtol` is passed to Brent's method.
        """
        self.rho = rho
        self.lr = lr
        self.rounds_done = 0
        self.local_dim = -1
        self.num_points = -1
        self.max_step_size = max_step_size
        self.xtol = xtol
        self.points = None

    # FOR PICKLING PROTOCOL
    def __getstate__(self):
        return (self.rho, self.lr, self.rounds_done, self.local_dim, self.num_points, self.max_step_size, self.xtol, np.array(self.points))

    def __setstate__(self, state):
        rho, lr, rounds_done, local_dim, num_points, max_step_size, xtol, points = state
        self.rho = rho
        self.lr = lr
        self.rounds_done = rounds_done
        self.local_dim = local_dim
        self.num_points = num_points
        self.max_step_size = max_step_size
        self.xtol = xtol
        self.points = points

    def set_points(self, points):
        """
        Pre: if self.rho == np.inf then points[i,-1] == 0 for all i, otherwise each
        point must lie on the rho hyperboloid.
        """
        self.points = points
        self.num_points = self.points.shape[0]
        self.local_dim = self.points.shape[1] - 1
        if self.rho == np.inf:
            for i in range(self.num_points):
                assert self.points[i, self.local_dim] == 0.
        # TODO should check that points are on the hyperboloid
        self.gradients = np.zeros_like(self.points)
        self.scratch_pt = np.zeros_like(self.points[0,:])
        self.scratch_tangent = np.zeros_like(self.points[0,:])
        self.partial_costs = np.zeros((self.num_points,), dtype=np.float64)

    def get_points(self):
        """
        Return the points as a Numpy array.
        """
        if self.points is None:
            return None
        return np.array(self.points)
        
    def get_gradients(self):
        return np.array(self.gradients)

    def get_partial_costs(self):
        return np.array(self.partial_costs)

    def fit(self,
            double[:,::1] deltas,
            double[:,::1] normalisations,
            double stopping_distance,
            int max_rounds,
            int checkpoint_interval):
        """
        Pre: self.get_points() is not None.
        """
        cdef:
            double norm

        checkpoints = []
        def checkpoint(step_sizes, numfuncs, gradient_norms):
            cost = np.array(self.partial_costs).sum()
            checkpoints.append((np.array(self.points).copy(),
                                self.rounds_done,
                                cost,
                                np.array(self.partial_costs).copy(),
                                step_sizes.copy(),
                                numfuncs.copy(),
                                gradient_norms.copy()))

        self.deltas = deltas
        # derive weightings from normalisations
        # we use weightings
        self.weightings = np.zeros_like(normalisations)
        for i in range(self.num_points):
            for j in range(self.num_points):
                if i != j:
                    self.weightings[i,j] = normalisations[i,j] ** -1
        self.weightings /= np.array(self.weightings).mean()

        step_sizes = np.zeros((self.num_points,), dtype=np.float64)
        numfuncs = np.zeros((self.num_points,), dtype=np.int64)
        gradient_norms = np.zeros((self.num_points,), dtype=np.float64)

        self.rounds_done = 0
        while max_rounds == -1 or self.rounds_done < max_rounds:
            will_checkpoint = (self.rounds_done % checkpoint_interval == 0)
            try:
                if will_checkpoint:
                    # calculate the gradient norms
                    for i in range(self.num_points):
                        self.partial_cost_and_gradient(i)
                        gradient_norms[i] = tangent_norm(self.gradients[i,:])

                # TODO if want to use the (aggregate?) gradient norm as a stopping condition, do so here

                for i in range(self.num_points):
                    self.partial_cost_and_gradient(i)
                    if self.lr == 0.:
                        # use bounded Brent
                        step_sizes[i], numfuncs[i] = self._update_point_via_bounded_brent(i)
                    else:
                        # gradient descent : just step with -1 * lr * gradient

                        # clip gradient if necessary to ensure self.max_step_size is not exceeded
                        norm  = tangent_norm(self.gradients[i,:])
                        if norm * self.lr > self.max_step_size:
                            for k in range(self.local_dim + 1):
                                self.gradients[i,k] *= (self.max_step_size / (self.lr * norm))
                            step_sizes[i] = self.max_step_size
                        else:
                            step_sizes[i] = self.lr * norm

                        _effect_update(self.points[i,:],
                                       self.gradients[i,:],
                                       -1 * self.lr,
                                       self.points[i,:], self.rho)


            except (FloatingPointError, ZeroDivisionError):
                # NB the only reason we are catching this exception and
                # returning a failure flag is because we want to be able to
                # call fit() from joblib parallel, which can't handle
                # exceptions.
                return False, True, checkpoints

            if will_checkpoint:
                checkpoint(step_sizes, numfuncs, gradient_norms)

            # the maximum of the step sizes determines whether to stop
            largest_step = np.max(step_sizes)
            if (largest_step < stopping_distance):
                # stopping condition is satified
                return True, False, checkpoints
            self.rounds_done += 1
        return False, False, checkpoints

    cdef _update_point_via_bounded_brent(MDS self, int i):
        """
        Update the ith point using Brent's method in the
        direction of the gradient vector.  Return a 2-tuple
        (step_size, numfunc) giving the step size and the
        number of iterations.
        Pre: self.gradients[i,:] contains the unnormalised gradient of the
        objective w.r.t the ith point.
        """
        cdef:
            int k
            double norm

        # normalise the gradient to obtain the direction of search
        norm = tangent_norm(self.gradients[i,:])
        if norm < MIN_GRAD_NORM:
            return 0, 0
        for k in range(self.local_dim + 1):
            self.scratch_tangent[k] = self.gradients[i,k] / norm

        # perform the line search using Brent's method
        best_t, func_val, _, numfunc = fminbound(
            func=self.line_function,
            x1=0.,
            x2=self.max_step_size,
            args=(i,),
            xtol=self.xtol,
            disp=1,  # print a message if a round fails to converge
            full_output=True)

        # update the point
        _effect_update(self.points[i,:], self.scratch_tangent, -1 * best_t, self.points[i,:], self.rho)

        return best_t, numfunc

    cpdef double line_function(MDS self, double t, int i) except? -1:
        """
        Return the value of the cost function when the ith point p is replaced with
            exp_p (-t * gradients[i,:]).
        Used for line search.
        """
        cdef:
            double cost = 0
            double mdp, dist
            int j

        # calculate the provisional point
        _effect_update(self.points[i,:], self.gradients[i,:], -t, self.scratch_pt, self.rho)

        # calculate the partial cost at the provisional point
        for j in range(self.num_points):
            if i == j:
                continue
            if self.rho == np.inf:
                dist = euclidean_distance(self.scratch_pt, self.points[j,:])
            else:
                mdp = minkowski_dot(self.scratch_pt, self.points[j,:])
                dist = mdp_to_distance(mdp, self.rho)
            cost += ((dist - self.deltas[i,j]) ** 2) * self.weightings[i,j]

        # normalise by number of other points
        cost /= (self.num_points - 1)

        return cost

    cpdef int partial_cost_and_gradient(MDS self, int i) except -1:
        """
        Calculate the summand of the "likelihood" function
        corresponding to the ith point and the gradient with
        respect to the ith point, writing these to the instance
        variables.
        The gradient is a vector in the hyperboloid tangent
        space of the ith point.
        """
        cdef:
            int j, k
            double dist
            double d_dist_wrt_mdp
            double mdp
            double scalar

        # set to zero
        for k in range(self.local_dim + 1):
            self.gradients[i, k] = 0
        self.partial_costs[i] = 0

        for j in range(self.num_points):
            if i == j:
                continue
            if self.rho == np.inf:
                dist = euclidean_distance(self.points[i,:], self.points[j,:])
            else:
                mdp = minkowski_dot(self.points[i,:], self.points[j,:])
                dist = mdp_to_distance(mdp, self.rho)

            self.partial_costs[i] += ((dist - self.deltas[i,j]) ** 2) * self.weightings[i,j]

            scalar = 2 * (dist - self.deltas[i,j]) * self.weightings[i,j]
            if self.rho == np.inf:
                scalar *= 2 / dist
                for k in range(self.local_dim + 1):
                    self.gradients[i, k] += scalar * (self.points[i, k] - self.points[j, k])
            else:
                d_dist_wrt_mdp = -1 / sqrt((mdp / self.rho) ** 2 - self.rho ** 2)
                scalar *= d_dist_wrt_mdp
                for k in range(self.local_dim + 1):
                    self.gradients[i, k] += scalar * self.points[j, k]

        # normalise by number of other points
        self.partial_costs[i] /= (self.num_points - 1)
        for k in range(self.local_dim + 1):
            self.gradients[i, k] /= (self.num_points - 1)

        if self.rho != np.inf:
            project_onto_tangent_space(self.points[i,:], self.gradients[i,:], self.rho)


# note: this doesn't need to be callable from Python at all - so long as we can extract the vectors from the Cythonized sequence MDS (which _does_ need to be Python callable) then we can use the Python hyperboloid class for evaluation calculations
# indeed it would be smart to continue using the current Python language Hyperboloid class for everything that is not fitting (e.g. init).

cdef inline double minkowski_dot(double[::1] u, double[::1] v):
    cdef int i, rank
    cdef double result = 0
    rank = u.shape[0]
    for i in range(rank - 1):
        result += u[i] * v[i]
    result -= u[rank - 1] * v[rank - 1]
    return result

cdef inline double tangent_norm(double[::1] tangent):
    """
    Return the tangent norm of the supplied vector, assumed to be tangent to
    the hyperboloid (for some rho) or lying in the Euclidean subspace spanned
    by all co-ordinates except the last.
    """
    cdef double mdp, norm
    mdp = minkowski_dot(tangent, tangent)
    if mdp < 0:
        mdp = 0
    return sqrt(mdp)

cdef inline double mdp_to_distance(double mdp, double rho):
    """
    Given the value of the Minkowski dot product between two points on the rho
    hyperboloid, return the distance between the points.
    """
    cdef double arccosh_arg
    arccosh_arg = -1 * mdp / (rho ** 2)
    if arccosh_arg < 1:
        arccosh_arg = 1
    return rho * acosh(arccosh_arg)

# TODO cross check maths
cdef inline project_onto_tangent_space(double[::1] hyperboloid_point, double[::1] minkowski_tangent, double rho):
    """
    """
    cdef double scalar
    cdef int i
    scalar = minkowski_dot(hyperboloid_point, minkowski_tangent) / (rho ** 2)
    for i in range(hyperboloid_point.shape[0]):
        minkowski_tangent[i] += scalar * hyperboloid_point[i]

cdef inline ensure_on_hyperboloid(double[::1] pt, double rho):
    """
    Given a time-like vector `pt`, scale it so that it lies on the `rho` hyperboloid.
    """
    cdef double scalar
    cdef double mdp
    cdef int i
    mdp = minkowski_dot(pt, pt)
    scalar = rho / sqrt(-1 * mdp)
    for i in range(pt.shape[0]):
        pt[i] *= scalar

cdef inline exponential(double[::1] base, double[::1] tangent, double scalar, double[::1] result, double rho):
    """
    Calculate the exponential of scalar * tangent, considered as a vector
    tangent to the point `base` on the rho hyperboloid, storing the result in
    `result`.
    Pre: tangent norm is > 0
    """
    cdef double norm, arg
    cdef int i
    norm = tangent_norm(tangent)
    arg = scalar * norm / rho
    for i in range(base.shape[0]):
        result[i] = cosh(arg) * base[i] + rho * sinh(arg) * tangent[i] / norm

cdef inline double euclidean_distance(double[::1] u, double[::1] v):
    """
    Return the Euclidean distance between `u` and `v`.
    """
    cdef:
        double dist
        int k
    acc = 0
    for k in range(u.shape[0]):
        acc += (u[k] - v[k]) ** 2
    return sqrt(acc)

cdef inline _effect_update(double[::1] pt, double[::1] tangent, double scalar, double[::1] result, double rho):
    """
    Calculate the movement of scalar * tangent away from pt, storing the result
    in `result`. `rho` specifies the geometry: finite positive values
    indicate points are on tho rho hyperboloid (in its Minkowski ambient),
    while if rho == np.inf, then the points are in flat (Euclidean) space.
    """
    cdef:
        int rank
    rank = pt.shape[0]
    if rho == np.inf:
        for k in range(rank):
            result[k] += scalar * tangent[k] 
    else:
        if tangent_norm(tangent) > MIN_GRAD_NORM:  # TODO find way to avoid computing norm here and in exponential
            exponential(pt, tangent, scalar, result, rho)
            ensure_on_hyperboloid(result, rho)
