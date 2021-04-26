#cython: boundscheck=True, wraparound=False, language_level=3, initializedcheck=True
import numpy as np
import logging


cdef extern from "math.h":
    double sqrt(double)
    double log(double)
    double exp(double)
    double cosh(double)
    double sinh(double)
    double acosh(double)

np.seterr(all='raise')  # floating point errors will be raised as exceptions

cdef double MU = 4. / 3.  # for JC69 (which is assumed throughout)
cdef double MIN_GRAD_NORM = 1e-12  # minimum gradient norm for normalisation to be attempted
cdef double MIN_DISTANCE = 1e-5  # the minimum distance b/w two points (gradient is singular at distance 0).


cdef class LogalikeOptimiser:
    cdef:
        public double[:,::1] points
        public double[:,::1] gradients
        public double[::1] partial_lals
        public int rounds_done
        public int local_dim
        public int num_points
        public double lr
        public double rho
        public double max_step_size
        double[:,::1] rates_of_difference

    def __init__(self, rho, lr, max_step_size=0.25):
        """
        If `rho` > 0, it is interpreted as the "radius" of the hyperboloid.
        `rho` = np.inf means use Euclidean space.
        """
        self.rho = rho
        self.lr = lr
        self.rounds_done = 0
        self.local_dim = -1
        self.num_points = -1
        self.max_step_size = max_step_size
        self.points = None

    # FOR PICKLING PROTOCOL
    def __getstate__(self):
        return (self.rho, self.lr, self.rounds_done, self.local_dim, self.num_points, self.max_step_size, np.array(self.points))

    def __setstate__(self, state):
        rho, lr, rounds_done, local_dim, num_points, max_step_size, points = state
        self.rho = rho
        self.lr = lr
        self.rounds_done = rounds_done
        self.local_dim = local_dim
        self.num_points = num_points
        self.max_step_size = max_step_size
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
        self.partial_lals = np.zeros((self.num_points,), dtype=np.float64)

    def get_points(self):
        """
        Return the points as a Numpy array.
        """
        if self.points is None:
            return None
        return np.array(self.points)
        
    def get_gradients(self):
        return np.array(self.gradients)

    def get_partial_lals(self):
        return np.array(self.partial_lals)

    def fit(self,
            double[:,::1] rates_of_difference,
            double stopping_distance,
            int max_rounds,
            int checkpoint_interval):
        """
        Performs at most `max_rounds` rounds of updates (each round updates
        every point) or continues without such a limit if set to -1.
        """
        cdef:
            double ll
            double scaled_lr

        checkpoints = []
        def checkpoint(step_sizes, gradient_norms):
            checkpoints.append((np.array(self.points).copy(),
                                self.rounds_done,
                                np.array(self.partial_lals).sum(),
                                np.array(self.partial_lals).copy(),
                                step_sizes.copy(),
                                gradient_norms.copy()))

        self.rates_of_difference = rates_of_difference

        step_sizes = np.zeros((self.num_points,), dtype=np.float64)
        gradient_norms = np.zeros((self.num_points,), dtype=np.float64)

        self.rounds_done = 0
        while max_rounds == -1 or self.rounds_done < max_rounds:
            will_checkpoint = (self.rounds_done % checkpoint_interval == 0)
            try:
                if will_checkpoint:
                    # calculate the gradient norms
                    for i in range(self.num_points):
                        self.partial_logalike_and_gradient(i)
                        gradient_norms[i] = tangent_norm(self.gradients[i,:])

                for i in range(self.num_points):
                    self.partial_logalike_and_gradient(i)
                    # gradient descent : just step with -1 * lr * gradient

                    norm  = tangent_norm(self.gradients[i,:])
                    # clip gradient if necessary to ensure self.max_step_size is not exceeded
                    # we do this by scaling the learning rate (which is equivalent)
                    scaled_lr = self.lr
                    if norm * scaled_lr > self.max_step_size:
                        scaled_lr = self.max_step_size / norm
                    step_sizes[i] = scaled_lr * norm

                    if self.rho == np.inf:
                        for k in range(self.local_dim + 1):
                            self.points[i,k] += scaled_lr * self.gradients[i,k] 
                    else:
                        # skip if the gradient it too small, for numerical stability
                        if norm < MIN_GRAD_NORM:
                            step_sizes[i] = 0.
                            continue
                        # normalise the gradient
                        for k in range(self.local_dim + 1):
                            self.gradients[i,k] /= norm
                        normalised_exponential(self.points[i,:], self.gradients[i,:], scaled_lr * norm, self.points[i,:], self.rho)
                        ensure_on_hyperboloid(self.points[i,:], self.rho)

            except (FloatingPointError, ZeroDivisionError):
                logging.exception('fit() caught exception')
                # NB the only reason we are catching this exception and
                # returning a failure flag is because we want to be able to
                # call fit() from joblib parallel, which can't handle
                # exceptions.
                return False, True, checkpoints

            if will_checkpoint:
                checkpoint(step_sizes, gradient_norms)

            # the maximum of the step sizes determines whether to stop
            largest_step = np.max(step_sizes)
            if (largest_step < stopping_distance):
                # stopping condition is satified
                return True, False, checkpoints
            self.rounds_done += 1
        return False, False, checkpoints

    cpdef int partial_logalike_and_gradient(LogalikeOptimiser self, int i) except -1:
        """
        Calculate the summand of the log-a-like function
        corresponding to the ith point and the gradient with
        respect to the ith point, writing these to the instance
        variables.
        The gradient is a vector in the hyperboloid tangent
        space of the ith point.
        """
        cdef:
            int j, k
            double dist
            double d_dist_wrt_mdp, d_log_proba
            double mdp
            double p_off_diag, p_diag
            double decay
            double r
            double scalar

        # set to zero
        for k in range(self.local_dim + 1):
            self.gradients[i, k] = 0
        self.partial_lals[i] = 0

        for j in range(self.num_points):
            if i == j:
                continue
            if self.rho == np.inf:
                dist = euclidean_distance(self.points[i,:], self.points[j,:])
            else:
                dist = hyperboloid_distance(self.points[i,:], self.points[j,:], self.rho)

            # can't allow distances to get too close to zero, for numerical stability
            if dist < MIN_DISTANCE:
                dist = MIN_DISTANCE

            r = self.rates_of_difference[i, j]
            decay = exp(-MU * dist)
            p_diag = 0.25 + 0.75 * decay
            p_off_diag = 0.25 * (1 - decay)
            self.partial_lals[i] += r * log(p_off_diag) + (1 - r) * log(p_diag)
            d_log_proba = decay * ((r / 3) / p_off_diag - (1 - r) / p_diag)

            if self.rho == np.inf:
                scalar = 2 * d_log_proba / dist
                for k in range(self.local_dim + 1):
                    self.gradients[i, k] += scalar * (self.points[i, k] - self.points[j, k])
            else:
                mdp = -1 * self.rho ** 2 * cosh(dist / self.rho)
                d_dist_wrt_mdp = -1 / sqrt((mdp / self.rho) ** 2 - self.rho ** 2)
                scalar = d_dist_wrt_mdp * d_log_proba
                for k in range(self.local_dim + 1):
                    self.gradients[i, k] += scalar * self.points[j, k]

        if self.rho != np.inf:
            project_onto_tangent_space(self.points[i,:], self.gradients[i,:], self.rho)


# Note: these helpers don't need to be callable from Python - so long as we can
# extract the vectors from the Cythonized sequence MDS (which _does_ need to be
# Python callable) then we can use the Python Hyperboloid class for evaluation
# calculations.
# Indeed it is better to use the Python language Hyperboloid class for
# everything that is not fitting (e.g. init), since its functions are tested.

cdef inline double minkowski_dot(double[::1] u, double[::1] v):  # no except here (but FP errors are unlikely)
    cdef int i, rank
    cdef double result = 0
    rank = u.shape[0]
    for i in range(rank - 1):
        result += u[i] * v[i]
    result -= u[rank - 1] * v[rank - 1]
    return result

cdef inline double tangent_norm(double[::1] tangent) except -1:
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

cdef inline double hyperboloid_distance(double[::1] u, double[::1] v, double rho) except -1:
    """
    Return the distance between the two provided points on the
    rho hyperboloid.
    """
    cdef double arccosh_arg, mdp
    mdp = minkowski_dot(u, v)
    arccosh_arg = -1 * mdp / (rho ** 2)
    if arccosh_arg < 1:
        arccosh_arg = 1
    return rho * acosh(arccosh_arg)

cdef inline int project_onto_tangent_space(double[::1] hyperboloid_point, double[::1] minkowski_tangent, double rho) except -1:
    """
    """
    cdef double scalar
    cdef int i
    scalar = minkowski_dot(hyperboloid_point, minkowski_tangent) / (rho ** 2)
    for i in range(hyperboloid_point.shape[0]):
        minkowski_tangent[i] += scalar * hyperboloid_point[i]

cdef inline int ensure_on_hyperboloid(double[::1] pt, double rho) except -1:
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

cdef inline int normalised_exponential(double[::1] base, double[::1] tangent, double step_size, double[::1] result, double rho) except -1:
    """
    Calculate the exponential of scalar * tangent, considered as a vector
    tangent to the point `base` on the rho hyperboloid, storing the result in
    `result`.
    Pre: tangent is a unit vector
    """
    cdef double arg
    cdef int i
    arg = step_size / rho
    for i in range(base.shape[0]):
        result[i] = cosh(arg) * base[i] + rho * sinh(arg) * tangent[i]

cdef inline double euclidean_distance(double[::1] u, double[::1] v) except -1:
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
