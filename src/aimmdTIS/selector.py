
import logging
import numpy as np
from openpathsampling import ShootingPointSelector

logger = logging.getLogger(__name__)

class UniformRCModelSelector(ShootingPointSelector):
    """
    Select uniformly in q-space which is the space represented by the RC model.

    Parameters
    ----------
    model - :class:`aimmd.base.RCModel` a wrapped model predicting RC values
    states - list of :class:`openpathsampling.Volume`, one for each state
    distribution - string specifying the SP selection distribution,
                   either 'gaussian', 'lorentzian' or 'uniform_q'
                   'gaussian': p_{sel}(x) ~ exp(-alpha * z_{sel}(x)**2)
                   'lorentzian': p_{sel}(x) ~ gamma**2
                                              / (gamma**2 + z_{sel}(x)**2)
                    'uniform_q': p_{sel}(x) ~ (1/N_{bins})* 1/(H_{bin}[x])                                             

    scale - float, 'softness' parameter of the selection distribution,
            higher values result in a broader spread of SPs around the TSE,
            1/alpha for 'gaussian' and gamma for 'lorentzian'
    density_adaptation - bool, whether we try to correct for imbalances in
                         the density of points on TPs to achieve a more uniform
                         SP density along the reaction coordinate,
                         NOTE: updating the density estimate needs to be
                         enabled by adding the aimmd.ops.DensityCollectionHook
                         for this feature to have an effect

    Notes
    -----
    We use the z_sel function of the RCModel as input for the SP selection.

    """

    def __init__(self, model, states=None, distribution='uniform',
                 scale=1., density_adaptation=True):
        """Initialize a RCModelSelector."""
        super(UniformRCModelSelector, self).__init__()
        self.model = model
        if states is None:
            logger.warning('Consider passing the states to speed up'
                           + ' accepting/rejecting.')
        self.states = states
        self.distribution = distribution
        self.scale = scale
        self.density_adaptation = density_adaptation
        # self.q_bins = np.linspace(-50,50,201)
        self.q_bins = 100


    @classmethod
    def from_dict(cls, dct):
        """Will be called by OPS when loading self from storage."""
        # atm we set model = None,
        # since we can not arbitrary models in OPS storages
        # but if used with an aimmd.TrainingHook it will (re)set the model
        # to the one saved besides the OPS storage
        obj = cls(None,
                  dct['states'],
                  distribution=dct['distribution'],
                  scale=dct['scale'],
                  # this should make selectors saved before adding
                  # denisty_adaptadtion work as intended
                  density_adaptation=dct.get('density_adaptation', False)
                  )
        logger.warning('Restoring RCModelSelector without model.'
                       + 'Please take care of resetting the model yourself.')
        return obj

    def to_dict(self):
        """Will be called by OPS when saving self to storage."""
        dct = {}
        dct['distribution'] = self._distribution
        dct['scale'] = self.scale
        dct['states'] = self.states
        dct['density_adaptation'] = self.density_adaptation
        return dct

    @property
    def distribution(self):
        """Return the name of the shooting point selection distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, val):
        if val == 'gaussian':
            self._f_sel = lambda z: self._gaussian(z)
            self._biases = self._biases_for_q_function
            self._distribution = val
        elif val == 'lorentzian':
            self._f_sel = lambda z: self._lorentzian(z)
            self._biases = self._biases_for_q_function
            self._distribution = val
        elif val == "uniform":
            self._biases = self._biases_uniform_in_q
            self.f = self.f_uniform
            print("changed biases function I think")
            print(self._biases)
            self._distribution = val
        else:
            raise ValueError('Distribution must be one of: '
                             + '"gaussian", "lorentzian" or "uniform"')

    def _lorentzian(self, z):
        return self.scale / (self.scale**2 + z**2)

    def _gaussian(self, z):
        return np.exp(-z**2/self.scale)

    def f(self, snapshot, trajectory):
        """Return the unnormalized proposal probability of a snapshot."""
        z_sel = self.model.z_sel(snapshot)
        any_nan = np.any(np.isnan(z_sel))
        if any_nan:
            logger.warning('The model predicts NaNs. '
                           + 'We used np.nan_to_num to proceed')
            z_sel = np.nan_to_num(z_sel)
        # casting to python float solves the problem that
        # metropolis_acceptance is not saved !
        ret = float(self._f_sel(z_sel))
        if self.density_adaptation:
            committor_probs = self.model(snapshot)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = self.model.density_collector.get_correction(
                                                            committor_probs
                                                                       )
            ret *= float(density_fact)
        if ret == 0.:
            if self.sum_bias(trajectory) == 0.:
                return 1.
        return ret  

    def f_uniform(self, snapshot, trajectory):
        z_sels = self.model.z_sel(trajectory)
        z_sel = self.model.z_sel(snapshot)
        # TODO: this can be saved for the given trajectory to reduce computation time 
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.warning('The model predicts NaNs. '
                            'We used np.nan_to_num to proceed')
            print("the model output including NaNs: ", z_sels)
            z_sels = np.nan_to_num(z_sels)
            z_sel = np.nan_to_num(z_sel)
        hist_q, bins= self._histogram_q(z_sels)
        # print(z_sels)
        idx_z_sel = np.digitize(z_sel, bins[:-1])
        # print(idx_z_sel)
        N_bins_filled = np.count_nonzero(hist_q)
        # print(N_bins_filled)
        # print(hist_q[idx_z_sel-1])
        ret = (1/N_bins_filled)*(1/(hist_q[idx_z_sel-1]))
        return ret


    def probability(self, snapshot, trajectory):
        """Return proposal probability of the snapshot for this trajectory."""
        if self.states is not None:
            # only evaluate costly symmetry functions if needed,
            # if trajectory is no TP it has weight 0 and p_pick = 0 for all points
            self_transitions = [1 < sum(s(p)
                                        for p in [trajectory[0], trajectory[-1]])
                                for s in self.states]
            # trajectory is a TP if it has no self-transitions -> calculate p_pick
            if any(self_transitions):
                return 0.
        sum_bias = self.sum_bias(trajectory)
        # print(sum_bias)
        # print(self.f(snapshot, trajectory))
        if sum_bias == 0.:
            return 1./len(trajectory)
        return self.f(snapshot, trajectory) / sum_bias

    def sum_bias(self, trajectory):
        """
        Return the partition function of proposal probabilities for trajectory.
        """
        # casting to python float solves the problem that
        # metropolis_acceptance is not saved !
        return float(np.sum(self._biases(trajectory)))
    
    def _biases_for_q_function(self, trajectory):
        z_sels = self.model.z_sel(trajectory)
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.warning('The model predicts NaNs. '
                           'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        ret = self._f_sel(z_sels)
        if self.density_adaptation:
            committor_probs = self.model(trajectory)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = self.model.density_collector.get_correction(
                                                            committor_probs
                                                                       )
            ret *= density_fact
        return ret

    def _biases_uniform_in_q(self, trajectory):
        z_sels = self.model.z_sel(trajectory)
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.warning('The model predicts NaNs. '
                           'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        # TODO this is now computed multiple times:
        hist_q, bins= self._histogram_q(z_sels)
        # print("bias function")
        # print(z_sels)
        # print(bins)
        # print(hist_q)
        # print(np.shape(hist_q))
        # print(np.shape(bins))
        idx_z_sel = np.digitize(z_sels, bins[:-1])
        # print(idx_z_sel)
        N_bins_filled = len(np.unique(idx_z_sel))
        ret = (1/N_bins_filled)*(1/(hist_q[idx_z_sel-1]))
        return ret

    def _histogram_q(self, z_sels):
        """
        Return the histogram bins and counts for z_sels the q-values from a given trajectory.
        """
        hist_q, q_bins = np.histogram(z_sels,bins=self.q_bins)
        return hist_q, q_bins
        
    def pick(self, trajectory):
        """Return the index of the chosen snapshot within trajectory."""
        biases = self._biases(trajectory)
        sum_bias = np.sum(biases)
        if sum_bias == 0.:
            logger.error('Model not able to give educated guess.\
                         Choosing based on luck.')
            # we can not give any meaningfull advice and choose at random
            return np.random.randint(len(trajectory))

        rand = np.random.random() * sum_bias
        idx = 0
        prob = biases[0]
        while prob <= rand and idx < len(biases) - 1:
            idx += 1
            prob += biases[idx]

        # let the model know which SP we chose
        self.model.register_sp(trajectory[idx])
        # print("Chosen snapshot {}".format(idx))
        return idx
