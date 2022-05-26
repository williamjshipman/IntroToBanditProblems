
from typing import Union, Tuple
from scipy import stats
import numpy as np


class BayesGaussian(object):
    '''
    Use Bayesian methods to estimate the mean and variance of data.

    This class assumes that the sampled data is drawn from a normal distribution
    with an unknown mean and variance. It assumes that the unknown mean and
    variance follow a normal-gamma distribution, i.e. the prior distribution
    that encodes our belief in the mean and variance prior to observing new
    data, and the posterior distribution that is our updated belief after
    observing the data are both normal-gamma distributions. The likelihood
    function, which is the probability of observing a sample given the mean
    and variance, is a normal distribution. The normal-gamma distribution is a
    conjugate prior of the normal distribution.

    Once some data has been observed, you can retrieve the latest expected
    value of the mean and variance by calling :meth:`mean` and :meth:`variance`.
    The :meth:`percentile` method returns the score (or value) at which
    that samples are expected to have values less than or equal to that score
    with the specified probability.
    
    The posterior predictive probability distribution in this case is Student's
    t-distribution. This distribution is the probability of observing a new
    sample given the posterior distribution over the mean and variance. In
    practice, the prior and posterior distributions are over the mean and a
    random variable T. T is related to the variance:
    :math:`\sigma^2 = \frac{1}{v T}`.
    Wikipedia replaces :math:`v` with :math:`\lambda` in some pages.
    The :meth:`percentile` method uses the Student's t-distribution. The
    :meth:`mean` and :meth:`variance` methods use the normal-gamma distribution
    describing the mean and variance.

    Source: `Wikipedia <https://en.wikipedia.org/wiki/Normal-gamma_distribution>`,
    `Wikipedia Conjugate prior <https://en.wikipedia.org/wiki/Conjugate_prior>`,
    `Wikipedia Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`
    '''
    def __init__(self,
                 prior_mean: float = 0.0,
                 prior_mean_samples: int = 0,
                 prior_variance: float = 0,
                 prior_variance_samples: int = 0):
        self._mu_zero: float = prior_mean
        self._v: float = prior_mean_samples
        if prior_variance_samples <= 0:
            self._beta: Union[None, float] = None
            self._alpha: float = -0.5
        else:
            self._beta = prior_variance_samples / (2 * 1.0 / prior_variance)
            self._alpha = prior_variance_samples / 2

    def push(self, x: float, weight: float = 1.0) -> None:
        '''
        Add a new data point to the estimate of the mean and variance.
        Data points can be weighted, allowing one to down-weight

        Parameters
        ----------
        x : float
            A new data point.
        weight : float, optional
            The weight assigned to this data point, from 0.0 to 1.0,
            by default 1.0.
        '''
        if self._beta is None:
            self._beta = 0 # abs(x)
            # self._mu_zero = x
        mu_zero = (self._v * self._mu_zero + weight * x) / (self._v + weight)
        v = self._v + weight
        alpha = self._alpha + 0.5
        beta = self._beta + (weight * self._v / (self._v + weight)) * ((x - self._mu_zero) ** 2) / 2

        self._mu_zero = mu_zero
        self._v = v
        self._alpha = alpha
        self._beta = beta

    def sample_normal_gamma(self) -> Tuple[float, float]:
        T = stats.gamma.rvs(self._alpha, 1 / self._beta)
        variance = 1 / (self._v * T)
        mean = stats.norm.rvs(self._mu_zero, np.sqrt(variance))
        return mean, variance

def main() -> None:
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    true_distributions = [[110, 10], [120, 10], [130, 10], [120, 10]]
    num_actions = len(true_distributions)
    estimated_distributions = [BayesGaussian() for _ in range(num_actions)]
    reward_generators = [np.random.default_rng() for _ in range(num_actions)]
    last_action = None
    for step in range(10):
        if step < 2: # Initially take all actions twice.
            rewards = np.array([rg.normal(loc, scale) for rg, [loc, scale] in zip(reward_generators, true_distributions)])
            for r, action in zip(rewards, estimated_distributions):
                action.push(r)
            print(f'Step {step+1}: rewards {rewards}')
        else:
            estimated_mean_variance = [action.sample_normal_gamma() for action in estimated_distributions]
            # print(f'Step {step}: estimated distributions: {np.array(estimated_mean_variance)}')
            means = np.array([m for m, v in estimated_mean_variance])
            print(f'Step {step+1}: estimated means: {means}')
            action = np.argmax(means)
            reward = reward_generators[action].normal(true_distributions[action][0], true_distributions[action][1])
            estimated_distributions[action].push(reward)
            print(f'Step {step+1}: action: {action+1}, reward: {reward:0.1f}')


if __name__ == '__main__':
    main()