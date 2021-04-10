Use GMM to fit TPP with time-varying lambda that is also a function of the past.
Let [d1, d2, ...] be the a series of inter-arrival time(-deltas).

 * Step 1. Fit a density function p(d1, d2) = GMM(h(d1, d2))
 * Step 2. Extract lambda from the density estimation

Provide exponential-1d and Gaussian-2d examples.
