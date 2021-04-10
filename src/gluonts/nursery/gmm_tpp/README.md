Use GMM to fit TPP with time-varying lambda that is also a function of the past.
Let x=[x1, x2, ...] be the a series of inter-arrival time(-deltas).

 * Step 1. Fit a density function p(x1, ..., xk) = GMM(h(x1, ..., xk))
 * Step 2. Infer lambda from the nonparametric density estimation

Examples
 * Gaussian-2d
   + Generate independent x from GMM
   + Fit GMM
   + Match marginal likelihood

 * exponential-1d
   + Generate independent x from exp(lambda=0.5)
   + Fit GMM to match pdf(x)
   + Infer lambda=0.397

 * time-varying exponential (TODO)
   + Generate independent x from exp(lambda(t)=t**-0.5)
   + Fit GMM to match pdf(x)
   + Infer lambda(0) and lambda(2) separately

 * Hawkes process (TODO)
   + Generate sequential x from Hawkes with some decaying kernel
   + Fit GMM to match pdf(x1, ..., xk)
   + Infer lambda(T|x1, ..., xk) for different sequences

