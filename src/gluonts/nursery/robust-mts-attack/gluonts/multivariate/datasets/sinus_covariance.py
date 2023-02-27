import numpy as np
import pandas as pd

from gluonts.multivariate.datasets.dataset import MultivariateDatasetInfo


def sinus_covariance(max_target_dim: int = 5, rank: int = 2):
    """
    For a rank 2 intrinsic noise, we can also try to learn with $\Sigma_t = U \hat{\Sigma_t} U^T$.
    Where U is drawn randomly in $\mathbb{R}^{d\times 2}$ and $$\hat{\Sigma_t} =
    \begin{bmatrix}
        \sigma_1^2 & \pho_t \sigma_1 \sigma_2 \\
        \pho_t \sigma_1 \sigma_2 & \sigma_2^2 \\
    \end{bmatrix}$$
    where $\sigma_1, \sigma_2$ are fixed constants and $\pho_t = \sin(t)$.

    """
    np.random.seed(10)

    freq = "1H"
    start = pd.Timestamp("1800-01-01", freq)
    prediction_length: int = 24
    num_freq = 1000
    num_obs_per_freq = prediction_length
    num_observations = num_freq * num_obs_per_freq
    tt = np.arange(0, num_observations) * (2 * np.pi) / num_obs_per_freq
    pho_t = np.sin(tt)

    U = np.random.uniform(size=(max_target_dim, rank)) - 0.5

    sigma_0 = 0.1

    def get_Sigma(pho):
        if rank == 2:
            S = np.array(
                [
                    [sigma_0 ** 2, sigma_0 ** 2 * pho],
                    [sigma_0 ** 2 * pho, sigma_0 ** 2],
                ]
            )
        if rank == 1:
            S = np.array(pho + 1 + sigma_0)
        # return np.dot(S, S)
        # return S
        return np.dot(np.dot(U, S), U.T)

    mu = pho_t.reshape((-1, 1)).dot(np.ones((1, max_target_dim)))

    Sigma_true = np.stack([get_Sigma(pho_t[i]) for i in range(len(tt))])
    Sigma_true /= Sigma_true.std()

    # for k in range(0, target_dim):
    #     plt.plot(Sigma_true[:100, 0, k], label=k)
    # plt.legend()
    # plt.show()

    values = np.stack(
        [
            np.random.multivariate_normal(mu[t], Sigma_true[t])
            for t in range(len(tt))
        ]
    )

    # for j in range(0, values.shape[0]):
    #    plt.plot(values[j, :72])
    # plt.tight_layout()
    # plt.savefig("artificial-timeseries.pdf")

    # values = np.hstack([values, pho_t.reshape((-1, 1))])

    values = values.transpose()

    target_dim = values.shape[0]

    train_ds = [
        {'item': '0', 'start': start, 'target': values[:, :-prediction_length]}
    ]

    test_ds = [{'item': '0', 'start': start, 'target': values}]

    return (
        MultivariateDatasetInfo(
            "artificial-sinusoidal",
            train_ds,
            test_ds,
            prediction_length,
            freq,
            target_dim,
        ),
        Sigma_true,
    )
