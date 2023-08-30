import torch

import numpy as np

from gluonts.dataset.repository import get_dataset

from gluonts.evaluation import Evaluator

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import (
    SplicedBinnedPareto,
    SplicedBinnedParetoOutput,
)

from matplotlib import pyplot as plt


first_batch_dim = 2
second_batch_dim = 3


bins_lower_bound, bins_upper_bound = -25.0, 25.0
nbins = 11
percentile_tail = 0.05
output_dim = 1

logits = torch.tensor(
    [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0]
).unsqueeze(0)
logits = torch.tensor(
    [0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0]
).unsqueeze(0)
logits = logits.repeat(first_batch_dim, second_batch_dim, 1)


binned_distr = SplicedBinnedPareto(
    bins_lower_bound=bins_lower_bound,
    bins_upper_bound=bins_upper_bound,
    tail_percentile_gen_pareto=0.05,
    numb_bins=nbins,
    logits=logits,
    lower_gp_xi=torch.tensor(0.5, dtype=torch.float).repeat(
        first_batch_dim, second_batch_dim, 1
    ),
    lower_gp_beta=torch.tensor(3.5, dtype=torch.float).repeat(
        first_batch_dim, second_batch_dim, 1
    ),
    upper_gp_xi=torch.tensor(0.5, dtype=torch.float).repeat(
        first_batch_dim, second_batch_dim, 1
    ),
    upper_gp_beta=torch.tensor(1.0, dtype=torch.float).repeat(
        first_batch_dim, second_batch_dim, 1
    ),
)

## Getting the median, mean and mode
print("Median: ", binned_distr.median)
print("Mean: ", binned_distr.mean)
print("Mode: ", binned_distr.mode)


## Log prob
# print(torch.tensor([-1.0]).repeat(first_batch_dim, second_batch_dim).shape)
print(
    binned_distr.log_prob(
        torch.tensor([-1.0]).repeat(first_batch_dim, second_batch_dim)
    )
)


## ICDF:
quantiles = []

plot_range = np.linspace(0.0, 1.0, 300)
for q in plot_range:
    quantiles.append(
        binned_distr.icdf(torch.tensor([q]))[0, 0].detach().cpu().numpy()
    )
plt.plot(plot_range, quantiles)
plt.show()

## CDF:
quantiles = []

plot_range = range(-40, 30)
for q in plot_range:
    quantiles.append(
        binned_distr.cdf(torch.tensor([q]).repeat(2, 3))[0, 0]
        .detach()
        .cpu()
        .numpy()
    )
plt.plot(plot_range, quantiles)

plt.show()

## PDF:
quantiles = []

plot_range = range(-40, 35)
for q in plot_range:
    quantiles.append(
        binned_distr.pdf(torch.tensor([q]).repeat(2, 3))[0, 0]
        .detach()
        .cpu()
        .numpy()
    )
plt.plot(plot_range, quantiles)

plt.plot(binned_distr.mean[0, 0], 0.0, "o")
plt.plot(binned_distr.median[0, 0], 0.0, "o")
plt.plot(binned_distr.mode[0, 0], 0.0, "o")

plt.legend(["pdf", "mean", "median", "mode"])

plt.show()


### Empirical PDF from samples:

samples = binned_distr.sample(torch.tensor(range(0, 5000)).shape)

print("samples", samples.shape)

plt.hist(samples[:, 0, 0].detach().cpu().numpy(), bins=100)
plt.xlim([-45, 35])
plt.show()


### Fitting the distribution:

dataset = get_dataset("electricity")
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

prediction_length = 2  # dataset.metadata.prediction_length

print(prediction_length)

deepar_estimator_spline = DeepAREstimator(
    freq=dataset.metadata.freq,
    prediction_length=prediction_length,
    num_feat_static_cat=len(dataset.metadata.feat_static_cat),
    cardinality=[int(f.cardinality) for f in dataset.metadata.feat_static_cat],
    # distr_output=PiecewiseLinearOutput(20),
    # loss = CRPS(),
    distr_output=SplicedBinnedParetoOutput(
        bins_lower_bound=0.0,
        bins_upper_bound=100.0,
        num_bins=22,
        tail_percentile_gen_pareto=0.05,
    ),
    batch_size=2,
    trainer_kwargs={
        "gpus": -1 if torch.cuda.is_available() else None,
        "max_epochs": 5,  # Put the number of epoch here
    },
)


deepar_predictor_spline = deepar_estimator_spline.train(
    dataset.train,
    cache_data=True,
)
