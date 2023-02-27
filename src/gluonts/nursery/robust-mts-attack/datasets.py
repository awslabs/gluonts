from gluonts.multivariate.datasets.dataset import electricity, exchange_rate, solar, traffic, wiki, taxi_30min

DATASETS = ['electricity', 'exchange_rate', 'solar', 'traffic', 'wiki', 'taxi']

def get_dataset(dataset, max_target_dim):
    if dataset == 'electricity':
        return electricity(max_target_dim=max_target_dim)
    elif dataset == 'exchange_rate':
        return exchange_rate(max_target_dim=max_target_dim)
    elif dataset == 'solar':
        return solar(max_target_dim=max_target_dim)
    elif dataset == 'traffic':
        return traffic(max_target_dim=max_target_dim)
    elif dataset == 'wiki':
        return wiki(max_target_dim=max_target_dim)
    elif dataset == 'taxi':
        return taxi_30min(max_target_dim=max_target_dim)

