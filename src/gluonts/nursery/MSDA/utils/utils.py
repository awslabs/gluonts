import pickle


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)
