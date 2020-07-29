import pandas as pd
from io import StringIO


def store_csv(file, result):
    with open(file, 'a') as f:
        f.write(result)
        f.write('\n')
        f.close()


def store_pd(file, result):
    StringData = StringIO(result)
    df = pd.read_csv(StringData, sep=",")
    df.to_pickle(file)