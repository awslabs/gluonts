import matplotlib.pyplot as plt
import pandas as pd
import numpy


def object_plot(estimator):
    df = pd.read_csv("t.csv")
    df.columns = ['ncalls','tottime','percall','cumtime','percall','file','dataset','estimator']
    df1 = df[df['estimator'] == estimator]
    time = list(df1.tottime)  # gets only the column related to the number of times methods are called
    objects = list(df1.file)
    x_axis = list(set(objects))
    datasets = list(df1.dataset)
    result = {}
    for i in range(len(time)):
        element = objects[i]
        idx = x_axis.index(element)
        t = time[i]
        prev = result.get(datasets[i], [0] * len(x_axis))
        prev[idx] = t
        result[datasets[i]] = prev
    for d, ts in result.items():
        total = sum(ts)
        ts_per = [ti / total * 100 if total else 0 for ti in ts]
        result[d] = ts_per
        plt.plot(x_axis, ts_per, marker='o', color=numpy.random.rand(3,), label=d)
    plt.ylabel('time consumption %')
    plt.xlabel('transformation')
    plt.title(estimator+' transformation')
    plt.legend()
    plt.xticks(rotation=30)
    plt.savefig('wwwww', bbox_inches='tight')
    plt.show()




