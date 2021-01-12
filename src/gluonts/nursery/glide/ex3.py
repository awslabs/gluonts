from gluonts.nursery.glide import Map, partition
from gluonts.nursery.glide import Apply


xs = range(100)

xs2 = Map(lambda n: n + 1, Map(lambda n: n * 2, xs))


print(list(Apply(lambda x: x, partition(xs2, 3))))
