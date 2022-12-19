import time
import math

"""
This is from Pytorch tutorial (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
"""


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


if __name__ == "__main__":

    main()
