# First-party imports
from gluonts.dataset.common import TimeSeriesItem
from gluonts.dataset.jsonl import JsonLinesFile


def test_file(path):
    for raw_dataset in JsonLinesFile(path):
        TimeSeriesItem.parse_obj(raw_dataset.content)
    print('ok')


def run():
    import sys

    test_file(sys.argv[1])


if __name__ == '__main__':
    run()
