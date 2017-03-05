# environmental settings
from __future__ import print_function, division
import os

_INPUT_LOCATIONS = ['./input/', '/input/']
_OUTPUT_LOCATIONS = ['./output/', '/output/']
_FX_DATASETS = ['./fxdata']


def input_dir():
    """
    function to return the input directory
    Used to provide compatibility with running on both Floydhub and locally.
    searches the list of _INPUT_LOCATIONS and returns the first match
    """
    return _get_location('input location', _INPUT_LOCATIONS)


def output_dir():
    """
    function to return the output directory
    Used to provide compatibility with running on both Floydhub and locally.
    searches the list of _OUTPUT_LOCATIONS and returns the first match
    """
    return _get_location('output location', _OUTPUT_LOCATIONS)


def fx_dataset_dir():
    return _get_location('fx dataset dir', _FX_DATASETS)


def _get_location(name, locations):
    for loc in locations:
        if os.path.exists(loc):
            return loc
    raise IOError("Unable to find {}. Tried {}.".format(name, locations))


if __name__ == '__main__':
    print("input dir",input_dir())
    print("output dir", output_dir())
    print("fx dataset dir", fx_dataset_dir())