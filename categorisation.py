import numpy as np
import pandas as pd

def _boundary_quantiles(q):
    """
    :param q: number of categories
    :return:

    >>> _boundary_quantiles(3)
    array([ 0.33333333])
    >>> _boundary_quantiles(5)
    array([ 0.2,  0.6])
    >>> _boundary_quantiles(7)
    array([ 0.14285714,  0.42857143,  0.71428571])

    """
    assert q%2 ==1, "only odd q supported"
    # only works for odd q at present

    bds = np.linspace(0,1,q+1)
    idx = filter(lambda x: x%2 == 1, np.linspace(0,q, q+1).astype(int))[:-1]
    return bds[idx]



def rolling_qcut(ser, q, window):
    """
    creates quantile based categories.
    Care is taken to ensure that directional information is not lost, so the make catgories are always symmetrical.
    - take abs of returns to  find quantiles, then consider these as both +ve and -ve limits.
    - this gives twice as many categories as required. so every other boundary is then skipped.

    e.g. if 5 categories:
    +e  (4)
    +d  (3)
    ----
    +c  (2)
    +b  (1)
    ----
    +a  (0)
    ..............................
    -a
    -----
    -b
    -c
    -----
    -d
    -e
    categories are: (<-c), (>-c & < -a), (>-a, < +a), (>+a & < +c) (>+c)

    0 1 2 3
      1

    0 1 2 3 4 5
      1 . 3 .

    0 1 2 3 4 5 6 7
      1 . 3 . 5 .

    """

    # can do something clever with linspace and list indexing to get just the required boundaries


    df = pd.DataFrame(index=ser.index)
    quantiles = _boundary_quantiles(q)
    print "quantiles:", quantiles

    for qt in quantiles:
        boundary = ser.abs().rolling(window).quantile(qt)
        df["{}".format(qt)] = boundary
        df["-{}".format(qt)] = -boundary
    sorted_cols = sorted([float(c) for c in df.columns])
    print df[sorted_cols].dropna().mean()
    return df

if __name__ == '__main__':
    import doctest
    doctest.testmod()