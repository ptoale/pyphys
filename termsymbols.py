#!/usr/bin/env python

import math
import numpy as np
import itertools
from fractions import Fraction


def n_microstates(l, n):
    """
    Determine the number of microstates for a given l and n.
    
    Args:
        l (int): The angular momentum quantum number of the orbital. Must be
                 greater than or equal to zero.
        n (int): The number of electrons that need to be placed. Must be greater
                 than zero and less than or equal to 2*(2*l+1)
    Returns:
        The number of arrangements consistent with the Pauli principle.

    >>> n_microstates(0, 1)
    2
    >>> n_microstates(0, 2)
    1
    >>> n_microstates(1, 1)
    6
    """
    # Make sure l is valid
    assert type(l) == int, "l must be an integer"
    assert l >= 0, "l must be greater than or equal to zero"

    # Calculate the number of states 
    g = 2 * (2 * l + 1)

    # Now make sure n is valid
    assert type(n) == int, "n must be an integer"
    assert 0 < n <= g, "n must be greater than zero and less than or equal to 2*(2*l+1)"

    return math.factorial(g) / (math.factorial(n) * math.factorial(g - n))


def gen_microstates(l, n, dump=False):
    """
    Generate all possible microstates of m_l and m_s consistent with the Pauli
    principle.
    
    Args:
        l (int):     The angular momentum quantum number of the orbital. Must be
                     greater than or equal to zero.
        n (int):     The number of electrons that need to be placed. Must be greater
                     than zero and less than or equal to 2*(2*l+1)
    Kwargs:
        dump (bool): Pretty print of table. Default is to be quiet.
    Returns:
        A matrix of states. Each row is a microstate and the columns are m_l, m_s.

    >>> gen_microstates(0, 1)
    matrix([[ 0. ,  0.5],
            [ 0. , -0.5]])
    >>> gen_microstates(0, 2)
    matrix([[ 0.,  0.]])
    >>> gen_microstates(1, 1)
    matrix([[-1. ,  0.5],
            [-1. , -0.5],
            [ 0. ,  0.5],
            [ 0. , -0.5],
            [ 1. ,  0.5],
            [ 1. , -0.5]])
    """

    # Make sure l is valid
    assert type(l) == int, "l must be an integer"
    assert l >= 0, "l must be greater than or equal to zero"

    # Calculate the number of m_l values and the range of values
    n_ml = 2 * l + 1
    ml_min = -l
    ml_max = l

    # Now make sure n is valid
    assert type(n) == int, "n must be an integer"
    assert 0 < n <= 2 * n_ml, "n must be greater than zero and less than or equal to 2*(2*l+1)"

    # Initialize the table
    entries = []

    if dump:
        # Print the table header
        print "Microstate Table for l=%d, n=%d:" % (l, n)
        header = ["{0:^5}|".format(i) for i in range(ml_min, ml_max + 1)]
        header.insert(0, "|")
        print ("|{0:^" + repr(n_ml * 6 - 1) + "}|{1:^5}|{2:^5}|").format('m_l', 'M_l', 'M_s')
        print "".join(header) + '     |' * 2
        print "|" + "-----|" * n_ml + '-----|' * 2

    # Iterate over all assignments of n electrons to 2*(2*l+1) states
    # Each i is a tuble specifing the states of the electrons
    # For example, l=1, n=1 gives (0,),(1,),(2,),(3,),(4,),(5,)
    #       m_l = floor(x/2) - l                  --> (-1,   -1,    0,    0,   +1,   +1)
    #       m_s = 0.5 for x even, -0.5 for x odd  --> (+0.5, -0.5, +0.5, -0.5, +0.5, -0.5)
    for i in itertools.combinations(range(0, 2 * n_ml), n):
        Ml = sum([math.floor(x / 2.0) - l for x in i])
        Ms = sum([0.5 if x % 2 == 0 else -0.5 for x in i])

        # Add it to the table
        entries.append([Ml, Ms])

        if dump:
            # Format spin as either U D, U, D, or none
            labels = ['U D' if (2 * x in i and 2 * x + 1 in i)
                      else 'U' if (2 * x in i)
            else 'D' if (2 * x + 1 in i)
            else '' for x in range(0, n_ml)]
            labels = ['{0:^5}|'.format(x) for x in labels]
            print '|' + ''.join(labels) + '{0:^5}|{1:^5}|'.format(Ml, Ms)

    return np.matrix(entries)


def gen_macrostates(l, n, dump=False):
    """
    Generate all possible macrostates of M_l and M_s consistent with the Pauli
    principle.
    
    Args:
        l (int):     The angular momentum quantum number of the orbital. Must be
                     greater than or equal to zero.
        n (int):     The number of electrons that need to be placed. Must be greater
                     than zero and less than or equal to 2*(2*l+1)
    Kwargs:
        dump (bool): Pretty print of table. Default is to be quiet.
    Returns:
        A matrix of states. Rows label M_l and columns label M_s

    >>> gen_macrostates(0, 1)
    matrix([[1, 1]])
    >>> gen_macrostates(0, 2)
    matrix([[0, 1, 0]])
    >>> gen_macrostates(1, 1)
    matrix([[1, 1],
            [1, 1],
            [1, 1]])
    """

    # We need the table of microstates, so we'll let that function valid the inputs
    microstates = gen_microstates(l, n, dump=dump)

    # Find the max M_L
    max_Ml = 0
    for s in microstates.getA():
        max_Ml = max(max_Ml, int(s[0]))

    # Initialize the table
    tm = [[0 for i in range(0, 2 * n + 1, 2)] for i in range(2 * max_Ml + 1)]

    if dump:
        # Print the table header
        print "\nMacrostate Table for l=%d, n=%d:" % (l, n)
        header = ["{0:^5}|".format(i / 2.0) for i in range(-n, n + 1, 2)]
        print "|M_l\M_s |" + ''.join(header)


    # Iterate over the M_l's
    for i in range(-max_Ml, max_Ml + 1):
        t = []

        # Iterate over the M_s's
        for j in range(-n, n + 1, 2):
            # Add the number of configurations that match this M_l and M_s
            t.append(len([x for x in microstates.getA() if x[0] == i and x[1] == j / 2.0]))

            ii = i + max_Ml
            jj = int((j + n) / 2.0)
            tm[ii][jj] = len([x for x in microstates.getA() if x[0] == i and x[1] == j / 2.0])

        if dump:
            print "|{0:^8}|".format(i) + ''.join(["{0:^5}|".format(x) for x in t])

    return np.matrix(tm)


def term_name(i, j):
    if i == 0:
        name = 'S'
    elif i == 1:
        name = 'P'
    elif i == 2:
        name = 'D'
    elif i == 3:
        name = 'F'
    elif i == 4:
        name = 'G'
    elif i == 5:
        name = 'H'
    elif i == 6:
        name = 'I'
    elif i == 7:
        name = 'K'
    elif i == 8:
        name = 'L'
    elif i == 9:
        name = 'M'
    elif i == 10:
        name = 'N'
    elif i == 11:
        name = 'O'
    elif i == 12:
        name = 'Q'
    elif i == 13:
        name = 'R'
    elif i == 14:
        name = 'T'
    else:
        name = 'X'

    return name + str(j)


def gen_sequence(L, S, dump=False):
    """
    Generate a sequence of terms. This could even be an iterator.
    
    Args:
        L (int):     The maximum total angular momentum quantum number (L>0)
        S (float):   The maximum total spin quantum number (S=n*1/2)
    Kwargs:
        dump (bool): Dump some details.
    Returns:
        A list of possible terms.
    
    >>> gen_sequence(0, 0.5)
    ([array([[1, 1]])], ['S2'])
    >>> gen_sequence(1, 0.5)
    ([array([[1, 1],
           [1, 1],
           [1, 1]]), array([[0, 0],
           [1, 1],
           [0, 0]])], ['P2', 'S2'])
    >>> gen_sequence(0, 1.0)
    ([array([[1, 1, 1]]), array([[0, 1, 0]])], ['S3', 'S1'])
    
    """

    terms = []
    names = []

    S_mult = int(2 * S + 1)
    i_max = 2 * L + 1
    j_max = S_mult

    for j in range(S_mult, 0, -2):
        j_size = j
        for i in range(L, -1, -1):
            i_size = 2 * i + 1
            i_off = (i_max - i_size) / 2
            j_off = (j_max - j_size) / 2

            name = term_name((i_size - 1) / 2, j_size)
            term = np.zeros((i_max, j_max), dtype=int)
            term[i_off:i_max - i_off, j_off:j_max - j_off] = term[i_off:i_max - i_off, j_off:j_max - j_off] + np.ones(
                (i_max - 2 * i_off, j_max - 2 * j_off), dtype=int)

            if dump:
                print "L=%d, SM=%d size=(%d,%d) off=(%d,%d) name=%s" % (i, j, i_size, j_size, i_off, j_off, name)
                print term

            terms.append(term)
            names.append(name)

    return terms, names


def find_terms(l, n, dump=False):
    # We need the macrostate table
    macrostates = gen_macrostates(l, n, dump=dump)
    print macrostates

    # Next we need to find the max M_l and M_s
    sh = macrostates.shape
    max_Ml = (sh[0] - 1) / 2
    max_Ms = (sh[1] - 1) / 2.
    if dump:
        print "\nFinding Terms for max L=%d, S=%2.2f" % (max_Ml, max_Ms)

    seq, names = gen_sequence(max_Ml, max_Ms, dump=dump)
    m0 = macrostates
    for t, n in zip(seq, names):
        m1 = m0 - t
        if np.amin(m1) >= 0:
            m0 = m1
            print "Found term: %s" % n


# print t


class TermSymbol(object):
    """
    Class representing a term symbol


    >>> print TermSymbol(0, 0.5, 0.5)
    2_S_1/2
    >>> print TermSymbol(1, 1, 2)
    3_P_2
    """

    to_string = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I'}

    def __init__(self, L, S, J):
        self.L = L
        self.S = S
        self.J = J


    def gen_table(self, ts=None):
        """

        :param ts: A term symbol to define the size of the returned table.
        :return: A numpy array representing the M_l/M_s table for this symbol.

        >>> ts = TermSymbol(1, 1, 2)
        >>> print ts.gen_table()
        [[1 1 1]
         [1 1 1]
         [1 1 1]]

        """
        n_row = int(2 * self.L + 1)
        n_col = int(2 * self.S + 1)
        return np.ones((n_row, n_col), dtype=int)


    def __str__(self):
        return "%d_%s_%s" % (2 * self.S + 1, self.to_string[self.L], Fraction(self.J))


if __name__ == '__main__':
    import argparse
    import doctest

    doctest.testmod()

    # Setup the parser
    parser = argparse.ArgumentParser(description='Term Symbol Generator')
    parser.add_argument('l', type=int, default=0,
                        help='Angular momentum quantum number (l>=0)')
    parser.add_argument('n', type=int, default=1,
                        help='Number of electrons (0<n<=2(2l+1)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug flag')

    # Parse the command line
    args = parser.parse_args()

    # Check that the inputs are valid
    if args.l < 0:
        parser.error("l must be greater than or equal to zero")
    if args.n < 1 or args.n > 2 * (2 * args.l + 1):
        parser.error("n must be greater than zero and less than or equal to 2(2l+1)")

# N = n_microstates(args.l, args.n)
#    microstates = gen_microstates(args.l, args.n, dump=args.debug)
#    macrostates = gen_macrostates(args.l, args.n, dump=args.debug)
#    gen_sequence(2, 1.5, True)
#    find_terms(args.l, args.n, dump=args.debug)
