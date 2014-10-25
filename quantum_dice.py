#!/usr/bin/env python
"""
Calculates the probability table for all possible outcomes of rolling
'n' die with 'd' sides, which obey either classical, bosonic, or fermionic
statistics.

"""
import itertools

def throw_table(n, d=6, type='classical'):
    """
    Produce all possible outcomes of n d-sided dice, which obey the statistics
    of a particle of a given type.
    
    Arguments:
        n (int): The number of dice
        d (int): The number of sides on the dice
        type (str): The type of statistics [classical, bosonic, fermionic]
        
    Returns:
        A list of allowed throws
        
    >>> throws_classical = throw_table(2, d=6, type='classical')
    >>> len(throws_classical)
    36
    >>> throws_bosonic = throw_table(2, d=6, type='bosonic')
    >>> len(throws_bosonic)
    21
    >>> throws_fermionic = throw_table(2, d=6, type='fermionic')
    >>> len(throws_fermionic)
    15
    
    """
    table = None
    roll = range(1, d+1)
    
    if type == 'classical':
        table = list(itertools.product(roll, repeat=n))
    else:
        table = list(itertools.combinations(roll, n))
        if type == 'bosonic':
            # TODO: This only works for 2 dice!!!!
            for i in roll:
                table.append((i,i))

    return table

def prob(throw, n, d=6, type='classical'):
    """
    Calculate the probability of a given throw.
    
    Arguments:
        throw (int): A throw
        n (int): The number of dice
        d (int): The number of sides on the dice
        type (str): The type of statistics [classical, bosonic, fermionic]
    
    Returns:
        The probability of the throw, based on the other inputs.
    
    >>> prob(7, 2, d=6, type='classical')
    0.16666666666666666
    >>> prob(7, 2, d=6, type='bosonic')
    0.14285714285714285
    >>> prob(7, 2, d=6, type='fermionic')
    0.2
        
    """
    count = 0
    table = throw_table(n, d, type)
    for t in table:
        if sum(t) == throw:
            count += 1
            
    return float(count)/len(table)

if __name__ == '__main__':
    import doctest
    import argparse
    doctest.testmod()

    # Setup the parser
    parser = argparse.ArgumentParser(description='Quantum Dice')
    parser.add_argument('n', type=int,
                        help='Number of dice')
    parser.add_argument('-d', type=int, default=6,
                        help='Number of sides [6]')
    parser.add_argument('-t', choices=['classical', 'bosonic', 'fermionic'], default='classical',
                        help='Type of statistics')

    # Parse the command line
    args = parser.parse_args()

    min = args.n
    max = args.n*args.d
    for i in range(min, max+1):
        p = prob(i, args.n, d=args.d, type=args.t)
        print "p(%d) = %f" % (i, p)
