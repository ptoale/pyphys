#!/usr/bin/env python
"""

http://stackoverflow.com/questions/6284396
"""
import itertools

class unique_element:
    def __init__(self, value, occurrences):
        self.value = value
        self.occurrences = occurrences
        
def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

def macrostates(n, E):

    result = {}
    for i in itertools.combinations_with_replacement(range(0,E+1), n):
        if sum(i) == E:
            W = len(list(perm_unique(i)))
            print "Macrostate: ", i, " W =", W
            result[i] = W
#            for j in perm_unique(i):
#                print "  Microstate: ", j
    return result

def probs(n, E):
    p = []

    s = macrostates(n, E)
    
    Wsum = 0
    for i, j in s.iteritems():
        Wsum += j
#    print "Number of microstates = %d" % Wsum

    for i in range(E+1):
#        print "Prob for energy %d" % i
        sum = 0
        for j in s:
#            print j, j.count(i), s[j]
            sum += j.count(i)*s[j]
        p.append(float(sum)/(n*Wsum))
        print float(sum)/(n*Wsum)

    return p

def plot(n, E):
    import matplotlib.pyplot as plt

    e = list(range(E+1))
    p = probs(n, E)    

    plt.plot(e, p)
    plt.show()

if __name__ == '__main__':
    import argparse
    import doctest

    doctest.testmod()

    # Setup the parser
    parser = argparse.ArgumentParser(description='Oscillator-Like Particle Statistics')
    parser.add_argument('-n', type=int, default=1,
                        help='Number of particles (n>0)')
    parser.add_argument('-E', type=int, default=1,
                        help='Total energy (E>0)')
    parser.add_argument('-t', choices=['Classical', 'Bosonic', 'Fermionic'], default='Classical',
                        help='Type of particle')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug flag')

    # Parse the command line
    args = parser.parse_args()

    # Check that the inputs are valid
    if args.n < 1:
        parser.error("n must be greater than or equal to 1")
    if args.E < 1:
        parser.error("E must be greater than or equal to 1")

    plot(args.n, args.E)