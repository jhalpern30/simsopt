#!/usr/bin/env python3

"""
This module contains small utility functions and classes.
"""

import numpy as np

def isbool(val):
    """
    Test whether val is any boolean type, either the native python
    bool or numpy's bool_.
    """
    return isinstance(val, bool) or isinstance(val, np.bool_)

def isnumber(val):
    """
    Test whether val is any kind of number, including both native
    python types or numpy types.
    """
    return isinstance(val, int) or isinstance(val, float) or \
        isinstance(val, np.int_) or isinstance(val, np.float)

class Struct():
    """
    This class is just a dummy mutable object to which we can add attributes.
    """

class Identity():
    """
    This class represents a term in an objective function which is just
    the identity. It has one degree of freedom, and the output of the function
    is equal to this degree of freedom.
    """
    def __init__(self, x=0.0):
        self.x = x
        self.fixed = np.full(1, False)
        self.names = ['x']

    def J(self):
        return self.x

    @property
    def f(self):
        """
        Same as the function J(), but a property instead of a function.
        """
        return self.x
    
    def get_dofs(self):
        return np.array([self.x])

    def set_dofs(self, xin):
        self.x = xin[0]

class Adder():
    """This class defines a minimal object that can be optimized. It has
    n degrees of freedom, and has a function that just returns the sum
    of these dofs. This class is used for testing.
    """

    def __init__(self, n=3):
        self.x = np.zeros(n)
        self.fixed = np.full(n, False)        

    def J(self):
        """
        Returns the sum of the degrees of freedom.
        """
        return np.sum(self.x)
        
    def get_dofs(self):
        return self.x

    def set_dofs(self, xin):
        self.x = np.array(xin)


def unique(inlist):
    """
    Given a list or tuple, return a list in which all duplicate
    entries have been removed. Unlike a python set, the order of
    entries in the original list will be preserved.  There is surely
    a faster algorithm than the one used here, but this function will
    not be used in performance-critical code.
    """

    outlist = []
    seen = set()
    for j in inlist:
        if j not in seen:
            outlist.append(j)
            seen.add(j)
    return outlist

def function_from_user(target):
    """
    Given a user-supplied "target" to be optimized, extract the
    associated callable function.
    """
    if callable(target):
        return target
    elif hasattr(target, 'J') and callable(target.J):
        return target.J
    else:
        raise TypeError('Unable to find a callable function associated with the user-supplied target ' + str(target))

class Target():
    """
    Given an attribute of an object, which typically would be a
    @property, form a callable function that can be used as a target
    for optimization.
    """
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        self.depends_on = [obj]
        
    def J(self):
        return getattr(self.obj, self.attr)
    
    # Eventually add a dJ function here

def optimizable(obj):
    """
    Given any object that has a get_dofs() function, add attributes
    fixed, mins, and maxs. fixed = False by default.
    """
    n = len(obj.get_dofs())
    obj.fixed = np.full(n, False)
    obj.mins = np.full(n, np.NINF)
    obj.maxs = np.full(n, np.Inf)
    return obj
