import numpy as np
from collections import Counter, OrderedDict
import copy
import json
import re

class Unit:
    def __init__(self, unit, number = 0.0):
        unit = unit.replace("(", "")
        unit = unit.replace(")", "")
        if unit in ['PI', 'NO', 'NK', 'None']:
            self.num, self.den = Counter([]), Counter([])
        elif unit.isdigit():
            self.num, self.den = Counter([]), Counter([])
        else:
            self.num, self.den = self.parse(unit)
        try:
            self.number = float(number)
        except:
            self.number = 0.0
        if unit == 'None':
            self.isNone = True
        else:
            self.isNone = False
    
    def parse(self, unit):
        if unit.isdigit():
            return Counter([]), Counter([])
        elif '/' in unit:
            num, den = unit.strip().split('/')
            num, _ = self.parse(num)
            den, _ = self.parse(den)
            return self.reduce(num, den)
        else:
            unit = unit.strip()
            num, den  = [], []
            toks = unit.split('.')
            for tok in toks:
                if '^' in tok:
                    base, exp = tok.strip().split('^')
                    for _ in range(int(exp)):
                        num.append(base.strip())
                else:
                    num.append(tok)
            return Counter(num), Counter(den)
    
    def reduce(self, num, den):
        upd_num, upd_den = copy.deepcopy(num), copy.deepcopy(den) 
        for unit in set(num.keys()).intersection(set(den.keys())):
            upd_num[unit] = (num[unit] - min(num[unit], den[unit]))
            upd_den[unit] = (den[unit] - min(num[unit], den[unit]))
            
        upd_num = Counter(el for el in upd_num.elements() if upd_num[el] > 0)
        upd_den = Counter(el for el in upd_den.elements() if upd_den[el] > 0)
        
        return upd_num, upd_den
    
    def to_string(self, num, den):
        if self.isNone:
            return 'None'
        rep = ''
        if len(num.keys()) > 1:
            rep = rep + '('
        for unit in num.keys():
            count = num[unit]
            if count == 1:
                rep += unit + '.'
            elif count > 1:
                rep += unit + '^' + str(count) + '.'
        rep = rep[:-1]
        if len(num.keys()) > 1:
            rep = rep + ')'
        
        if len(den.keys()) == 1:
            rep = rep + '/'
        elif len(den.keys()) > 1:
            rep = rep + '/('
        for unit in den.keys():
            count = den[unit]
            if count == 1:
                rep += unit + '.'
            elif count > 1:
                rep += unit + '^' + str(count) + '.'
        if len(den.keys()) == 1:
            rep = rep[:-1]
        elif len(den.keys()) > 1:
            rep = rep[:-1] + ')'
        return str(rep)
        
    def __str__(self):
        return str(self.number) + ' Unit(\'' + self.to_string(self.num, self.den) + '\')'
    
    def __eq__(self, other):
        if self.isNone and other.isNone:
            return True
        elif (not self.isNone) and (not other.isNone):
            if (self.num == other.num) and (self.den == other.den):
                return True
            else:
                return False
        else:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
        if (self.isNone or other.isNone):
            return Unit('None')
        if (self.num == other.num) and (self.den == other.den):
            return Unit(self.to_string(self.num, self.den), self.number + other.number)
        else:
            return Unit('None')
            # raise Exception("units mismatch in addition")
    
    def __sub__(self, other):
        if (self.isNone or other.isNone):
            return Unit('None')
        if (self.num == other.num) and (self.den == other.den):
            return Unit(self.to_string(self.num, self.den), self.number - other.number)
        else:
            return Unit('None')
            # raise Exception("units mismatch in subtraction")
            
    def __mul__(self, other):
        if (self.isNone or other.isNone):
            return Unit('None')
        num, den = self.reduce(self.num + other.num, self.den + other.den)
        return Unit(self.to_string(num, den), self.number * other.number)
    
    def __truediv__(self, other):
        if (self.isNone or other.isNone):
            return Unit('None')
        num, den = self.reduce(self.num + other.den, self.den + other.num)
        if other.number != 0:
            return Unit(self.to_string(num, den), self.number / other.number)
        else:
            # raise Exception("zero division error")
            return Unit('None')

    def __pow__(self, other):
        if (self.isNone or other.isNone):
            return Unit('None')
        if (len(other.num) == 0) and (len(other.den) == 0):
            exp = int(other.number)
            if other.number != exp:
                return Unit('None')
                # raise Exception("fraction pows not supported")
            else:
                if (other.number >= 0) and (other.number <= 3):
                    new_num, new_den = Counter([]), Counter([])
                    for _ in range(exp):
                        new_num += self.num
                        new_den += self.den
                    num, den = self.reduce(new_num, new_den)
                    return Unit(self.to_string(num, den), self.number ** other.number)
                else:
                    return Unit('None')
        else:
            return Unit('None')
            # raise Exception("units mismatch in pow")