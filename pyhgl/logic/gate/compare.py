####################################################################################################
#
# PyHGL - A Python-embedded Hardware Generation Language 
# Copyright (C) 2022 Jintao Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
####################################################################################################


from __future__ import annotations

from pyhgl.array import *
from pyhgl.logic.hgl_core import *
import pyhgl.logic.utils as utils 
from pyhgl.logic.hgl_assign import __hgl_when__, __hgl_elsewhen__, __hgl_partial_assign__


# FIXME unknown immd is not allowed

@dispatch('Eq', Any, Any) 
class _Eq(Gate):
    
    id = 'Eq'
    
    def __head__(self, left: Reader, right: Union[int,str,list,dict,Reader,BitPat], id='') -> Reader:
        """ 
        """
        assert isinstance(left._data, LogicData)
        self.id = id or self.id 
        ret = UInt[1](0)
        self.output = self.write(ret)

        if isinstance(left, Reader) and isinstance(right, Reader):
            self.left = self.read(left)
            self.right = self.read(right) 
            return ret
        else:  # BitPat or Logic
            self.left = self.read(left)
            self.right: Union[BitPat, Logic] = self.left._type._eval(right)
            return ret

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py   
        target = self.output._data 
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        if isinstance(right, BitPat):
            mask = ~right.x
            while 1:
                yield 
                v1 = right.v 
                v2 = left.v & mask 
                x2 = left.x & mask 
                if x2:
                    if v1 & (~x2) == v2 & (~x2):  # bits without x
                        out_v = gmpy2.mpz(0)
                        out_x = gmpy2.mpz(1)
                    else:
                        out_v = gmpy2.mpz(0)
                        out_x = gmpy2.mpz(0) 
                else:
                    out_v = gmpy2.mpz(v1==v2)
                    out_x = gmpy2.mpz(0)
                simulator.update_v(delay, target, out_v) 
                simulator.update_x(delay, target, out_x)
        else:
            while 1:
                yield 
                v1, x1 = left.v, left.x 
                v2, x2 = right.v, right.x 
                if x1 or x2:
                    x = ~(x1 | x2)
                    if v1 & x == v2 & x:   # has x, rest eq
                        out_v = gmpy2.mpz(0) 
                        out_x = gmpy2.mpz(1)
                    else:                 # rest not eq
                        out_v = gmpy2.mpz(0) 
                        out_x = gmpy2.mpz(0)
                else:
                    out_v = gmpy2.mpz(v1==v2)
                    out_x = gmpy2.mpz(0)
                simulator.update_v(delay, target, out_v) 
                simulator.update_x(delay, target, out_x) 

    
    def sim_v(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py   
        target = self.output._data 
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data 
        else:
            right = self.right
        if isinstance(right, BitPat):
            mask = ~right.x
            while 1:
                yield 
                out_v = gmpy2.mpz((left.v & mask) == right.v)
                simulator.update_v(delay, target, out_v) 
        else:
            while 1:
                yield 
                out_v = gmpy2.mpz(left.v == right.v)
                simulator.update_v(delay, target, out_v) 


    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):  
        """ bitpat: x ==? 6'b1??1??
        """
        left = builder.get_name(self.left)
        right = builder.get_name(self.right)
        if isinstance(right, BitPat):
            op = '==?'
        else:
            op = '=='
        x = f"{left} {op} {right}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 
        
        
        
@dispatch('Ne', Any, Any) 
class _Ne(_Eq):
    
    id = 'Ne'

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py   
        target = self.output._data 
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        if isinstance(right, BitPat):
            mask = ~right.x
            while 1:
                yield 
                v1 = right.v 
                v2 = left.v & mask 
                x2 = left.x & mask 
                if x2:
                    if v1 & (~x2) == v2 & (~x2):  # bits without x
                        out_v = gmpy2.mpz(0)
                        out_x = gmpy2.mpz(1)
                    else:
                        out_v = gmpy2.mpz(0)
                        out_x = gmpy2.mpz(0) 
                else:
                    out_v = gmpy2.mpz(v1==v2)
                    out_x = gmpy2.mpz(0) 
                simulator.update_v(delay, target, gmpy2.mpz(1)-out_v) 
                simulator.update_x(delay, target, out_x)
        else:
            while 1:
                yield 
                v1, x1 = left.v, left.x 
                v2, x2 = right.v, right.x 
                if x1 or x2:
                    x = ~(x1 | x2)
                    if v1 & x == v2 & x:   # has x, rest eq
                        out_v = gmpy2.mpz(0) 
                        out_x = gmpy2.mpz(1)
                    else:                 # rest not eq
                        out_v = gmpy2.mpz(0) 
                        out_x = gmpy2.mpz(0)
                else:
                    out_v = gmpy2.mpz(v1==v2)
                    out_x = gmpy2.mpz(0)
                simulator.update_v(delay, target, gmpy2.mpz(1) - out_v) 
                simulator.update_x(delay, target, out_x) 

    
    def sim_v(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py   
        target = self.output._data 
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data 
        else:
            right = self.right
        if isinstance(right, BitPat):
            mask = ~right.x
            while 1:
                yield 
                out_v = gmpy2.mpz((left.v & mask) != right.v)
                simulator.update_v(delay, target, out_v) 
        else:
            while 1:
                yield 
                out_v = gmpy2.mpz(left.v != right.v)
                simulator.update_v(delay, target, out_v) 


    def dump_sv(self, builder: sv.ModuleSV):  
        left = builder.get_name(self.left)
        right = builder.get_name(self.right)
        if isinstance(right, BitPat):
            op = '!=?'
        else:
            op = '!='
        x = f"{left} {op} {right}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 


@dispatch('Lt', Any, Any) 
class _Lt(Gate):
    
    id = 'Lt'
    _op = '<'
    
    def __head__(self, left: Reader, right: Union[int,str,list,dict,Reader], id='') -> Reader:
        assert isinstance(left._data, LogicData)
        self.id = id or self.id 
        ret = UInt[1](0)
        self.output = self.write(ret)

        if isinstance(left, Reader) and isinstance(right, Reader):
            self.left = self.read(left)
            self.right = self.read(right) 
            return ret
        else:  
            self.left = self.read(left)
            self.right: Logic = self.left._type._eval(right)
            assert not isinstance(self.right, BitPat)
            return ret
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            if left.x or right.x:
                out_v = gmpy2.mpz(0)
                out_x = gmpy2.mpz(1)
            else:
                out_v = gmpy2.mpz(left.v < right.v)
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            out_v = gmpy2.mpz(left.v < right.v)
            simulator.update_v(delay, target, out_v) 

    def dump_sv(self, builder: sv.ModuleSV):  
        """ bitpat: x ==? 6'b1??1??
        """
        left = builder.get_name(self.left)
        right = builder.get_name(self.right)
        x = f"{left} {self._op} {right}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 

        
@dispatch('Gt', Any, Any) 
class _Gt(_Lt):
    
    id = 'Gt' 
    _op = '>'
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            if left.x or right.x:
                out_v = gmpy2.mpz(0)
                out_x = gmpy2.mpz(1)
            else:
                out_v = gmpy2.mpz(left.v > right.v)
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            out_v = gmpy2.mpz(left.v > right.v)
            simulator.update_v(delay, target, out_v) 



@dispatch('Le', Any, Any) 
class _Le(_Lt):
    
    id = 'Le'
    _op = '<='
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            if left.x or right.x:
                out_v = gmpy2.mpz(0)
                out_x = gmpy2.mpz(1)
            else:
                out_v = gmpy2.mpz(left.v <= right.v)
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            out_v = gmpy2.mpz(left.v <= right.v)
            simulator.update_v(delay, target, out_v) 


@dispatch('Ge', Any, Any) 
class _Ge(_Lt):
    
    id = 'Ge'
    _op = '>='
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            if left.x or right.x:
                out_v = gmpy2.mpz(0)
                out_x = gmpy2.mpz(1)
            else:
                out_v = gmpy2.mpz(left.v >= right.v)
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        left = self.left._data 
        if isinstance(self.right, Reader):
            right = self.right._data  
        else:
            right = self.right
        while 1:
            yield    
            out_v = gmpy2.mpz(left.v >= right.v)
            simulator.update_v(delay, target, out_v) 


# FIXME not use

def MuxSel(sel: Reader, default: Any, table: dict) -> Any:
    """ ex. 
    MuxSel(x, [UInt('111'), UInt('00')], {
        0: [1,2],
        1: [3,4],
        '1??': [5,6]
    })

    MuxSel(x, UInt[4](0), [4,5,6,7])
    """ 
    assert isinstance(sel, Reader) 
    if isinstance(table, list):
        table = {i:v for i, v in enumerate(table)}
    return MuxSeq(default=default, table={sel==k:v for k, v in table.items()})


def MuxSeq(default: Any, table: dict ) -> Any:
    """ ex. 
    MuxSeq([UInt('111'), UInt('00')], {
        x == 0: [1,2],
        x == 1: [3,4],
        x == '1??': [5,6]
    })
    
    """
    default = ToArray(default) 
    ret = WireNext(default)
    sequence = [(cond,v) for cond, v in table.items()] 
    cond0, value0 = sequence[0]
    with __hgl_when__(cond0): 
        __hgl_partial_assign__(ret, ToArray(value0))
    for cond, value in sequence[1:]:
        with __hgl_elsewhen__(cond):
            __hgl_partial_assign__(ret, ToArray(value))
    return ret



