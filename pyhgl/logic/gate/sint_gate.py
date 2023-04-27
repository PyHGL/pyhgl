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
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import inspect 
import random
from itertools import chain

from pyhgl.array import *
from pyhgl.logic.hgl_core import *
import pyhgl.logic.utils as utils
import pyhgl.logic.module_cpp as cpp
import pyhgl.logic.module_sv as sv 
from pyhgl.logic.signal import *


@dispatch('Rshift', SIntType, Any) 
class _Lshift(Gate):
    
    id = 'Rshift'
    _op = '>>'

    """
    SInt >> UInt  ==>  SInt
    """
    
    def __head__(self, 
            a: Reader, 
            b: Union[int, Reader, Logic, str], 
            id: str = '', 
            name: str='temp_shift'
        ):
        assert isinstance(a._data, LogicData) 
        self.id: str = id or self.id
        self.width = len(a)
        self.a = self.read(a)
        if isinstance(b, Reader): 
            assert isinstance(b._type, UIntType)
            self.b = self.read(b)
        else:
            self.b = Logic(b)
            assert self.b.v >= 0 and self.b.x == 0
        ret: Reader = SInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret
    
    def forward(self):
        a: Logic = self.a._data._getval_py()
        b: Logic = self.b 
        mask = gmpy2.bit_mask(self.width)
        if isinstance(b, Reader):
            b = b._data._getval_py() 
        if b.x:    # unknown
            self.output._data._setval_py(Logic(0, mask), dt=self.delay, trace=self)
        else: 
            v = a.v 
            x = a.x
            if v[self.width-1]:   # 1... 
                v = v | ~mask 
            if x[self.width-1]:   # x...
                x = x | ~mask
            self.output._data._setval_py(
                Logic(
                    (v >> b.v) & mask,
                    (x >> b.v) & mask,
                ),
                dt = self.delay, trace=self
            )

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = f"$signed({builder.get_name(self.a)}) {self._op} {builder.get_name(self.b)}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 
    
    

class _GateSigned2(Gate):

    _op: str = ''   

    """
    binary op, at least one of oprand is SInt 
    ret bit-length = max(len(a), lan(b))
    """
    
    def __head__(self, a: Reader, b: Reader, id: str='', name: str='temp') -> Reader:
        assert isinstance(a._type, (UIntType, SIntType))
        assert isinstance(b._type, (UIntType, SIntType))
        self.id: str = id or self.id
        self.a: Reader = self.read(a)
        self.b: Reader = self.read(b)
        self.signed_a: bool = isinstance(a._type, SIntType)
        self.signed_b: bool = isinstance(b._type, SIntType) 
        assert self.signed_a or self.signed_b
        self.width_a: int = len(a)
        self.width_b: int = len(b)
        self.width: int = max(len(a), len(b))

        ret: Reader = SInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret
    
    def to_signed(self, a: Logic, b: Logic) -> Tuple[int, int]:
        a = a.v 
        b = b.v 
        if self.signed_a and a[self.width_a-1]:
            a = a | ~gmpy2.bit_mask(self.width_a)
        if self.signed_b and b[self.width_b-1]:
            b = b | ~gmpy2.bit_mask(self.width_b)
        return a, b

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV): 
        name_a = builder.get_name(self.a)
        if self.signed_a:
            name_a = f'$signed({name_a})'
        name_b = builder.get_name(self.b)
        if self.signed_b:
            name_b = f'$signed({name_b})'
        
        x = f"$signed({name_a} {self._op} {name_b})"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 


@dispatch('Mul', SIntType, SIntType)
@dispatch('Mul', SIntType, UIntType)
@dispatch('Mul', UIntType, SIntType)
class _SMul(_GateSigned2):
        
    id = 'Mul'
    _op = '*'

    """
    a:SInt * b:SInt ==> ret:SInt 
    a:UInt * b:UInt ==> ret:SInt

    len(ret) = max(len(a), len(b)) 
    """
    
    def forward(self):
        a: Logic = self.a._data._getval_py()
        b: Logic = self.b._data._getval_py()
        if a.x or b.x:
            self.output._data._setval_py(Logic(0, gmpy2.bit_mask(self.width)), dt=self.delay, trace=self)
        else:
            a, b = self.to_signed(a,b) 
            self.output._data._setval_py(
                Logic( (a * b) & gmpy2.bit_mask(self.width), 0), 
                dt=self.delay, trace=self
            )


def div_round_towards_zero(a: gmpy2.mpz, b: gmpy2.mpz) -> Tuple[gmpy2.mpz, gmpy2.mpz]:
    """ behavior of cpp, not python
    """
    q = a // b 
    if q < 0:
        q = q + 1 
    r = a - b * q 
    return q,r

    
@dispatch('Floordiv', SIntType, SIntType) 
@dispatch('Floordiv', SIntType, UIntType)
@dispatch('Floordiv', UIntType, SIntType)
class _SFloordiv(_GateSigned2):
    
    id = 'Floordiv'
    _op = '/'

    def forward(self):
        a: Logic = self.a._data._getval_py()
        b: Logic = self.b._data._getval_py()
        if a.x or b.x or b.v == 0:
            self.output._data._setval_py(Logic(0, gmpy2.bit_mask(self.width)), dt=self.delay, trace=self)
        else:
            a, b = self.to_signed(a,b) 
            q, r = div_round_towards_zero(a,b)
            self.output._data._setval_py(
                Logic( q & gmpy2.bit_mask(self.width), 0), 
                dt=self.delay, trace=self
            )

    
@dispatch('Mod', SIntType, SIntType) 
@dispatch('Mod', SIntType, UIntType)
@dispatch('Mod', UIntType, SIntType)
class _SMod(_GateSigned2):
    
    id = 'Mod'
    _op = '%'
    
    def forward(self):
        a: Logic = self.a._data._getval_py()
        b: Logic = self.b._data._getval_py()
        if a.x or b.x or b.v == 0:
            self.output._data._setval_py(Logic(0, gmpy2.bit_mask(self.width)), dt=self.delay, trace=self)
        else:
            a, b = self.to_signed(a,b) 
            q, r = div_round_towards_zero(a,b)
            self.output._data._setval_py(
                Logic( r & gmpy2.bit_mask(self.width), 0), 
                dt=self.delay, trace=self
            )
