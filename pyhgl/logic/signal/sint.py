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
from typing import  Optional, Dict, Any, List, Tuple, Generator, Callable, Literal
from pyhgl.logic.hgl_core import *


class SIntType(LogicType):

    _cache = {}

    def __init__(self, width: int = 0): 
        assert width > 0
        super().__init__(width)

    def _eval(self, v: Union[int, str, Logic, BitPat]) -> Union[Logic, BitPat]:
        """
        - overflow:
            BitPat not allowed 
            others cut off
        - underflow:
            negative int|Logic is sign-extended 
            other immds always zero-extend
        """
        if isinstance(v, str) and '?' in v:
            v = BitPat(v)  

        if isinstance(v, BitPat):
            assert len(v) <= self._width, f'{v} overflow for SInt[{self._width}]'
            return v 
        elif isinstance(v, str):   # zext for '4:-d1'
            _v, _x, _w =  utils.str2logic(v) 
            mask = gmpy2.bit_mask(self._width)
            return Logic(_v & mask, _x & mask)
        elif isinstance(v, Logic):
            _v = v.v 
            _x = v.x 
            mask = gmpy2.bit_mask(self._width)
            return Logic(_v & mask, _x & mask)
        else:
            _v = v 
            _x = 0
            mask = gmpy2.bit_mask(self._width)
            return Logic(_v & mask, _x & mask)

    def __call__(
        self, 
        v: Union[int, str, Reader, Logic]=0, 
        *, 
        name: str = ''
    ) -> Reader: 
        name = name or 'sint'
        # is type casting
        if isinstance(v, Reader):    
            data = v._data   
            assert isinstance(data, LogicData), f'signal {v} is not logic signal' 
            assert v._type._storage == 'packed', f'signal {v} has variable bit length'
            assert len(v) == len(self), 'bit length mismatch'
            return Reader(data=data, type=self, name=name) 
        else:
            data = self._eval(v)  
            assert isinstance(data, Logic)
            return Reader(data=LogicData(data.v, data.x, len(self)), type=self, name=name) 
        
    def __str__(self, data: LogicData = None):
        if data is None:
            return f"SInt[{len(self)}]" 
        else:
            return f"s{utils.logic2str(data.v, data.x, width=len(self))}"
    

@singleton
class SInt(HGLFunction):
    
    def __getitem__(self, key: int):
        """
        ex. SInt[2], SInt[8]
        """  
        assert key > 0 and isinstance(key, int)
        cache = SIntType._cache
        if key in cache:
            return cache[key]
        else:
            cache[key] = SIntType(key)
            return cache[key]

    def __call__(
        self, 
        v: Union[str, int, Reader, Iterable, Logic]='0', 
        w: int = None,
        name: str = ''
    ) -> Reader: 

        # array
        v = ToArray(v) 
        w = ToArray(w)
        if isinstance(v, Array) or isinstance(w, Array):
            return Map(self, v, w, name=name)

        # with width, pass
        if w is not None:
            return SInt[w](v, name=name)
        # without width
        if isinstance(v, Reader):    
            _w = len(v)
        else:
            if isinstance(v, str):
                _, _, _w = utils.str2logic(v) 
            elif isinstance(v, Logic):
                _w = max(utils.width_infer(v.v, signed=True), utils.width_infer(v.x, signed=True))
            else:
                _w = utils.width_infer(v, signed=True)
        return SInt[_w](v, name=name)
    
    