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
        if isinstance(v, str) and '?' in v:
            v = BitPat(v)  
        if isinstance(v, BitPat):
            assert len(v) <= self._width, f'{v} overflow for SInt[{self._width}]'
            return v 

        if isinstance(v, str): 
            _v, _x, _w =  utils.str2logic(v) 
        elif isinstance(v, Logic):
            _v = v.v 
            _x = v.x 
            _w = max(utils.width_infer(v.v, signed=True), utils.width_infer(v.x))
        else:
            _w = utils.width_infer(_v, signed=True) 
            _v = v & gmpy2.bit_mask(_w)
            _x = 0
        # overflow not allowed
        if self._width < _w or _v < 0 or _x < 0:
            raise Exception(f'value {v} overflow for {self}') 
        return Logic(_v, _x)

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
            return Reader(data=LogicData(data.v, data.x), type=self, name=name) 
    

@singleton
class SInt(HGLFunction):
    
    def __getitem__(self, key: int):
        """
        ex. SInt[2], SInt[8]
        """  
        assert key > 0 and isinstance(key, int)
        cache = UIntType._cache
        if key in cache:
            return cache[key]
        else:
            cache[key] = UIntType(key)
            return cache[key]

    def __call__(
        self, 
        v: Union[int, str, float, Reader, Iterable, Logic]=0, 
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
                _w = max(utils.width_infer(v.v, signed=True), utils.width_infer(v.x))
            else:
                _w = utils.width_infer(v, signed=True)
        return SInt[_w](v, name=name)