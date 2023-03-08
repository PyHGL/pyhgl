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
from pyhgl.logic.hgl_basic import *


_sint_cache = {}

@singleton
class SInt(LogicType):
    
    def __getitem__(self, key: int):
        """
        ex. SInt[1], SInt[8]
        """ 
        if self._width > 0:
            raise TypeError(f'{self}[{key}]')
        else:
            assert key > 0
            if key in _sint_cache:
                return _sint_cache[key]
            else:
                ret = self.__class__(key)
                _sint_cache[key] = ret 
                return ret
    
    def _eval(self, v: Union[int, str, float]) -> Union[gmpy2.mpz, BitPat]:
        """ raise exception if overflow
        """
        if self._width == 0:
            raise Exception(f'{self} is not a valid type')
        if isinstance(v, str): 
            _v, _w =  utils.str2int(v) 
        else:
            _v = v
            _w = utils.width_infer(_v, signed=True) 
         
        if self._width < _w:
            raise Exception(f'value {v} overflow for {self}')
        if _v < 0:
            _v = _v & gmpy2.bit_mask(self._width)
        if isinstance(_v, gmpy2.mpz):
            return _v 
        else:
            return gmpy2.mpz(_v)

    def __call__(
        self, 
        v: Union[int, str, float, Reader, Iterable]=0, 
        w: int = None,
        *, 
        name: str = 'uint'
    ) -> Reader: 
        """
        v:
            - int, str, float 
            - signal 
            - Iterable container of above
        """ 
        if w is not None:
            assert self._width == 0 and w > 0
            return SInt[w](v, name=name)
        
        # array
        v = ToArray(v)
        if isinstance(v, Array):
            return Map(self, v, name=name)
        # is type cast
        if isinstance(v, Reader):    
            data = v._data  
            if not isinstance(data, LogicData):
                raise ValueError(f'signal {v} is not logic signal')
            # has pre defined width
            if self._width:
                return Reader(data=data, type=self, name=name) 
            else:
                w = len(v)
                assert w > 0
                return Reader(data=data, type=SInt[w], name=name)
        else:
            if self._width:
                _v = self._eval(v) 
                return Reader(data=LogicData(_v), type=self, name=name) 
            else:
                if isinstance(v, str):
                    _, _w = utils.str2int(v) 
                else:
                    _w = utils.width_infer(v, signed=True)
                return SInt[_w](v, name=name)