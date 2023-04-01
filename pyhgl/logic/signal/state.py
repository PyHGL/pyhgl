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


class EnumType(SignalType):
    """ dynamic bit width, add new states dynamically
    
    _eval(state:str) will record the state under encoding 'binary', 'onehot', 'gray' 
    
    args:
        states: pre-defined states 
            - list: number grow from 0 
            - dict: user defined map state to value (may out of encoding)  
        encoding: one of 'binary', 'onehot', 'gray' 
        frozen:
            - False: can add new state/change state by call _eval 
            - True: can not add new state, width is fixed 
    methods:
        frozen: 
            - once called, this type is immutable 
    
    TODO onehot, gray encoding
    TODO s['idle.1']
    """ 
    
    __slots__ = '_encoding', '_width', '_frozen', '_order', '_names', '_values'
    
    def __init__(
        self, 
        states: Union[List[str], Dict[str, int]], 
        *, 
        encoding: Literal['binary', 'onehot', 'gray'] = 'binary', 
        frozen = False
    ):
        encoding = encoding.lower()
        assert encoding in ['binary', 'onehot', 'gray'] 
        self._encoding = encoding 

        self._storage = 'packed variable'
        
        self._frozen = False
        # current width, variable
        self._width = 1
        # max order in encoding space
        self._order = 0
        # one-to-one mapping
        self._names: Dict[str, int] = {}  
        self._values: Dict[int, str] = {} 
        
        if isinstance(states, dict):
            for k, v in states.items(): 
                self._eval(k, predefined=v)
        else:
            for i in states:
                self._eval(i) 
                
        if frozen:
            self.frozen()
            
            
    def __len__(self) -> int:
        return self._width 
    
    
    def frozen(self: Reader):
        if isinstance(self, EnumType):
            self._frozen = True  
            self._storage = 'packed'
        else:
            assert isinstance(self._type, EnumType)
            self._type._frozen = True 
            self._type._storage = 'packed'
    
    
    def _eval(self, x: Union[int, str, Logic], *, predefined: int = None) -> Logic: 
        """
        since width may change, _eval a same integer may return different value
        """

        if isinstance(x, str) and '?' in x:
            x = BitPat(x)  
        if isinstance(x, BitPat):
            assert len(x) <= self._width, 'overflow'
            return x 

        # changing the map from state to value 
        if predefined is not None:
            assert isinstance(x, str) and isinstance(predefined, int) and not self._frozen 
            # pop odd name and value
            if (_v:=self._names.pop(x)) is not None:
                self._values.pop(_v)  
            assert predefined not in self._values, f'confilct: {predefined}'
            ret = self._new_state(x, predefined)
            return Logic(ret, 0)  
             
        if isinstance(x, str):
            if self._frozen: 
                assert x in self._names, f'state {x} not found' 
                return Logic(self._names[x], 0)  
            else:
                if (ret:=self._names.get(x)) is not None:
                    return Logic(ret, 0) 
                else:
                    ret = self._new_state(x)
                    return Logic(ret, 0)    
        elif isinstance(x, Logic):
            assert self._width >= utils.width_infer(x.v) and self._width >= utils.width_infer(x.x)
            return x  
        else:
            x = int(x) 
            assert self._width >= utils.width_infer(x)
            return Logic(x, 0)
            
    def _new_state(self, state: str, value: Optional[int] = None) -> int:
        if self._encoding == 'binary':
            if value is None:
                value = self._order 
                self._order += 1             
            else:
                self._order = max(value+1, self._order)  
        elif self._encoding == 'onehot':
            if value is None:
                value = utils.binary2onehot(self._order)
                self._order += 1 
            else:
                self._order = max(self._order, utils.width_infer(value))
        elif self._encoding == 'gray':
            if value is None:
                value = utils.binary2gray(self._order)
                self._order += 1 
            else:
                self._order = max(self._order, utils.gray2binary(value)+1)
        else:
            raise Exception('unknown encoding') 
        
        self._width = max(utils.width_infer(value),self._width)
        self._names[state] = value 
        self._values[value] = state 
        return value
    
    
    def __call__(self, v: Union[str, int] = None, *, name='state_enum') -> Reader: 
        if v is None:
            if self._values:
                v = next(iter(self._values))
            else:
                v = 0
        v = self._eval(v)  
        assert isinstance(v, Logic)
        return Reader(data=LogicData(v.v, v.x), type=self, name=name)
        
    def __str__(self, data: LogicData = None):  
        if self._names:
            k, v = next(iter(self._names.items()))
            temp = f'{k}={v},...'
        else:
            temp = 'None'
        if data is None:
            return f'Enum[{temp}]'  
        else:
            v = int(data.v)
            if state:=self._values.get(v):
                return f'Enum[{temp}]({state}={v})' 
            else:
                return f'Enum[{temp}]'
    
    def _slice(self, high_key):
        """ 
        signal: 
            None: called by partial assignment 
            other: called by signal.__getitem__ 
        """  
        if high_key is None:
            return None, self 
        else:
            raise Exception(f'EnumType does not support slicing {high_key}')

    def __getitem__(self, key) -> gmpy2.mpz:
        assert isinstance(key, str)
        return self._eval(key)  
        
    

@singleton 
class Enum:
    """
    sel_t = Enum['a','b','c',...]   # frozen=False, encoding='binary'
    sel_t = Enum['a','b','c']       # frozen=True, encoding='binary'
    
    """
    def __init__(self):
        pass 
    
    def __getitem__(self, keys):
        if keys is ...:
            return EnumType([])
        else:
            assert isinstance(keys, tuple) 
            if keys[-1] is ...: 
                return EnumType(states=keys[:-1]) 
            else:
                return EnumType(states=keys, frozen=True) 
            
    def __call__(self) -> Any:
        return EnumType([])()

            
@singleton 
class EnumOnehot:
    """
    sel_t = Enum['a','b','c',...]   # frozen=False, encoding='binary'
    sel_t = Enum['a','b','c']       # frozen=True, encoding='binary'
    
    """
    def __init__(self):
        pass 
    
    def __getitem__(self, keys):
        if keys is ...:
            return EnumType(encoding='onehot')
        else:
            assert isinstance(keys, tuple) 
            if keys[-1] is ...: 
                return EnumType(states=keys[:-1], encoding='onehot') 
            else:
                return EnumType(states=keys, frozen=True, encoding='onehot') 
            
    def __call__(self) -> Any:
        return EnumType([], encoding='onehot')()
            
@singleton 
class EnumGray:
    def __init__(self):
        pass 
    
    def __getitem__(self, keys):
        if keys is ...:
            return EnumType(encoding='gray')
        else:
            assert isinstance(keys, tuple) 
            if keys[-1] is ...: 
                return EnumType(states=keys[:-1], encoding='gray') 
            else:
                return EnumType(states=keys, frozen=True, encoding='gray') 
            
    def __call__(self) -> Any:
        return EnumType([], encoding='gray')()