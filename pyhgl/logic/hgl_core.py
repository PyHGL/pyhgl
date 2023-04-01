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
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union, Iterable, Literal

import bisect
import gmpy2
import random

from pyhgl.array import *
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic.module_cpp as cpp
import pyhgl.logic._session as _session  
import pyhgl.logic.module_sv as sv
import pyhgl.logic.utils as utils



#------------------
# Literal
#------------------
    

class BitPat(HGL):
    """ only used in comparation ==, regard as immediate, not signal value
    """
    def __init__(self, v: str) -> None:
        """ 0101??11  -> value: 01010011, mask: 00001100
        """
        assert 'x' not in v, 'unknown value is not allowed in BitPat'
        v, x, width = utils.str2logic(v)
        self.v = gmpy2.mpz(v)
        self.x = gmpy2.mpz(x) 
        self.width = width

    def __len__(self):
        """ BitPat has fixed width """
        return self.width

    def _eq(self, other: Logic) -> Logic:
        """ return 1 bit logic
        """
        mask = ~self.x 
        v1 = self.v 
        v2 = other.v & mask 
        x2 = other.x & mask 
        if x2:
            if v1 & (~x2) == v2 & (~x2):  # bits without x
                return Logic(0,1)
            else:
                return Logic(0,0)
        else:
            return Logic(v1 == v2,0)

    def __str__(self):
        ret = re.sub('x', '?', utils.logic2str(self.v, self.x ,width=self.width))
        return f'BitPat({ret})'  


@vectorize_first 
def _ToLogic(a):
    if isinstance(a, Logic):
        return a 
    elif isinstance(a, str):
        v, x, *_ = utils.str2logic(a)
        return Logic(v,x)
    else:   # int
        return Logic(a, 0)


class Logic(HGL):
    """
    3-valued logic literal: 0, 1, X 

    00 -> 0 
    10 -> 1 
    01 -> X 
    11 -> X 
    """
    
    __slots__ = 'v', 'x'

    def __head__(self, v: int, x: int) -> None:
        # value
        self.v: int = gmpy2.mpz(v)  
        # unknown
        self.x: int = gmpy2.mpz(x)

    def __new__(cls, v: Union[int, list, Array], x: Optional[int] = None):
        if x is None:
            return _ToLogic(v)
        else: 
            self = object.__new__(cls)
            self.__head__(v, x)
            return self
    
    def __bool__(self):
        raise Exception('Logic does not support python bool method')  
    
    def __hash__(self):
        return hash((self.v, self.x))
    
    def __str__(self):
        return utils.logic2str(self.v, self.x)

    def __getitem__(self, key) -> Logic:
        """ bit select, v[0], v[1], ...
        """
        return Logic(
            self.v.__getitem__(key), 
            self.x.__getitem__(key)
        ) 

    def __eq__(self, x: Union[str, int, Logic]) -> bool:
        """ full equal, 01x === 01x, return bool, just for testbench
        """
        other = Logic(x)
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 == x2:
            return (v1 & ~x1) == (v2 & ~x2)
        else:
            return False

    #--------------------------------
    # arithmetic operation
    #--------------------------------
    def __invert__(self) -> Logic:
        return Logic(~self.v, self.x) 
    
    def __and__(self, other: Logic) -> Logic:
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        return Logic(
            v1 & v2,
            x1 & x2 | v1 & x2 | x1 & v2
        ) 

    def __or__(self, other: Logic) -> Logic:
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        return Logic(
            v1 | v2,
            x1 & x2 | ~v1 & x2 | x1 & ~v2
        )

    def __xor__(self, other: Logic) -> Logic:
        return Logic(
            self.v ^ other.v, 
            self.x | other.x
        )

    def __lshift__(self, l: Logic) -> Logic: 
        # TODO minimize x
        if l.x:
            return Logic(0, -1)
        else:
            return Logic(
                self.v << l.v,
                self.x << l.v,
            )
    
    def __rshift__(self, l: Logic) -> Logic: 
        # TODO minimize x
        if l.x:
            return Logic(0, -1)
        else:
            return Logic(
                self.v >> l.v,
                self.x >> l.v,
            )

    def _eq(self, other: Logic) -> Logic:
        """ logic equal
        """
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            x = ~(x1 | x2)
            if v1 & x == v2 & x:  # has x, rest eq
                return Logic(0,1)
            else:                 # rest not eq
                return Logic(0,0) 
        else:
            return Logic(v1 == v2, 0) 
        
    def _ne(self, other: Logic) -> Logic:
        """ logic equal
        """
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            x = ~(x1 | x2)
            if v1 & x == v2 & x:
                return Logic(0,1)
            else:
                return Logic(1,0) 
        else:
            return Logic(v1 != v2, 0) 
        
    def _lt(self, other: Logic) -> Logic:
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, 1)
        else:
            return Logic(v1 < v2, 0) 
        
    def _gt(self, other: Logic) -> Logic:
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, 1)
        else:
            return Logic(v1 > v2, 0) 
        
    def _le(self, other: Logic) -> Logic:
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, 1)
        else:
            return Logic(v1 <= v2, 0) 
        
    def _ge(self, other: Logic) -> Logic:
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, 1)
        else:
            return Logic(v1 >= v2, 0) 

    def _andr(self) -> Logic:
        """ and reduce
        """
        v = self.v 
        x = self.x 
        if (~v) & (~x):
            return Logic(0, 0)
        elif x:
            return Logic(0, 1)
        else:
            return Logic(1,0)
    
    def _orr(self) -> Logic:
        v = self.v 
        x = self.x 
        if v & (~x):
            return Logic(1, 0)
        elif x:
            return Logic(0, 1)
        else:
            return Logic(0, 0)

    def _xorr(self) -> Logic:
        v = self.v 
        x = self.x 
        if x:
            return Logic(0, 1)
        else:
            return Logic(utils.parity(v), 0)

    # calculate 
    def __pos__(self):     
        return self  
    
    def __neg__(self):              
        if self.x:
            return Logic(0, -1)
        else:
            return Logic(-self.v,0)
        
    def __add__(self, other: Logic):   
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, -1)
        else:
            return Logic(v1 + v2, 0)
        
    def __sub__(self, other: Logic):   
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, -1)
        else:
            return Logic(v1 - v2, 0)
        
    def __mul__(self, other: Logic):   
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2:
            return Logic(0, -1)
        else:
            return Logic(v1 * v2, 0)
        
    def __mod__(self, other: Logic):   
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2 or v2 == 0:
            return Logic(0, -1)
        else:
            return Logic(v1 % v2, 0)
        
    def __floordiv__(self, other: Logic):   
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x
        if x1 or x2 or v2 == 0:
            return Logic(0, -1)
        else:
            return Logic(v1 // v2, 0)

    def __truediv__(self, other):   
        raise NotImplementedError()
    
    def _merge(self, unknown: bool, mask: int, other: Logic) -> Logic:
        """
        mask: positions of bits to merge  
        other: another value, should not overflow
        unknown: override or merge 

        ex. self = 1001x, other = 0x000, mask = 11000 -> xx01x
        """
        v1 = self.v 
        v2 = other.v 
        x1 = self.x 
        x2 = other.x 
        mask_n = ~mask
        if unknown:
            return Logic(
                v1 & mask_n | v2,
                x1 & mask_n | x2 | ( (v1 & mask) ^ v2)
            ) 
        else: 
            return Logic(
                v1 & mask_n | v2,
                x1 & mask_n | x2,
            )



#--------------------------------------
# signal data, update in simulation
#--------------------------------------

class SignalData(HGL):
    """ 
    graph node
    
    Attribute:
        - v: value 
        - x: unknown
        - writer: None | Writer 
            - Writer: out-edge of Gate
        - reader: Dict[Reader]
            - Reader: in-edge of Gate 
            - arbitrary number of readers
    """

    _sess: _session.Session

    __slots__ = 'writer', 'reader', '_name', '_module', 'v', 'x', '_v', '_x'
    
    def __init__(self, v: Union[int, gmpy2.mpz], x: Union[int, gmpy2.mpz]):  
        # value
        self.v: gmpy2.mpz = gmpy2.mpz(v)  
        # unknown
        self.x: gmpy2.mpz = gmpy2.mpz(x)
        # to cpp
        self._v: List[cpp.TData] = None
        self._x: List[cpp.TData] = None    
        # 0 or 1 writer
        self.writer: Optional[Writer] = None 
        # any number of reader
        self.reader: Dict[Reader, None] = {}
        # prefered name 
        self._name: str = 'temp'
        # position, None means constant value 
        # if len(self._sess.module._position) == 1: # top level 
        #     self._module: module_hgl.Module = self._sess.module 
        # else:
        self._module: module_hgl.Module = None


    def _update_py(self, new: Logic) -> bool:
        """ only called by python simulator, return whether value changes or not
        """
        if self.v == new.v and self.x == new.x:  
            return False
        else:
            self.v = new.v 
            self.x = new.x 
            return True
        
    def _dump_sv(self, builder: sv.ModuleSV):
        """ dump declaration to builder. if necessary, also dump constant assignment

        ex. logic [7:0] a; assign a = '1;
        """
        raise NotImplementedError() 
    
    def _dump_sv_slice(self, low_key: Tuple, builder: sv.ModuleSV) -> str: 
        """ ex. x[idx +: 2]
        """
        raise NotImplementedError()  


    def _getval_py(self, low_key: Optional[Tuple] = None) -> Logic:
        """ return current value, called by Gate or testbench 

        low_key: for bit select
        """
        raise NotImplementedError()  

    def _setval_py(
        self, 
        value: Union[Logic, List], 
        multiple: bool = False,
        dt: int = 1, 
        trace: Union[None, Gate, str] = None,
    ):
        """  
        low_keys: 
            - None: updata the whole data  
            - List: multiple assignments, calculate corresponding mask and value 
        
        insert a signal update event to py/cpp simulator, called by Gate or testbench
        """
        raise NotImplementedError()   

    def _getval_cpp(self, part: Literal['v', 'x']) -> List[cpp.TData]:
        """ dump to cpp. return list of TData of value/unknown
        """
        raise NotImplementedError() 

    def _setval_cpp(
        self, 
        values: List[List[cpp.TData]],
        low_keys: List[Optional[Tuple]], 
        dt: int,
        part: Literal['v', 'x'],
        builder: cpp.Node,
    ) -> None:
        """ accept multiple assignments, mainly for wire
        """
        raise NotImplementedError() 



class LogicData(SignalData):
    """ 
    packed 3-valued logic 

    - support variable width  
    - default is 0 

    type casting: only valid for same fixed bit length

    XXX notice: xmpz does not support LogicNot;do not slice negative xmpz;key must be python int
    """
    
    __slots__ = ()
    
    def __len__(self) -> int:
        """ bit length is defined by first writer/reader
        """
        if self.writer is not None:
            return len(self.writer)
        elif self.reader:
            return len(next(iter(self.reader)))
        else:
            return 0

    def __hash__(self):             
        return id(self) 


    def _dump_sv(self, builder: sv.ModuleSV) -> None:

        if self._module is None:  # used as constant 
            return 

        if self not in builder.signals: 
            if self.writer is not None:
                netlist = self.writer._driver._netlist 
            else:
                netlist = 'logic'
            if len(self) == 1:
                width = ''
            else:
                width = f'[{len(self)-1}:0]'
            builder.signals[self] = f'{netlist} {width} {{}}'   # declaration 

        if self.writer is None:                         # assignment
            _builder = self._module._module
            if self not in _builder.gates:
                _builder.Block(self, f'assign {_builder.get_name(self)} = {_builder._sv_immd(self)};')

    def _dump_sv_slice(self, low_key: Tuple, builder: sv.ModuleSV) -> str: 
        """ ex. x[idx +: 2]
        """ 
        ret = builder.get_name(self)
        if low_key is None:
            return ret 
        else:
            start, length = low_key 
            if length == 1:
                key_str = f'[{builder.get_name(start)}]'
            else:
                key_str = f'[{builder.get_name(start)}+:{length}]' 
            return f'{ret}{key_str}'
        
    def _getval_py(self, low_key: Optional[Tuple] = None) -> Logic:
        """ 
        - low_key: None | (start, length)
            - start: Logic | Signal 
            - length: int
        - return: Logic with certain width
        """
        if self._v:
            v: gmpy2.xmpz = cpp.getval(self._v)
            x: gmpy2.xmpz = cpp.getval(self._x)
        else:
            v = self.v 
            x = self.x 
        if low_key is None:
            return Logic(v, x)
        else:
            start, length = low_key 
            if isinstance(start, Reader):       # dynamic part select
                start = start._data._getval_py() 
                if start.x:                     # unknown key, return unknown
                    return Logic(0, gmpy2.bit_length(length)) 
                else:
                    start = start.v 
            end = start + length
            max_width = len(self)
            if start >= max_width:
                return Logic(0, gmpy2.bit_length(length)) # out of range, return unknown 
            elif end <= max_width:
                return Logic(v[start:end], x[start:end])  # in range 
            else:                                             # partly in range
                mask =  gmpy2.bit_length(end - max_width) << (max_width-start)
                return Logic(v[start:end], x[start:end] | mask)

    def _setval_py(
        self, 
        value: Union[Logic, List], 
        multiple: bool = False,
        dt: int = 1, 
        trace: Union[None, Gate, str] = None,
    ):
        """ multiple dynamic partial assignment

        value:
            Logic | List[(cond, key, value)]
        multiple:
            true when there are multiple assignments
        dt: delay
        """ 
        if not multiple: 
            data: Logic = value
        else:
            data = Logic(0,0)  
            w = len(self) 
            mask_full = gmpy2.bit_mask(w) 
            for cond, key, new in value: 
                if isinstance(new, Reader):
                    new: Logic = new._data._getval_py()

                if cond is None:
                    unknown = False  
                else:
                    cond: Logic = cond._data._getval_py()  
                    if cond.v == 0 and cond.x == 0:         # skip untriggered 
                        continue 
                    unknown = cond.x > 0 

                if key is None:
                    mask_curr = mask_full
                    new = Logic(new.v & mask_full, new.x & mask_full) 
                else:
                    start, width = key 
                    if isinstance(start, Reader):
                        start: Logic = start._data._getval_py() 
                    else:
                        start: Logic = Logic(start,0)
                
                    if start.x: 
                        mask_curr = mask_full 
                        new = Logic(0, mask_full)
                    else:
                        start: int = start.v 
                        mask_curr = (gmpy2.bit_mask(width) << start) & mask_full  
                        new = Logic(
                            (new.v << start) & mask_curr,
                            (new.x << start) & mask_curr,
                        ) 
                data = data._merge(unknown, mask_curr, new)
        # TODO
        if self._v: 
            w = len(self._v) * 64
            # new_v = utils.split64(new.v, w) 
            # new_x = utils.split64(new.x, w)
            # masks = utils.split64(mask, w)
            # for _target, _mask, _new in zip(self._v, masks, new_v):
            #     cpp.setval(_target, _mask, _new, dt) 
            # for _target, _mask, _new in zip(self._x, masks, new_x):
            #     cpp.setval(_target, _mask, _new, dt) 
        else:
            self._sess.sim_py.insert_signal_event(dt, self, data, trace) 

    def _getval_cpp(self, part: Literal['v', 'x']) -> List[cpp.TData]:
        """ dump cpp. return list of value & unknown
        """
        if not self._v:
            bit_length = len(self) 
            assert bit_length, 'unused signal (no reader and no writer)' 
            self._v = cpp.GlobalArray(self.v, bit_length, self._name+'_v')
            self._x = cpp.GlobalArray(self.x, bit_length, self._name+'_x')
        if part == 'v':
            return self._v 
        elif part == 'x':
            return self._x 
        else:
            raise ValueError()

    def __str__(self):
        w = len(self)
        if w:
            return utils.logic2str(self.v, self.x, width=w)
        else:
            return utils.logic2str(self.v, self.x)




class MemData(SignalData):
    """ 
    unpacked array of bits

    - ex. shape = (1024, 8) means 1024 bytes 
    - does not support bitwise operator, does not support comparation
    - support part select and partial assign

    """

    __slots__ = 'shape', 'length'

    def __init__(
        self, 
        v: Union[int, gmpy2.mpz, gmpy2.xmpz], 
        x: Union[int, gmpy2.mpz, gmpy2.xmpz],
        shape: Tuple[int,...]
    ) -> None:
        super().__init__(v, x)
        # shape 
        assert len(shape) >= 2 and all(int(i)>0 for i in shape), f'error shape {shape}'
        self.shape: Tuple[int,...] = shape  
        self.length: int = 1 
        for i in self.shape:
            self.length *= i 
    
    def __len__(self):
        return self.length

    def __str__(self):
        ret = hex(self.v)[2:] 
        if len(ret) > 35:
            return f'mem{self.shape}{ret[-32:]}'
        else:
            return f'mem{self.shape}{ret}'
        
    def _dump_sv(self, builder: sv.ModuleSV):
        """ dump declaration to builder. if necessary, also dump constant assignment
        """
        if self._module is None:  # constant
            return 
        if self not in builder.signals:
            if self.writer is not None:
                netlist = self.writer._driver._netlist 
            else:
                netlist = 'logic'
            bit_length = self.shape[-1] - 1
            shape = ''.join(f'[0:{w-1}]' for w in self.shape[:-1]) 
            builder.signals[self] = f'{netlist} [{bit_length}:0] {{}} {shape}' 

        if self.writer is None:    # constant assignment 
            _builder = self._module._module 
            if self not in _builder.gates:
                _builder.gates[self] = f'assign {_builder.get_name(self)} = {_builder._sv_immd(self)};'


    def _getval_py(self, low_key: tuple) -> Logic:
        """ low_key: idxes   idx: int | signal
        """
        assert len(low_key) == len(self.shape) - 1 
        valid_keys: List[int] = []
        for key, max_width in zip(low_key, self.shape):
            if isinstance(key, Reader):
                key: Logic = key._data._getval_py()
            if key.x:
                return Logic(0, gmpy2.bit_length(self.shape[-1])) 
            else:
                idx = key.v 
                if idx >= max_width:
                    return Logic(0, gmpy2.bit_length(self.shape[-1]))
                else:
                    valid_keys.append(idx) 

        if self._v:
            _v = self._v 
            _x = self._x 
            for i in valid_keys:
                _v = _v[i]
                _x = _x[i]
            return Logic(cpp.getval(_v), cpp.getval(_x))
        else:
            ret_width = self.shape[-1] 
            shift_width = ret_width 
            start: int = 0
            for width_n, key_n in zip(reversed(self.shape[:-1]),reversed(valid_keys)):
                start += key_n * shift_width 
                shift_width *= width_n  
            end = start + ret_width 
            return Logic(self.v[start, end], self.x[start, end])
            
    def _setval_py(
        self, 
        value: Union[Logic, List], 
        multiple: bool = False,
        dt: int = 1, 
        trace: Union[None, Gate, str] = None,
    ):
        valid_keys: List[int] = []
        # TODO x in key?
        

    def _getval_cpp(self, part: Literal['v', 'x']) -> List[cpp.TData]:
        """ dump cpp. return data list of value/unknown
        """
        if not self._v:
            self._v = cpp.GlobalMem(self.v, self.shape, self._name+'_v', part='v')
            self._x = cpp.GlobalMem(self.x, self.shape, self._name+'_x', part='x')
        if part == 'v':
            return self._v 
        elif part == 'x':
            return self._x 
        else:
            raise ValueError()

#-----------------------------------------
# Signal: in/out edge of SignalData
#-----------------------------------------
    

class Writer(HGL):
    """ a writer is an outedge of specific gate 
    """

    __slots__ = '_data', '_type', '_driver'
    
    def __init__(self, data: SignalData, type: SignalType, driver: Gate): 
        self._data: SignalData = data 
        self._type: SignalType = type
        self._driver: Gate = driver 
        assert data.writer is None, 'cannot drive data already has driver'
        data.writer = self   
        data._module = self._sess.module
        
    def _exchange(self, new_data: SignalData) -> SignalData:
        """ write another data
        """
        if new_data.writer is not None:
            raise Exception('cannot drive data already has driver')
        # new data
        new_data.writer = self 
        # odd data
        self._data.writer = None  
        
        ret, self._data = self._data, new_data
        return ret  
    
    def _delete(self) -> None:
        """ when deleting a gate
        """
        self._data.writer = None 
        self._data = None
        self._driver = None
        self._type = None
        
        
    def __len__(self):
        return len(self._type)
    


class Reader(HGL):
    """ a reader is inedges of gates
    
    _data: 
        instance of SignalData 
        
    _type:
        instance of SignalType
        
    __getattr__:
        get attr of _type. if method, bound to self 
    """
    
    __slots__ = '_data', '_type', '_direction', '_driven'
    
    def __init__(self, data: SignalData, type: SignalType, name: str = ''):
        self._data: SignalData = data; self._data.reader[self] = None
        self._type: SignalType = type 
        self._direction: Literal['inner', 'input', 'output', 'inout'] = 'inner'
        self._driven: Dict[Gate, int] = {}

        if name: self._data._name = name
        
    @property 
    def __hgl_type__(self) -> type:
        """ type for dynamic dispatch
        """
        return type(self._type) 

    @property 
    def _name(self) -> str:
        return self._data._name
        
    def _exchange(self, data: SignalData) -> SignalData:
        """ read from another data
        """
        self._data.reader.pop(self) 
        ret, self._data = self._data, data 
        self._data.reader[self] = None
        return ret  
        
        
    def __getitem__(self, high_key) -> Any:  
        """ building stage, dynamic partial slicing 

        high_key: high-level keys
            - SignalKey: from Array, return self 
            - other: return Slice
        """ 
        assert high_key is not None
        if type(high_key) is SignalKey:
            return self 
        else:
            return Slice(self, key=high_key)
    
    def __partial_assign__(
        self,  
        acitve_signals: List[Reader],
        cond_stacks: List[hgl_assign.CaseGate],
        value: Union[int, Reader, list, dict, str, Any], 
        high_key: Union[SignalKey, int, Reader, tuple, Any], 
    ) -> None:
        """ building stage. recursively call of partial assign
        
        ex. a[key1, key2, key3,...] <== value 

        conds: 
            condition signals 
        value: 
            signal or immd 
        keys:
            high level key: SignalKey, signal or immd
        """
        if type(high_key) is SignalKey: 
            high_key = None
        # insert a wire behind the signal, make it assignable 
        if self._data.writer is None or not isinstance(self._data.writer._driver, Assignable):
            Wire(self)  
        self._data.writer._driver.__partial_assign__(self,acitve_signals,cond_stacks,value,high_key)  

    def __getattr__(self, name: str) -> Any:
        """ 
        get attribute of self._type when:
            - attr not in self
            - name not startswith '_' 
        object method is turned into bounded method by '__get__'
        """
        if name[0] == '_':
            raise AttributeError(f'{self} has no attribute {name}')
        else:
            # first search class attr
            cls = type(self._type)
            if hasattr(cls, name):
                ret = getattr(cls, name)
                # if function, return bounded method
                if hasattr(ret, '__get__'):
                    return ret.__get__(self, self.__class__)  
            # second search object attr
            return object.__getattribute__(self._type, name) 
        
    def __len__(self):
        return len(self._type)
    
    def __str__(self): 
        return f'{self._name}:{self._type.__str__(self._data)}'
    
    def __hash__(self):             return id(self)
    def __bytes__(self):            raise NotImplementedError() 
    def __setitem__(self, k, v):    raise Exception('use <== for partial assign')
    # bitwise
    def __invert__(self):           return Not(self)                # ~a      
    def __or__(self, other):        return Or(self, other)          # a | b
    def __and__(self, other):       return And(self, other)         # a & b
    def __xor__(self, other):       return Xor(self, other)         # a ^ b
    def __lshift__(self, other):    return Lshift(self, other)      # a << b
    def __rshift__(self, other):    return Rshift(self, other)      # a >> b
    def __pow__(self, other):       return Pow(self, other)         # a ** b
    # comparison, return 1-bit signal 
    def __lt__(self, other):        return Lt(self, other)          # a < b
    def __gt__(self, other):        return Gt(self, other)          # a > b
    def __le__(self, other):        return Le(self, other)          # a <= b
    def __ge__(self, other):        return Ge(self, other)          # a >= b
    def __eq__(self, other):        return Eq(self, other)          # a == b
    def __ne__(self, other):        return Ne(self, other)          # a != b
    # calculate 
    def __pos__(self):              return Pos(self)                # +a    
    def __neg__(self):              return Neg(self)                # -a 
    def __add__(self, other):       return Add(self, other)         # a + b
    def __sub__(self, other):       return Sub(self, other)         # a - b
    def __mul__(self, other):       return Mul(self, other)         # a * b
    def __matmul__(self, other):    return Matmul(self, other)      # a @ b
    def __truediv__(self, other):   return Div(self, other)         # a / b
    def __mod__(self, other):       return Mod(self, other)         # a % b
    def __floordiv__(self, other):  return Floordiv(self, other)    # a // b

    def __ilshift__(self, other):   raise NotImplementedError()     # <<=
    def __iadd__(self, other):      raise NotImplementedError()     # +=
    def __isub__(self, other):      raise NotImplementedError()     # -=
    def __imul__(self, other):      raise NotImplementedError()     # *=  
    def __imatmul__(self, other):   raise NotImplementedError()     # @=           
    def __ifloordiv__(self, other): raise NotImplementedError()     # //=
    def __itruediv__(self, other):  raise NotImplementedError()     # /=
    def __imod__(self, other):      raise NotImplementedError()     # %= 
    def __ipow__(self,other):       raise NotImplementedError()     # **=
    def __irshift__(self, other):   raise NotImplementedError()     # >>=
    def __iand__(self, other):      raise NotImplementedError()     # &=
    def __ior__(self, other):       raise NotImplementedError()     # |=
    def __ixor__(self, other):      raise NotImplementedError()     # ^=

    

#----------------------------------
# Signal Type
#----------------------------------

class SignalType(HGL):
    """ signal type system 
    
    - the bit width of signal may not fixed value during building stage
    - _eval accept python literal such as int, str, list, return Immd
        - Immd: Logic | BitPat   
        - overflow is not allowed
    - __call__ create and return a signal. 
        - default value is 0
        - the first positional arg can be Signal or SignalData, indicates type casting
    - _slice accept high-level keys, return low-level key
    - _getval_py: return current value of data
    - _setval_py: return valid mask and value for simulation
    """

    _sess: _session.Session 
    # determines the validity of bitwise operation, type casting, ...
    _storage:  Literal['packed', 'unpacked', 'packed variable'] = 'packed'

    def __len__(self) -> int:
        raise NotImplementedError() 
    
    def _eval(self, v: Any) -> Any:
        """ building stage 
        
        v: int, str, list, dict, bitpat 
        return Immd accepted by SignalData.__init__, or Immd for `==` op
        
        overflow will raise exception
        """
        raise NotImplementedError()
    
    def _slice(self, high_key) -> Tuple[Optional[tuple], SignalType]: 
        """ building stage. turn high-level keys into low-level key
        
        used in both part-select and partial-assignment 
        return (low_key, SignalType)
        """
        raise NotImplementedError()  
    
    def __call__(self) -> Reader:
        """ create a signal with default value
        """
        raise NotImplementedError()  
        
    def __str__(self, data: SignalData=None):
        raise NotImplementedError() 
    
        
@dispatch('Signal', Any)
def _signal(obj) -> Reader:
    """ literal to signal
    """
    if isinstance(obj, Reader):
        return obj 
    elif isinstance(obj, str):
        return UInt(obj)
    elif isinstance(obj, SignalType):
        return obj()
    else: 
        return UInt(obj)


        
class LogicType(SignalType):
    """ base type of UInt, SInt, Vector, Struct, Enum, ...
    """

    def __init__(self, width: int = 0):
        self._width: int = width  
        self._storage = 'packed'
        
    def __len__(self) -> int:
        return self._width
        
    def __str__(self, data: LogicData = None):
        T = f"{self.__class__.__name__}[{len(self)}]" 
        if data is None:
            return T 
        else:
            return f"{T}({data})"


    # def _getval_cpp(
    #     self, 
    #     data: LogicData, 
    #     low_key: Optional[Tuple[int, int]], 
    #     builder: cpp.Node,
    #     unknown: bool                       # slice value or unknown bits
    # ) -> List[cpp.TData]:
    #     """
    #     perform slicing and return sliced data

    #     special case: low_key is None, return original data since there is no down casting
    #     """ 
        # if data._x is None:
        #     data.dump_cpp()
        # source = data._x if unknown else data._v
        # source_bit_length: int = source.bit_length
        # source_array: List[cpp.TData] = source[:]

        # if low_key is None:
        #     low_key = (0, self._width)       

        # start, length = low_key             
        # ret_size = (length + 63) // 64      # number of return data
        # if isinstance(start, Reader):       # TODO dynamic part select
        #     pass
        # else:                               # static part select
        #     end = min(start + length, source_bit_length)  # out of range is 0
        #     start_idx = start // 64 
        #     end_idx = (end + 63) // 64  
        #     used = source_array[start_idx:end_idx] 
        #     for i in used:                  # record 
        #         builder.read(i)
        #     if end % 64 != 0 and end < source_bit_length:   # mask
        #         end_mask = gmpy2.bit_mask(end % 64)
        #         used[-1] = builder.And(used[-1], cpp.TData(end_mask, const=True))
        #     if start % 64 != 0:         # shift 
        #         right_shift = start % 64 
        #         left_shift = 64 - right_shift 
        #         for i in range(len(used)-1):
        #             used[i] = builder.And(
        #                 builder.RShift(used[i], right_shift), 
        #                 builder.LShift(used[i+1], left_shift)

        #             ) 
        #         if used:
        #             used[-1] = builder.RShift(used[-1], right_shift)
        #     used = used[:ret_size]
        #     for _ in range(len(used), ret_size):
        #         used.append(cpp.TData(0, const=True))        
        #     return used


 
    # def _setval_cpp(
    #     self, 
    #     data: LogicData, 
    #     v: List[cpp.TData],
    #     low_key: Optional[Tuple[int, int]], 
    #     builder: cpp.Node, 
    #     unknown: bool
    # ) -> Tuple[List[cpp.TData], List[cpp.TData]]:
    #     """ no overflow, no mask, 
    #     """
    #     if data._v is None: 
    #         data.dump_cpp()
    #     target = data._x if unknown else data._v 
    #     target_bit_legth: int = target.bit_length 
    #     target_array: List[cpp.TData] = target[:]
    #     if low_key is None:
    #         low_key = (0, self._width)

    #     start, length = low_key 
    #     size = (length+63) // 64   # number of changed data 
    #     if isinstance(start, Reader): # TODO dynamic partial assign 
    #         pass 
    #     else:                   # static assign should in range, only shift is needed
    #         ...

    #     v = data._v 
    #     x = data._x 
    #     return (v[:], x[:])

    
    def _slice(self, high_key) -> Tuple[Tuple[Union[int, Reader], int], SignalType]:
        """ uint slicing
        
        keys: high-level keys, ex. int, signal, slice 
        return: ((start,length), UInt)
        
        - index grow from left to right 
        - little endian 
        - include start, not stop  
        
        static slicing:
            - python style: x[0:4]  low 4 bits 
        dynamic partial slicing: 
            - x[signal::8], x[signal]
            - x[3::2]
        
        """
        assert self._storage == 'packed', f'{self} has variable bit length'
        
        if isinstance(high_key, slice):
            start, stop, step = high_key.start, high_key.stop, high_key.step 
            if start is not None and stop is None and step is not None:
                # [signal::8]
                return (start, step), UInt[step] 
            elif step is not None:
                # [:-1]
                raise KeyError(f'invalid slicing {high_key}')
        elif isinstance(high_key, int):
            if high_key < 0:
                high_key += self._width
            start, stop = high_key, high_key+1 
        elif isinstance(high_key, Reader):
            return (high_key, 1), UInt[1]
        elif high_key is None:
            return None, self
        else:
            raise KeyError(high_key) 
        
        if start is None: 
            start = 0 
        if stop is None: 
            stop = self._width 
        if not (isinstance(start, int) and isinstance(stop, int)):
            raise KeyError("constant slicing only support int")
        if start < 0: 
            start += self._width 
        if stop < 0:
            stop += self._width 
        if not (0 <= start < stop <= self._width):
            raise KeyError('slice nothing')  

        return (start, stop-start), UInt[stop-start] 
    

    def __matmul__(self, pos: int) -> Tuple[int, SignalType]: 
        """ only for LogicType
        
        T @ i -> (T, i), used in Struct to indicating bit position
        """
        if isinstance(pos, int):
            if len(self) == 0:
                raise TypeError(f'{self} is not complete signal type') 
            if pos < 0:
                raise Exception(f'negative position')
            return (self, pos) 
        else:
            return pos.__rmatmul__(self)
        
    def __rmul__(self, shape: Union[int, Tuple[int]]) -> Vector:
        """ T * (3, 2) -> [[T,T],[T,T],[T,T]] 
        
        return Vector of shape
        """
        if len(self) == 0:
            raise TypeError(f'{self} is not complete signal type')
        if isinstance(shape, int):
            return Vector(shape, self)
        else:
            t = self
            for i in reversed(shape):
                if not isinstance(i, int):
                    raise Exception('non-integer shape')
                t = Vector(i, t) 
            return t 
        
    def __mul__(self, shape) -> Vector:
        return self.__rmul__(shape)  

    def __pow__(self, shape: Union[tuple, int]) -> MemType:
        if isinstance(shape, int):
            shape = (shape,)
        return MemType(shape, self)

    # bit operations 
    def split(self: Reader) -> Array:
        return Split(self)
    
    def zext(self: Reader, w: int):
        return Zext(self, w) 

    def oext(self: Reader, w: int):
        return Oext(self, w)  

    def sext(self: Reader, w: int):
        return Sext(self, w)   



class UIntType(LogicType):

    # reduce memory cost
    _cache = {}

    def __init__(self, width: int = 0): 
        assert width > 0
        super().__init__(width)

    def _eval(self, v: Union[int, str, Logic, BitPat]) -> Union[Logic, BitPat]:
        """ 
        called when:
            - signal <== immd 
            - signal == immd 
            - UInt(immd) 
        return valid immd
        Exception if v overflowed
        """
        if isinstance(v, str) and '?' in v:
            v = BitPat(v)  
        if isinstance(v, BitPat):
            assert len(v) <= self._width, f'{v} overflow for UInt[{self._width}]'
            return v 

        if isinstance(v, str): 
            _v, _x, _w =  utils.str2logic(v) 
        elif isinstance(v, Logic):
            _v = v.v 
            _x = v.x 
            _w = max(utils.width_infer(v.v), utils.width_infer(v.x))
        else:
            _v = v
            _x = 0
            _w = utils.width_infer(_v) 
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
        name = name or 'uint'
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
class UInt(HGLFunction):
    
    def __getitem__(self, key: int):
        """
        ex. UInt[1], UInt[8]
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
            return UInt[w](v, name=name)
        # without width
        if isinstance(v, Reader):    
            _w = len(v)
        else:
            if isinstance(v, str):
                _, _, _w = utils.str2logic(v) 
            elif isinstance(v, Logic):
                _w = max(utils.width_infer(v.v), utils.width_infer(v.x))
            else:
                _w = utils.width_infer(v)
        return UInt[_w](v, name=name)


def Split(signal: Reader) -> Array:
    assert signal._type._storage == 'packed'
    return Array(signal[i] for i in range(len(signal)))


def Zext(signal: Reader, w: int) -> Reader:
    assert signal._type._storage == 'packed'
    _w = len(signal)
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(signal, UInt[w-_w](0))


def Oext(signal: Reader, w: int) -> Reader:
    assert signal._type._storage == 'packed'
    _w = len(signal)
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(signal, UInt[w-_w]((1<<(w-_w))-1)) 


def Sext(signal: Reader, w: int) -> Reader:
    assert signal._type._storage == 'packed'
    _w = len(signal)
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        msb = signal[-1] 
        return Cat([signal] + [msb] * (w-_w))



class Vector(LogicType):
    """ 1-d array, little endian
    """
    _length: int # vector length 
    _width: int  # bit width 

    def __init__(self, length: int, T: SignalType):
        assert isinstance(length, int) and length > 0, 'empty vector'
        assert isinstance(T, SignalType) and len(T) > 0, 'element should have fixed width'
        self._T = T 
        self._length = length  
        super().__init__(self._length * len(self._T))
    
    
    def _eval(self, v: Union[int, str, Iterable, Logic]) -> Logic:
        """ if not iterable, set UInt(v) as whole; if iterable, set each value 
        """
        v = ToArray(v) 
        
        if isinstance(v, Array): 
            assert len(v) == self._length, 'shape mismatch' 
            _v = gmpy2.mpz(0) 
            _x = gmpy2.mpz(0)
            _w = len(self._T)
            for i, x in enumerate(v): 
                temp: Logic = self._T._eval(x)
                _v |= (temp.v) << (i*_w) 
                _x |= (temp.x) << (i*_w)
            return Logic(_v, _x)
        # regard as UInt
        else:
            return UInt[len(self)]._eval(v) 
    
    def __call__(
        self, 
        v: Union[Reader, Iterable, str, int]=0, 
        *, 
        name: str = 'vector'
    ) -> Reader:
        
        if isinstance(v, Reader):
            v = v._data  
            assert isinstance(v, LogicData)   
            assert len(v) == len(self)
            return Reader(data=v, type=self, name=name)
        else:
            _v = self._eval(v)
            return Reader(data=LogicData(_v.v, _v.x), type=self, name=name)
    
    def _slice(self, keys: Union[int, Reader, tuple]) -> Tuple[Any, SignalType]: 
        # TODO index overflow    

        # deal with multiple slice
        if isinstance(keys, tuple):       
            key = keys[0]
            rest_keys = keys[1:]
        else:
            key = keys
            rest_keys = tuple() 
        rest_keys_valid = rest_keys[0] if len(rest_keys) == 1 else rest_keys
            
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step 
            if isinstance(start, Reader) and stop is None and isinstance(step, int):
                # [signal::8]
                _idx = _full_mul(start, len(self._T)) 
                _type = Vector(step, self._T)
            elif step is None: 
                # [:-2]
                if start is None: 
                    start = 0 
                if stop is None:
                    stop = self._length 
                if not (isinstance(start, int) and isinstance(stop, int)):
                    raise KeyError("constant slicing only support int")
                if start < 0: 
                    start += self._length 
                if stop < 0:
                    stop += self._length 
                if not (0<= start < stop <= self._width):
                    raise KeyError('slice nothing')  
                _idx = start * len(self._T) 
                _type = Vector(stop-start, self._T)
            else:
                raise KeyError(keys)
        elif isinstance(key, int):
            if key < 0:
                key += self._length 
            assert 0 <= key < self._length 
            _idx = key * len(self._T)
            _type = self._T 
        elif isinstance(key, Reader):
            _idx = _full_mul(key, len(self._T))
            _type = self._T
        else:
            raise KeyError(key)
        
        if rest_keys:
            low_key, _type = _type._slice(rest_keys_valid) 
            return (_full_add(_idx, low_key[0]), low_key[1]), _type 
        else:
            return (_idx, len(_type)), _type
            
    def __str__(self, data: LogicData = None):
        T = f"{self._length}x{self._T}"
        if data is None:
            return T 
        else:
            return f"{T}({data})" 
        
    
class Struct(LogicType):
    """ 1-d struct

    ex. Struct(
            a = UInt[3]             @2,
            b = (3,4) * UInt[4]     @7,
            c = Struct(
                x = UInt[1],
                y = UInt[2]
            )                       @14
        )
    """
    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs, 'empty struct'
        self._fields: Dict[str, Tuple[SignalType, int]] = {}
        self._width: int = 0 
        
        curr_position = 0 
        for k, v in kwargs.items():
            if isinstance(v, tuple):
                T, pos = v
                assert pos >= 0 and isinstance(T, SignalType)
                self._fields[k] = v
                curr_position = pos + len(T)
            else:
                assert isinstance(v, SignalType) 
                self._fields[k] = (v, curr_position)
                curr_position += len(v)
            self._width = max(self._width, curr_position)

     
    def _eval(self, v: Union[int, str, Array, dict, Logic]) -> Logic:
                
        if not isinstance(v, (Array, dict)):
            return UInt[len(self)]._eval(v)
        else:
            v = Array(v)
            ret_v = gmpy2.mpz(0)
            ret_x = gmpy2.mpz(0)
            for key, value in v._items():
                T, start = self._fields[key]
                _v: Logic = T._eval(value) 
                _w: int = len(T) 
                ret_v = ret_v & ~(gmpy2.bit_mask(_w) << start) | _v.v << start 
                ret_x = ret_x & ~(gmpy2.bit_mask(_w) << start) | _v.x << start 
            return Logic(ret_v, ret_x) 
    
    def __call__(
        self, 
        v: Union[Logic, Reader, Iterable]=0, 
        *, 
        name: str = 'vector'
    ) -> Reader:
        
        if isinstance(v, Reader):
            v = v._data  
            assert isinstance(v, LogicData)   
            assert len(v) == len(self) 
            assert v._type._storage == 'packed'
            return Reader(data=v, type=self, name=name)
        else:
            _v = self._eval(v)
            return Reader(data=LogicData(_v.v, _v.x), type=self, name=name)
        
    def _slice(self, keys: Union[str, tuple]) -> Tuple[tuple, SignalType]: 
        
        if isinstance(keys, tuple):
            assert len(keys) > 1
            curr_key = keys[0] 
            rest_keys = keys[1:]
        else:
            curr_key = keys 
            rest_keys = ()
            
        rest_keys_valid = rest_keys[0] if len(rest_keys) == 1 else rest_keys 

        if isinstance(curr_key, str):
            t, position = self._fields[curr_key]
            if not rest_keys:
                return (position, len(t)), t 
            else:
                low_key, _type = t._slice(rest_keys_valid)
                return (_full_add(position, low_key[0]), low_key[1]), _type  
        else:
            assert len(rest_keys) == 0
            return UInt[len(self)]._slice(curr_key)
            
        
    def __str__(self, data=None):
        body = ''.join(f'[{v[1]+len(v[0])-1}:{v[1]}] `{k}` = {v[0]}\n' for k, v in self._fields.items())
        body = '  '.join(body.splitlines(keepends=True))
        T = f'Struct{{\n  {body}}}'
        if data is None:
            return T 
        else:
            return f"{T}({data})"
        

def _full_add(x: Union[int, Reader], y: Union[int, Reader]):
    if isinstance(x, int) and isinstance(y, int):
        return x + y 
    return AddFull(x, y)

def _full_mul(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return x * y 
    def twoN(a):
        # whether a == 2**n
        bits = []
        while a > 0:
            bits.append(a & 1)
            a >>= 1 
        if sum(bits) == 1:
            return len(bits) - 1 
        return None 
    if isinstance(x, int):
        x, y = y, x 
    if isinstance(y, int):
        if (n:=twoN(y)) is not None:
            return Cat(UInt[n](0), x) 
    return MulFull(x, y)
    
        

class MemType(SignalType):
    def __init__(self, shape: Tuple[int,...],  T: LogicType):
        """ 
        ex. shape=(1024,), T = UInt[8] --> self._shape = (1024,8)
        """
        super().__init__()
        assert isinstance(T, LogicType) 
        assert shape 

        self._T = T 
        self._shape = (*shape, len(T)) 
        self._width = len(T) 
        self._idxes: List[int] = []
        for i in reversed(shape):
            self._idxes.insert(0, self._width)
            self._width *= i  
        self._storage = 'unpacked'
        

    def __len__(self) -> int:
        return self._width 

    # def _getval_py(self, data: MemData, immd_key: Optional[Tuple[int]]) -> Logic:
    #     """ get integer value. out of range bits are 0 
        
    #     return: gmpy2.mpz 
    #     """
    #     if immd_key is None: 
    #         return data.v[0:self._width] 
    #     else:
    #         start = 0
    #         for x, w in zip(immd_key, self._idxes):
    #             start += x * w  
    #         return data.v[start:start+self._shape[-1]]

    # def _setval_py(self, data: MemData, immd_key: Optional[Tuple[int]], v: Logic) -> bool: 
    #     """ set some bits, reutrn True if value changes 

    #     immd_key: maybe mpz, turn into int
    #     """ 
    #     target: gmpy2.xmpz = data.v 
    #     if immd_key is None:
    #         odd = target.copy()
    #         target[0:self._width] = v  
    #         return target != odd

    #     v = v & gmpy2.bit_mask(self._shape[-1]) 
    #     start = 0
    #     for idx, max_idx, w in zip(immd_key, self._shape, self._idxes):
    #         if idx >= max_idx:
    #             return False  
    #         start += idx * w  
    #     stop = start + self._shape[-1] 

    #     start = int(start)
    #     stop = int(stop) 
    #     print(immd_key, v, self._sess.sim_py.t)
    #     if target[start:stop] == v:
    #         return False 
    #     else:
    #         target[start:stop] = v 
    #         return True


    
    def _slice(self, high_key) -> Tuple[Tuple[Union[int, Reader], int], SignalType]:
        if not isinstance(high_key, tuple):
            high_key = (high_key,)
        
        assert len(high_key) == len(self._idxes)
        for idx, max_idx in zip(high_key, self._shape):
            if isinstance(idx, int): 
                assert idx < max_idx
            else:
                assert isinstance(idx, Reader) 
        
        return high_key, self._T

    
    def _eval(self, v: Union[int, str, Iterable]) -> Logic:
        v = ToArray(v)
        ret = gmpy2.xmpz(0)
        if isinstance(v, Array):
            assert v._shape == self._shape 
            w = self._shape[-1]
            start = 0
            temp = UInt[w]
            for i in v._flat:
                x = temp._eval(i)
                if x != 0:
                    ret[start:start+w] = x 
                start += w 
            return gmpy2.mpz(ret)
        else:
            return UInt[len(self)]._eval(v)

    def __call__(self, v: int = 0, *, name: str = 'mem') -> Reader:
        _v = self._eval(v)
        return Reader(data=MemData(_v, self._shape), type=self, name=name)

    def __str__(self, data: LogicData = None):
        T = f"Mem({self._shape})"
        if data is None:
            return T 
        else:
            return f"{T}({data})" 

    def _verilog_name(self, x: Union[Reader, SignalData], m: sv.sv) -> str: 
        return
        if self._shape[-1] == 1:
            verilog_width = ''
        else:
            verilog_width = f'[{self._shape[-1]-1}:0]'

        array_shape = ''
        for i in self._shape[:-1]:
            array_shape = f'[0:{i-1}]' + array_shape 

        if isinstance(x, SignalData):
            ret = m.new_name(x, x._name)  
            m.new_signal(x, f'{x.writer._driver._netlist} {verilog_width} {ret} {array_shape}') 
            return ret
        elif isinstance(x, Reader):  
            data = x._data 
            assert isinstance(data, MemData)
            origin = m.get_name(data) 
            assert origin
            # same bit width 
            assert data.writer._type._shape == self._shape 
            m.update_name(x, origin)
            return origin

    def _verilog_key(self, low_key: Optional[tuple], gate: Gate) -> str:
        return 
        if low_key is None:
            return ''
        else:
            assert len(low_key) == len(self._idxes)
            ret = ''
            for i in low_key:
                if isinstance(i, Reader):
                    i = gate.get_name(i)
                else:
                    i = str(i)
                ret += f'[{i}]'
            return ret

#----------------------------------
# Gate
#----------------------------------

class Gate(HGL):
    """ hardware gate 

    no __init__, initialization defined in __new__ & __head__ 

    id: 
        default id of timing configuration 
    timing: 
        default timing configuration 
    iports:
        input signals 
    oports:
        output signals 
    """

    id = 'Gate'
    timing = {'delay': 1} 
    delay = 1 
    trace = 'no trace'

    iports: Dict[Reader, None] 
    oports: Dict[Writer, None]
    _sess: _session.Session 

    _netlist: str = 'logic'   # verilog netlist type of output signal

    
    def __head__(self, *args, **kwargs) -> Union[Reader, Tuple[Reader]]:
        """ initializating,  return Signal or Tuple[Signal]
        """
        pass

    def read(self, x: Optional[Reader]) -> Optional[Reader]:
        """ add x as input signal
        """  
        if x is None:
            return x 
        if self not in x._driven:
            x._driven[self] = 1 
        else:
            x._driven[self] += 1
        self.iports[x] = None
        return x 
    
    def write(self, x: Union[Reader, Writer]) -> Writer:
        """ add x as output signal
        """ 
        if isinstance(x, Reader):
            if x._data.writer is not None:
                raise Exception('cannot drive signal that already has driver')
            ret = Writer(x._data, x._type, self) 
            self.oports[ret] = None 
            return ret 
        elif isinstance(x, Writer):
            assert x._driver is self 
            self.oports[x] = None 
            return x
        
    def __new__(cls, *args, **kwargs):
        
        self = object.__new__(cls) 
        self._sess._add_gate(self) 
        self.iports = {}
        self.oports = {}
        ret = self.__head__(*args, **kwargs)
        # get timing config
        timing: dict = self._sess.timing.get(self.id) or self.timing 
        if 'delay' in timing:
            self.delay = timing['delay']         
        # ---------------------------------
        if self._sess.verbose_hardware:
            self._sess.print(f'{self}, {self.id}={self.timing}')
        if self._sess.verbose_trace:
            self.trace = utils.format_hgl_stack(2,4)
        #----------------------------------
        return ret 
    
    
    def forward(self) -> None:
        """ called by simulator when inputs changing or after init 
        """
        pass

    def dump_cpp(self):
        pass
    
    def dump_sv(self, builder: sv.ModuleSV) -> None:
        """ called when emitting verilog 
        
        ex. assign a = b | c;
        """
        pass
        
    def _graphviz(self, g, iports: List[Reader], oports: List[Writer], body = 'Node'):
        newline = '&#92;n'
        iports_str = '|'.join(f"<i{id(i)}> {i._data._name}{newline}{i._type.__str__(i._data)}" for i in iports)
        oports_str = '|'.join(f"<o{id(o._data)}> {o._data._name}{newline}{o._type.__str__(o._data)}" for o in oports)
        label = "{{%s} | %s | {%s}}" % ( iports_str, body, oports_str)
        
        curr_gate = str(id(self))  
        g.node(name = curr_gate, label = label, shape='record', color='blue') 
        
        for signal in iports:
            if signal._data.writer is not None:
                source_gate = str(id(signal._data.writer._driver))
                g.edge(f"{source_gate}:o{id(signal._data)}", f"{curr_gate}:i{id(signal)}")
        
    def dumpGraph(self, g):
        self._graphviz(g, self.iports, self.oports, self.id) 

    def __str__(self):
        return f"{self.__class__.__name__}"




#----------------------------------
# Assignable: Wire & Reg
#----------------------------------
        

class _Assignment(HGL):
    
    def __init__(
        self, 
        key: Optional[tuple], 
        value: Union[Reader, Any],
    ):
        """ target[key] <== value 
        """
        self.key = key
        self.value = value      

    def __str__(self):
        return f'(key={self.key}, value={self.value})' 
    
class _Switch(HGL):
    def __init__(self, sel: Any):
        self.sel = sel 
        self.case_items: List[Tuple[Union[tuple, None, Reader],CondTreeNode]] = []
    
    def __iter__(self):
        return iter(self.case_items) 
    
    def __len__(self):
        return len(self.case_items)

    def append(self, condition, condtree):
        self.case_items.append((condition, condtree))

class CondTreeNode:
    """ store conditions as a tree 
    
    - frames: list of _Switch | _Assignment
        - _Switch: list of (switch_case_item, CondTreeNode)
        - _Assignment: (target, target_key)        
    level of tree is level of condition frame
    """ 
    
    __slots__ = 'gate', 'is_analog', 'is_reg', 'curr_frame', 'frames'
    
    def __init__(self): 
        # whether use existing frame or append a new frame
        self.curr_frame: Optional[hgl_assign.CaseGate] = None 
        self.frames: List[Union[_Assignment, _Switch]] = [] 
    
    def insert(
        self, 
        conds: List[hgl_assign.CaseGate], 
        low_key: Tuple,
        value: Any,
    ) -> None:
        """ 
        check condition frame from layer 1 to n and insert assignment
        """
        # direct assign in current frame 
        if not conds: 
            if low_key is None:     # full assignment without cond
                self.frames.clear()              
            self.frames.append(_Assignment(low_key, value))
            self.curr_frame = None
        else:
            curr_frame = conds[0]
            # start a new frame 
            if self.curr_frame is not curr_frame:
                self.frames.append(_Switch(curr_frame.sel_signal)) 
                self.curr_frame = curr_frame 
            # list of condition
            curr_case_items = curr_frame.branches 
            # list of (condition, CondTreeNode)
            last_frame: _Switch = self.frames[-1]
            # update frame to catch previous conditions 
            if len(curr_case_items) > len(last_frame):
                for i in range(len(last_frame), len(curr_case_items)):
                    last_frame.append(curr_case_items[i][0], CondTreeNode())
            # recursive
            _, next_tree_node = last_frame.case_items[-1]
            next_tree_node.insert(conds[1:], low_key, value)  

    def merge(self, other: CondTreeNode):
        self.frames = other.frames + self.frames 
        self.read(other)
        
    def __str__(self) -> str: 
        def cond_str(cond, sel):
            if cond is None:
                return '_:'
            else:
                return f"{sel} == {'|'.join(str(x) for x in cond[1:])}:"
        body = []
        body.append('┌──────────────────────────────────────────────────')
        for i, branches in enumerate(self.frames): 
            if isinstance(branches, _Assignment):
                body.append(f'│☆ [{branches.key}] <- {branches.value}')
            else:
                for cond, node in branches:
                    body.append('│'+cond_str(cond, branches.sel))
                    body.append('│  '+'│  '.join(str(node).splitlines(keepends=True)))
                if i < len(self.frames) - 1:
                    body.append('├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌')
        body.append('└──────────────────────────────────────────────────')
        return '\n'.join(body) 
    
    def _dump_sv(self, target: Reader, builder: sv.ModuleSV, op: str = '=') -> str:
        """
        ex. 
        out = 0
        if (x == 1) begin
            out[1] = 1 
            if (y==1) begin 
                out = 2 
            end 
        end
        """
        
        ret = []
        for i in self.frames:   # no cond
            if isinstance(i, _Assignment): 
                left = target._data._dump_sv_slice(i.key, builder=builder) 
                right = builder.get_name(i.value)
                ret.append(f'{left} {op} {right};')  # x[idx +: 2] = y;
            else:       # case frame
                ret.append(f'case({builder.get_name(i.sel)})') 
                for item, node in i:  
                    _body = node._dump_sv(target, builder, op) 
                    _body = '  ' + '  '.join(_body.splitlines(keepends=True))
                    if item is None:  
                        _item = 'default'
                    elif isinstance(item, tuple): 
                        _item = ','.join(builder.get_name(j) for j in item) 
                    else: 
                        _item = builder.get_name(item)
                    ret.append(f'{_item}: begin')
                    ret.append(_body)
                    ret.append('end')
                ret.append('endcase')
        return '\n'.join(ret)
        
        
        

# TODO memtype
@dispatch('Slice', Any) 
class _Slice(Gate):
    
    id = 'Slice'
    
    def __head__(self, signal: Reader, key: Any = None, id='') -> Reader:
        assert isinstance(signal._data, LogicData) and signal._type._storage == 'packed'
        self.id = id or self.id
        self.input = self.read(signal)
        low_key, T = signal._type._slice(key)
        self.low_key = low_key 
        ret = T()
        self.output = self.write(ret)
        return ret 
    
    def forward(self):
        out = self.input._data._getval_py(self.low_key)
        self.output._data._setval_py(out, dt = self.delay, trace=self)

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV): 
        x =self.input._data._dump_sv_slice(self.low_key, builder)
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 


# class Assignable(Gate):
#     """ wire, reg, ..., multiple inputs, single output  
#     """
    
#     condtree: CondTreeNode 
#     output: Writer 

#     def __part_select__(self, signal: Reader, keys: Any) -> Reader:  
#         return _part_select_wire(signal, keys)

#     def __partial_assign__(
#         self, 
#         signal: Reader,
#         conds: List[Reader],
#         value: Union[Reader, int, Any], 
#         keys: Optional[Tuple[Union[Reader, int],int]], 
#     ) -> None: 
#         """ 
#         TODO casted type bit-width mismatch
#         """ 
#         assert signal._data.writer._driver is self
#         # get simplified_key 
#         if type(keys) is SignalKey:
#             low_key, T = None, self.output._type
#         else:
#             low_key ,T = signal._type._slice(keys) 
#         # get valid immd
#         if not isinstance(value, Reader):
#             value = T._eval(value)

#         self.condtree.insert(conds, _Assignment(low_key, value, None)) 

#     def __merge__(self, other: Assignable):
#         raise Exception()
               

class Assignable(Gate):
    """ base class of wire, reg, tri 

    by default, wire|reg has initial value of 0 or other user-defined value
    tri|trior|tirand always has initial value of x 
    """
    
    output: Writer 
    branches: List[Optional[Tuple[Reader, Any, Any]]]  # (cond, key, value) 
    condtree: CondTreeNode

    def __partial_assign__(
        self, 
        target: Reader,                     # target signal 
        acitve_signals: List[Reader],       # [None, signal, signal, ...]
        cond_stacks: List[hgl_assign.CaseGate],        
        value: Union[Reader, int, Any], 
        high_key: Optional[Tuple[Union[Reader, int],int]], 
    ) -> None: 
        assert target._data.writer._driver is self
        # get simplified_key 
        low_key ,T = target._type._slice(high_key) 
        # get valid immd
        if not isinstance(value, Reader):
            value = T._eval(value) 
        if acitve_signals[-1] is None and low_key is None and not isinstance(self, Analog): 
            self.branches.clear() 
        if not isinstance(self, _Reg):    # sensitive
            if isinstance(acitve_signals[-1], Reader):
                self.read(acitve_signals[-1])  
            if low_key is not None:
                for i in low_key:
                    if isinstance(i, Reader):
                        self.read(i)
            if isinstance(value, Reader):
                self.read(value)
        self.branches.append((acitve_signals[-1], low_key, value))  
        self.condtree.insert(cond_stacks, low_key, value)

    def _sv_assignment(self, assignment: Tuple, builder: sv.ModuleSV, op='=') -> str:
        """ conditioanl dynamic partial assignment  
        assignment:
            (cond, key, value)
        return: 
            if (cond) begin
                x[idx +: 2] = y;
            end
        """
        cond, key, value = assignment  
        target = self.output._data._dump_sv_slice(key, builder=builder) 
        if cond is None:
            return f'{target} {op} {builder.get_name(value)};'  
        else:
            return f'if({builder.get_name(cond)}) {target}{op} {builder.get_name(value)};'

    def __merge__(self, other: Assignable):
        # for analog
        raise NotImplementedError()

        
class _Wire(Assignable): 
    
    id = 'Wire'          
  
    def __head__(
        self, 
        x: Reader, 
        *, 
        id: str = '', 
        next: bool = False, 
    ): 
        """ turn into wire
        
        if next, return new signal, width=len(x); 
        else insert a wire between input x and its writer, width=len(x._data)
        
        branches: (active signal, low_key, value)
            - active signal: UInt[1]|None
            - low_key: None | signals | immd, depends on SignalType
            - value: signal|immd depends on SignalType
        """
        self.id = id or self.id    # timing config id   
        self.branches: List[Tuple[
            Union[Reader, None], 
            Union[None,Tuple],
            Union[Reader, Logic]
        ]] = []   
        self.condtree = CondTreeNode()
        self.output: Writer = None
        
        # insert a wire
        if not next:
            if x._data.writer is None:
                self.output = self.write(x)  # width = len(x)
                self.branches.append((None, None, x._data._getval_py())) 
                self.condtree.insert([], None, x._data._getval_py())
                return x 
            elif not isinstance(x._data.writer._driver, Assignable):
                # driver:Gate -> new_x:SignalData -> self:Gate -> x:SignalData
                new_x = x._data.writer._type()
                x._data.writer._exchange(new_x._data)
                self.branches.append((None, None, self.read(new_x))) 
                self.condtree.insert([], None, new_x)
                self.output = self.write(Writer(x._data, new_x._type, driver=self))
                return x 
            else:
                raise ValueError(f'singal {x} is already assignable')
        else:
            ret = x._type()
            self.output = self.write(ret) 
            self.branches.append((None, None, self.read(x))) 
            self.condtree.insert([], None, x)
            return ret
            
            
    def forward(self):
        """ simulation function 
        """
        # no cond, no key, direct assignment
        if len(self.branches) == 1:
            data = self.branches[-1][-1]  
            if isinstance(data, Reader):
                data = data._data._getval_py()
            self.output._data._setval_py(data, dt = self.delay, trace=self)
        else:
            self.output._data._setval_py(self.branches, multiple=True, dt=self.delay, trace=self)
        # active == 1: cover; active == x: merge; mask: merge
        # mask = gmpy2.mpz(0)
        # v = Logic(0,0)
        # for active, low_key, value in self.branches:
        #     active: LogicData = active._getval_py()
        #     if active:
        #         _mask, _v = self.output._type._setval_py(low_key, value)
        #         mask |= _mask 
        #         v &= ~_mask
        #         v |= _v 
        #     elif active._unknown: 
        #         _mask, _v = self.output._type._setval_py(low_key, value)
        #         mask |= _mask
        #         v @= _v 
        
        # self.output._setval_py(v, dt=self.delay, mask = mask)
        # active = []
        # # TODO record different cond frames. for tri-state, warning or merge; for wire, has priority
        # # multiple partial assign from multiple frames
        # # use active from back to front

        # assignments: List[_Assignment] = [] 
        # self.condtree.eval(assignments)
        # if len(assignments) == 1:
        #     key, value = assignments[0].eval()
        #     self.output._setval_py(value, self.delay, key)  
        # else:
        #     _data = self.output._data.copy()  # FIXME wrong, current output is not the future output 
        #     _type = self.output._type
        #     for assignment in reversed(assignments):
        #         key, value = assignment.eval()
        #         _type._setval_py(_data, key, value) 
        #     value = _type._getval_py(_data, None) 
        #     self.output._setval_py(value, self.delay, None)          

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        # TODO 
        """
        always_comb begin 
            x = '0;
            if (cond1) begin 
                x = v1;
            end
            if (cond2) begin 
                x[idx +: 3] = v2;
            end
            if (cond3) begin 
                x = v3;
            end
        end
        """ 
        # body = [self._sv_assignment(i, builder=builder) for i in self.branches] 
        # body.insert(0, 'always_comb begin')
        # body.append('end')
        # builder.gates[self] = '\n'.join(body)
        body = self.condtree._dump_sv(self.output, builder=builder, op='=') 
        body = '  ' + '  '.join(body.splitlines(keepends=True))
        ret = '\n'.join(['always_comb begin', body , 'end']) 
        builder.Block(self, ret)
        
    def __str__(self):
        return f'Wire(name={self.output._data._name})'



class _Reg(Assignable): 
  
    id = 'Reg'         # id
  
    def __head__(self, 
            x: Reader, 
            *, 
            id: str = '', 
            next: bool = False, 
            clock: Tuple[Reader, int] = ...,
            reset: Optional[Tuple[Reader, int]] = ...
        ) -> Reader: 
        """ 
        default:
          - clodk edge: posedge 
          - reset level: 1
          - reset value = inital value
          - input connect: x | self 
          
        reset: None | (rst, 1) | (rst, 0)
        clock: (clk, 0) | (clk, 1)
        """
        self.id = id or self.id
        self.branches: List[Tuple[
            Union[Reader, None], 
            Union[None,Tuple],
            Union[Reader, Logic]
        ]] = []  
        self.condtree = CondTreeNode()
        
        self.posedge_clk: List[Reader, bool] = []   # store previous value
        self.negedge_clk: List[Reader, bool] = [] 
        self.pos_rst: Reader = None
        self.neg_rst: Reader = None
        self.reset_value: Logic = x._data._getval_py()
        
        if clock is ...:   clock = self._sess.module.clock 
        if reset is ...:   reset = self._sess.module.reset

        # record initial value of clock 
        clk_init = clock[0]._data._getval_py() 
        if clk_init.x:
            _clk_init = False 
        else:
            _clk_init = clk_init.v > 0
        if clock[1]:
            self.posedge_clk = [self.read(clock[0]), _clk_init] 
        else:
            self.negedge_clk = [self.read(clock[0]), _clk_init] 
            
        # reset signal
        if reset is None:
            pass 
        elif reset[1]:
            self.pos_rst = self.read(reset[0])
        else:
            self.neg_rst = self.read(reset[0])
        
        # turn a constant signal into reg, and connect output to input
        if not next:
            if x._data.writer is not None:
                raise ValueError('Reg(signal) requires constant signal')
            self.output = self.write(x) 
            self.branches.append((None, None, x)) 
            self.condtree.insert([], None, x)
            return x  
        # reg next
        else:
            ret = x._type()
            self.output = self.write(ret) 
            self.branches.append((None, None, x)) 
            self.condtree.insert([], None, x)
            return ret 

    def forward(self):
        # TODO x state
        if (rst:=self.pos_rst) is not None: 
            if rst._data._getval_py() == Logic(1,0):
                self.output._data._setval_py(self.reset_value, dt = self.delay, trace=self) 
                return
        if (rst:=self.neg_rst) is not None:
            if rst._data._getval_py() == Logic(0,0):
                self.output._data._setval_py(self.reset_value, dt = self.delay, trace=self) 
                return 
        # don't store x value, let x means keep unchanged! so that 0-x-1 is posedge
        if s:=self.posedge_clk: 
            clk, odd = s 
            new = clk._data._getval_py() 
            if new.x:
                return 
            s[1] = new.v > 0
            if not (odd==False and s[1]==True):
                return    
        if s:=self.negedge_clk:
            clk, odd = s 
            new = clk._data._getval_py()  
            if new.x:
                return
            s[1] = new.v > 0
            if not (odd==True and s[1]==False):
                return  
            
        # assignments: List[_Assignment] = []  
        # self.condtree.eval(assignments)   # 800ms  

        # if len(assignments) == 1:
        #     key, value = assignments[0].eval()
        #     self.output._setval_py(value, self.delay, key) 
        # else: 
        #     _data = self.output._data.copy()
        #     _type = self.output._type  
        #     for assignment in reversed(assignments):
        #         key, value = assignment.eval() 
        #         _type._setval_py(_data, key, value)
        #     value = _type._getval_py(_data, None)
        #     self.output._setval_py(value, self.delay, None)
        if len(self.branches) == 1:
            data = self.branches[-1][-1]  
            if isinstance(data, Reader):
                data = data._data._getval_py()
            self.output._data._setval_py(data, dt = self.delay, trace=self)
        else:
            self.output._data._setval_py(self.branches, multiple=True, dt=self.delay, trace=self)

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV): 
        """
        always_ff @(posedge clk) begin 
            if (reset) out <= init; 
            else begin 

            end 
        end 
        """
        # triggers = [] 
        # has_reset = False
        # if (rst:=self.pos_rst) is not None: 
        #     triggers.append(f'posedge {builder.get_name(rst)}')  
        #     reset = builder.get_name(rst)
        #     has_reset = True 
        # elif (rst:=self.neg_rst) is not None:
        #     triggers.append(f'negedge {builder.get_name(rst)}') 
        #     reset = f'!{builder.get_name(rst)}'
        #     has_reset = True  
            
        # if self.posedge_clk:
        #     triggers.append(f'posedge {builder.get_name(self.posedge_clk[0])}')
        # else:
        #     triggers.append(f'negedge {builder.get_name(self.negedge_clk[0])}')
        # triggers = ' or '.join(triggers)
        
        # out = builder.get_name(self.output)
        # body = '\n'.join(self._sv_assignment(i, builder=builder,op='<=') for i in self.branches)   
        # if has_reset:
        #     body = f'if ({reset}) {out} <= {builder.get_name(self.reset_value)}; else begin\n{body}\nend'
        # body = '  ' + '  '.join(body.splitlines(keepends=True))
        # res = '\n'.join([
        #     # FIXME modelsim error
        #     # f'initial begin {out} = {self.reset_value}; end',
        #     f'always_ff @({triggers}) begin',
        #     body, 
        #     'end'
        # ])
        # builder.Block(self, res)


        triggers = [] 
        has_reset = False
        if (rst:=self.pos_rst) is not None: 
            triggers.append(f'posedge {builder.get_name(rst)}')  
            reset = builder.get_name(rst)
            has_reset = True 
        elif (rst:=self.neg_rst) is not None:
            triggers.append(f'negedge {builder.get_name(rst)}') 
            reset = f'!{builder.get_name(rst)}'
            has_reset = True  
            
        if self.posedge_clk:
            triggers.append(f'posedge {builder.get_name(self.posedge_clk[0])}')
        else:
            triggers.append(f'negedge {builder.get_name(self.negedge_clk[0])}')
        triggers = ' or '.join(triggers)
        
        out = builder.get_name(self.output)
        body = self.condtree._dump_sv(self.output, builder=builder, op='<=')
        body = '    ' + '    '.join(body.splitlines(keepends=True))
        if has_reset:
            body = f'  if ({reset}) {out} <= {builder.get_name(self.reset_value)}; else begin\n{body}\n  end'

        ret = '\n'.join([
            # FIXME modelsim error
            # f'initial begin {out} = {self.reset_value}; end',
            f'always_ff @({triggers}) begin',
            body, 
            'end'
        ]) 
        builder.Block(self, ret)
            

    def __str__(self):
        return f'Reg(name={self.output._data._name})'



class _Latch(Assignable):
    
    id = 'Latch'
    delay: int 
    condtree: CondTreeNode 
    output: Writer 
    _netlist = 'logic'
    
    def __head__(self, x: Reader, *, id: str = '', enable: Tuple[Reader, int] = None) -> Reader:
        """  
            always_latch begin
                if (enable) begin
                    a_latch <= something;
                end
            end
        """
        self.id = id or self.id  
        self.condtree = CondTreeNode(gate=self, is_analog=False, is_reg=False)
        # enable signal, no trigger 
        assert x._data.writer is None  
        self.output = self.write(x)
        return x 
        

    def forward(self):
        assignments: List[_Assignment] = [] 
        self.condtree.eval(assignments)

        assignments: List[_Assignment] = [] 
        self.condtree.eval(assignments)

        if len(assignments) == 0:
            return 
        elif len(assignments) == 1:
            key, value = assignments[0].eval()
            self.output._setval_py(value, self.delay, key)  
        else:
            _data = self.output._data.copy()
            _type = self.output._type
            for assignment in reversed(assignments):
                key, value = assignment.eval()
                _type._setval_py(_data, key, value) 
            value = _type._getval_py(_data, None) 
            self.output._setval_py(value, self.delay, None)      

                
    def dump_sv(self, v: sv.Verilog) -> str:
        body = self.condtree.dumpVerilog('=') 
        body = '  ' + '  '.join(body.splitlines(keepends=True))
        return '\n'.join(['always_latch begin', body , 'end'])


    def __str__(self):
        return f'Latch{{{self.output}}}'



class Mem(Assignable):

    id = 'Mem'          
    delay: int          
    condtree: CondTreeNode 
    _netlist = 'logic'   
    _sess: _session.Session 

    def __head__(self, x: Reader, *, id: str='', clock: Tuple[Reader, int] = ...) -> Reader:
        assert isinstance(x._type, MemType) and x._data.writer is None 

        self.id = id or self.id
        self.condtree =  CondTreeNode(gate=self, is_analog=False, is_reg=True)

        self.posedge_clk: List[Reader, gmpy2.mpz] = []
        self.negedge_clk: List[Reader, gmpy2.mpz] = [] 

        if clock is ...:   clock = self._sess.module.clock 
        if clock[1]:
            self.posedge_clk = [self.read(clock[0]), clock[0]._getval_py()] 
        else:
            self.negedge_clk = [self.read(clock[0]), clock[0]._getval_py()] 

        self.output = self.write(x)
        return x 


    def __partial_assign__(
        self, 
        signal: Reader,
        conds,
        value: Union[Reader, int, Any], 
        keys: Optional[Tuple[Union[Reader, int],int]], 
    ) -> None: 
        assert signal._data.writer._driver is self
        # get simplified_key 
        if type(keys) is SignalKey:
            low_key, T = None, self.output._type
        else:
            low_key ,T = signal._type._slice(keys) 
        # get valid immd
        if not isinstance(value, Reader):
            value = T._eval(value)

        self.condtree.insert(conds, _Assignment(low_key, value, None)) 

    def forward(self):
        if s:=self.posedge_clk: 
            clk, odd = s 
            new = clk._getval_py()
            s[1] = new 
            if not (odd==0 and new != 0):
                return    
        if s:=self.negedge_clk:
            clk, odd = s 
            new = clk._getval_py() 
            s[1] = new  
            if not (odd != 0 and new == 0):
                return  
            
        assignments: List[_Assignment] = []  
        self.condtree.eval(assignments)   # 800ms   
        if not assignments: 
            return
        # TODO simulation error if more than one assignments 
        key, value = assignments[0].eval()
        self.output._setval_py(value, self.delay, key) 
            
    
    def dumpVerilog(self, v: sv.Verilog) -> str: 
        triggers = [] 
            
        if self.posedge_clk:
            triggers.append(f'posedge {self.get_name(self.posedge_clk[0])}')
        else:
            triggers.append(f'negedge {self.get_name(self.negedge_clk[0])}')
        triggers = ' or '.join(triggers)
        
        out = self.get_name(self.output)
        body = self.condtree.dumpVerilog('<=')
        body = '    ' + '    '.join(body.splitlines(keepends=True)) 

        return '\n'.join([
            # FIXME modelsim error
            # f'initial begin {out} = {self.reset_value}; end',
            f'always_ff @({triggers}) begin',
            body, 
            'end'
        ])
            


class Analog(Assignable): 
    pass



class _WireTri(Analog):  
    
    id = 'Wtri'         # id
    delay: int          # timing
    condtree: CondTreeNode 
    output: Writer 
    _netlist = 'logic'   # netlist type
  
    def __head__(self, x: Reader, *, id: str) -> Reader: 

        self.id = id or self.id  
        self.condtree = CondTreeNode(gate=self, is_analog=True, is_reg=False)
        assert x._data.writer is None  

        self.output = self.write(x)
        # default pull up/down
        self.default = x._getval_py()
        return x 
        

    def __merge__(self, other: Assignable): 
        assert isinstance(other, _WireTri) and len(other.output._type) == len(self.output._type)
        self.condtree.merge(other.condtree)   
        other.output._exchange(self.output._data)   
        other.delete()

    def delete(self): 
        for i in self.iports:
            i._driven.pop(self)
        for o in self.oports: 
            o._delete() 
        self.iports.clear()
        self.oports.clear()
        self._sess._remove_gate(self)


    def __partial_assign__(
        self, 
        signal: Reader,
        conds: List,
        value: Union[Reader, int, Any], 
        keys: Optional[Tuple[Union[Reader, int],int]], 
    ) -> None: 
        """ 
        TODO casted type bit-width mismatch
        """ 
        assert signal._data.writer._driver is self
        # get simplified_key 
        if type(keys) is SignalKey:
            low_key, T = None, self.output._type
        else:
            raise Exception('partial assign to analog wire is invalid')
        # get valid immd
        if not isinstance(value, Reader):
            value = T._eval(value)

        self.condtree.insert(conds, _Assignment(low_key, value, None)) 
        

    def forward(self):
        """ simulation function 
        """
        assignments: List[_Assignment] = [] 
        self.condtree.eval(assignments)
        # TODO simulating error for multiple assignment 
        if not assignments:
            self.output._setval_py(self.default, self.delay, None)
        else:
            key, value = assignments[0].eval()
            self.output._setval_py(value, self.delay, key)  
    
    
    def dumpVerilog(self, v) -> str: 
        body = self.condtree.dumpVerilog('=')  
        body = f"{self.get_name(self.output)} = 'z;\n{body}" 
        body = '  ' + '  '.join(body.splitlines(keepends=True))
        return '\n'.join(['always_comb begin', body , 'end'])
            


class _WireOr(Analog): ... 

class _WireAnd(Analog): ...




@dispatch('Wire', Any)
def _wire(x: Reader, *, id: str = '') -> Reader:
    return _Wire(x, id=id, next=False) 


@dispatch('WireNext', Any)
def _wirenext(x: Reader, *, id: str = '') -> Reader:
    return _Wire(x, id=id, next=True) 


@dispatch('Reg', Any)
def _reg(x: Reader, *, id = '', clock = ..., reset = ...) -> Reader: 
    return _Reg(x, id=id, next=False, clock=clock, reset=reset)


@dispatch('RegNext', Any)
def _regnext(x: Reader,  *, id = '', clock = ..., reset = ...) -> Reader: 
    return _Reg(x, id=id, next=True, clock=clock, reset=reset)


@dispatch('Latch', Any)
def _latch(x: Reader, *, id: str = '') -> Reader:
    return _Latch(x, id=id)



@dispatch('Wtri', Any)
def _tri(x: Reader, id: str = '') -> Reader: 
    return _WireTri(x, id=id)




class Clock(Gate):
    
    id = 'Clock'
    timing = {'low':50, 'high':50, 'phase':0}
    
    def __head__(self, id: str = 'Clock') -> Reader:
        self.id = id 
        ret: Reader = UInt[1](0) 
        ret._data._name = id.lower()
        self.clk = self.read(ret)
        self.clk_w = self.write(ret)
        # get timing
        timing = self._sess.timing.get(self.id) or self.timing 
        self.low: int = timing['low'] 
        self.high: int = timing['high']
        self.phase: int = timing['phase']
        return ret

    def forward(self):
        odd: Logic = self.clk._data._getval_py() 
        if odd.v:  # 1 -> 0
            self.clk_w._data._setval_py(Logic(0,0), dt=self.high, trace=self) 
        else:    # 0 -> 1
            self.clk_w._data._setval_py(Logic(1,0), dt=self.low, trace=self)

    def dump_cpp(self):
        # TODO 
        return super().dump_cpp()
    
    def dump_sv(self, builder: sv.ModuleSV) -> None:
        clk = builder.get_name(self.clk)
        builder.Task(self, f'always begin {clk} = 0; # {self.low}; {clk} = 1; # {self.high}; end')
    
    def dumpVerilog(self, v: sv.Verilog) -> str: 
        clk = self.get_name(self.clk) 
        low = self.low
        high = self.high
        return f'always begin {clk} = 0; # {low}; {clk} = 1; # {high}; end'


class ClockDomain(HGL):
    
    _sess: _session.Session
    
    def __init__(self, clock = ..., reset = ...):
        if clock is ...:
            clock = self._sess.module.clock 
        if reset is ...:
            reset = self._sess.module.reset 
            
        clk, edge = clock 
        assert isinstance(clk, Reader) and isinstance(edge, (int, bool))
        self.clock: Tuple[Reader, int] = clock
        assert reset is None or isinstance(reset[0], Reader)
        self.reset: Optional[Tuple[Reader, int]] = reset  
        
        self._clk_restore = []
        self._rst_restore = []
        
    def __enter__(self):
        self._clk_restore.append(self._sess.module.clock)
        self._rst_restore.append(self._sess.module.reset)
        self._sess.module.clock = self.clock 
        self._sess.module.reset = self.reset

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._sess.module.reset = self._clk_restore.pop()
        self._sess.module.clock = self._rst_restore.pop()
    
    
class BlackBox(Gate):

    id = 'BlackBox'

    def __head__(self, inputs = [], outputs = [], id=''): 
        self.inputs: List[Reader] = []
        self.outputs: List[Writer] = []
        for i in inputs:
            self.inputs.append(self.read(i))
        for i in outputs:
            self.outputs.append(self.write(i))
        self.verilog: List[str] = [] 
        self.builder: sv.ModuleSV = None
        return self 

    def get_name(self, obj):
        return self.builder.get_name(obj) 

    def append(self, *args: str):
        self.verilog.extend(args) 
        
    def forward(self):
        pass
    
    def dump_cpp(self):
        pass

    def dump_sv(self, builder: sv.ModuleSV):  
        self.builder = builder
        self.__body__()
        builder.Task(self, '\n'.join(self.verilog))
        self.builder = None

    def __body__(self):
        return 


def blackbox(f: Callable):
    assert len(inspect.signature(f).parameters) == 1
    type(f.__name__, (BlackBox,),{'__body__':f})() 



"""
@blackbox testbench(self, builder): 
    builder.get_name(dut) 
    builder.get_name(dut.io.x)

    x = self.get_name(dut.x)
    y = self.get_name(dut.y) 
    return f'''
initial begin
    # 3 
    {x} = 1 
end
    '''

"""




@vectorize_first
def setv(left: Reader, right: Any, *, key = None, dt=0) -> None: 
    assert dt >= 0
    if key is None:
        immd: Logic = left._type._eval(right)  
        low_key = None
    else:
        low_key, sub_type = left._type._slice(key)  
        immd: Logic = sub_type._eval(right) 
    
    if HGL._sess.verbose_trace:
        trace = utils.format_hgl_stack(2, 4) 
    else:
        trace = None 
    
    if low_key is None:
        left._data._setval_py(immd, dt=dt, trace=trace)
    else:
        left._data._setval_py(branches=[(None, low_key, immd)],dt=dt, trace=trace)
    
    
    
@vectorize_first
def setr(left: Reader, dt=0) -> Logic: 
    """ random 2-valued logic
    """
    T = left._type 
    v = T._eval(random.randrange(1 << len(T))) 
    if HGL._sess.verbose_trace:
        trace = utils.format_hgl_stack(2, 4) 
    else:
        trace = None
    left._data._setval_py(v, dt=dt, trace=trace)
    return v

@vectorize_first
def setx(left: Reader, dt=0) -> Logic:  
    """ random 3-valued logic
    """
    T = left._type  
    _max = 1 << len(T)
    v = Logic(random.randrange(_max), random.randrange(_max))
    v = T._eval(v) 
    if HGL._sess.verbose_trace:
        trace = utils.format_hgl_stack(2, 4) 
    else:
        trace = None
    left._data._setval_py(v, dt=dt, trace=trace)
    return v

@vectorize 
def getv(signal: Reader, key = None) -> Logic: 
    # key: high key
    if key is None:
        return signal._data._getval_py()
    else:
        low_key, _ = signal._type._slice(key)
        return signal._data._getval_py(low_key)


class posedge(HGL):
    
    def __init__(self, s: Reader):
        
        if not isinstance(s, Reader) or len(s) != 1:
            raise ValueError(f'input should be 1 bit signal')
        self.signal = s 
        
    def __iter__(self):
        if self.signal._data._getval_py() == 0:
            yield self.signal 
        else: 
            while self.signal._data._getval_py() != 0:
                yield self.signal
            yield self.signal
    
    
class negedge(posedge):
        
    def __iter__(self):
        while self.signal._data._getval_py() == 0:
            yield self.signal 
        yield self.signal
        
        
class dualedge(posedge):

    def __iter__(self):
        yield self.signal


import pyhgl.logic.hgl_assign as hgl_assign
