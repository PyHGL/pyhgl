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

import gmpy2
import random

from pyhgl.array import *
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic._config as config
import pyhgl.logic.module_cpp as cpp
import pyhgl.logic._session as session  
import pyhgl.logic.module_sv as sv 
import pyhgl.logic.simulator_py as simulator
import pyhgl.logic.utils as utils



#------------------
# Literal
#------------------
    
class Logic(HGL):
    """
    3-valued logic literal: 0, 1, X 

    v, x
    0  0 -> 0 
    1  0 -> 1 
    0  1 -> X 
    1  1 -> X 

    XXX slicing negative mpz return incorrect value
    """
    
    __slots__ = 'v', 'x'

    def __init__(
            self, 
            v: Union[int, str, gmpy2.mpz, Logic], 
            x: Union[int, gmpy2.mpz] = gmpy2.mpz(0),
        ) -> None:
        """ ex. Logic(3), Logic(-1), Logic(-1,-1)
        """
        if isinstance(v, gmpy2.mpz) and isinstance(x, gmpy2.mpz):
            self.v = v    
            self.x = x    
        elif isinstance(v, str):        # ignore x
            _v, _x, _ = utils.str2logic(v)
            self.v = gmpy2.mpz(_v)
            self.x = gmpy2.mpz(_x)
        elif isinstance(v, Logic):      # ignore x
            self.v, self.x = v.v, v.x
        else:                           # v, x are int/float
            try:
                self.v = gmpy2.mpz(v)
                self.x = gmpy2.mpz(x)
            except:
                raise ValueError(f'wrong value: {v} or {x}') 
    
    def __bool__(self):
        raise Exception('Logic does not support python bool method')  
    
    def __hash__(self):
        return hash((self.v, self.x))
    
    def __len__(self):
        """ return minimum bit-length of v,x
        """
        return max(utils.width_infer(self.v), utils.width_infer(self.x))

    def to_bin(self, width: int = None) -> str:
        """ ex. width=None: '1x1x10'; width=8: '001x1x10'
        """
        if width is None:
            width = len(self)
        return utils.logic2bin(self.v, self.x, width)
    
    def to_hex(self, width: int = None) -> str:
        """ ex. width=None: '1f'; width=16: '001f'
        """
        if width is None:
            width = len(self)
        return utils.logic2hex(self.v, self.x, width)
    
    def to_int(self, width: int = None) -> Optional[int]:
        """ convert to python int, sign-extended with width
        """
        if width is None:
            if self.x: 
                return None 
            else:
                return int(self.v)
        else:
            mask = gmpy2.bit_mask(width)
            if self.x & mask:               # unknown value
                return None 
            else:
                v = self.v & mask
                if v[width-1]:              # sign-extended
                    return int(v | ~mask)
                else:
                    return int(v)
    
    def split(self, *shape: int) -> Array:
        assert len(shape) == 2, 'currently only support 2d shape'
        ret = Array([])
        n, w = shape 
        mask = gmpy2.bit_mask(n*w)
        v = self.v & mask 
        x = self.x & mask
        assert n > 0 and w > 0, 'negative shape'
        for i in range(n):
            _start, _end = i * w, i * w + w
            ret._append(Logic(
                v[_start:_end],
                x[_start:_end],
            )) 
        return ret
    
    @staticmethod 
    def zeros(*shape) -> Array:
        """ return array of signals
        """
        if isinstance(shape[0], int):   # zeros(2,3,4)
            shape = shape 
        else:                           # zeros((2,3,4))
            assert len(shape) == 1 
            shape = shape[0]
        return Map(lambda _: Logic(gmpy2.mpz(0), gmpy2.mpz(0)), Array(np.zeros(shape), recursive=True))

    
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
        
    def __ne__(self, x: Union[str, int, Logic]) -> bool:
        """ return bool
        """
        other = Logic(x)
        v1, v2 = self.v, other.v 
        x1, x2 = self.x, other.x 
        if x1 == x2:
            return (v1 & ~x1) != (v2 & ~x2)
        else:
            return True 

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
        if l.x:
            return Logic(0, -1)
        else:
            return Logic(
                self.v << l.v,
                self.x << l.v,
            )
    
    def __rshift__(self, l: Logic) -> Logic: 
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
    
    def _merge(self, unknown: bool, mask: gmpy2.mpz, other: Logic) -> Logic:
        """
        mask: positions of bits to merge  
        other: another value, should not overflow
        unknown: override or merge 

        ex. unknown = True, self = 1001x, other = 0x000, mask = 11000 -> xx01x
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
        ret = re.sub('x', '?', utils.logic2bin(self.v, self.x ,self.width))
        return f'BitPat({ret})'  




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
    TODO marked as sensitive
    TODO copy with data
    """

    _sess: session.Session

    __slots__ = 'v', 'x', 'length', '_v', '_x', 'writer', 'reader', '_name', '_module', 'tracker', 'events'
    
    def __init__(self, v: Union[int, gmpy2.mpz], x: Union[int, gmpy2.mpz], length: int):  
        # value
        self.v: gmpy2.mpz = gmpy2.mpz(v)  
        # unknown
        self.x: gmpy2.mpz = gmpy2.mpz(x)
        # bit length 
        self.length = length
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
        self._module: Optional[module_hgl.Module] = None

        # record history values
        self.tracker: simulator.Tracker = None 
        # record triggered coroutine events. also indicates sensitive
        self.events: Optional[List[Generator]] = None
        
    def _dump_sv(self, builder: sv.ModuleSV):
        """ dump declaration to builder. if necessary, also dump assignment/initial block

        ex. logic [7:0] a; assign a = '1;
        """
        raise NotImplementedError() 
    
    def _dump_sv_assign(
            self, 
            low_key: Optional[Tuple], 
            value: Union[Reader, int, Logic], 
            builder: sv.ModuleSV
        ) -> Tuple[str, str]:
        """ ex. mem = '{1,2,3}; x[idx +: 8] <= x_next; 
        return (left,right)
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
        need_merge: bool = False,
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
        return self.length
        # if self.writer is not None:
        #     return len(self.writer)
        # elif self.reader:
        #     return len(next(iter(self.reader)))
        # else:
        #     return 0

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
        
    def _dump_sv_assign(
            self, 
            low_key: Optional[Tuple],
            value: Union[Reader, int, Logic], 
            builder: sv.ModuleSV
        ) -> Tuple[str, str]:
        """ ex. mem = '{1,2,3}; x[idx +: 8] <= x_next; 
        return (left,right)
        """
        left = builder.get_name(self)
        right = builder.get_name(value) 
        if low_key is None:
            return left, right 
        else:
            start, length = low_key 
            if length == 1:
                key_str = f'[{builder.get_name(start)}]'
            else:
                key_str = f'[{builder.get_name(start)}+:{length}]' 
            return f'{left}{key_str}', right
        
        
    def _getval_py(self, low_key: Optional[Tuple] = None) -> Logic:
        """ 
        - low_key: None | (start, length)
            - start: Logic | Signal 
            - length: int
        - return: Logic with certain width
        """
        v = self.v 
        x = self.x 
        if low_key is None:
            return Logic(v, x)
        else:
            start, length = low_key 
            if isinstance(start, Reader):       # dynamic part select
                start = start._data._getval_py() 
                if start.x:                     # unknown key, return unknown
                    return Logic(gmpy2.mpz(0), gmpy2.bit_mask(length)) 
                else:
                    start = start.v 
            end = start + length
            max_width = len(self)
            if start >= max_width:
                return Logic(gmpy2.mpz(0), gmpy2.bit_mask(length)) # all out of range, return unknown 
            elif end <= max_width:
                return Logic(v[start:end], x[start:end])        # in range 
            else:                                               # partly in range
                mask =  gmpy2.bit_mask(end - max_width) << (max_width-start)
                return Logic(v[start:end], x[start:end] | mask)

    def _setval_py(
        self, 
        value: Union[Logic, List], 
        need_merge: bool = False,
        dt: int = 1, 
        trace: Union[None, Gate, str] = None,
    ):
        """ multiple dynamic partial assignment

        value:
            Logic | List[(cond, key, value)]
        need_merge:
            true when there are multiple assignments
        dt: delay
        """ 
        if not need_merge: 
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
                    new = Logic(new.v & mask_full, new.x & mask_full)  # bit truncation
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
        self._sess.sim_py.insert_signal_event(dt, self, data.v, sel=0)
        self._sess.sim_py.insert_signal_event(dt, self, data.x, sel=1)
        # self._sess.sim_py.insert_signal_event(dt, self, data, sel=2) 

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

    TODO 
    better performance: simulator update memdata by (key,value)
    special slice gate for memtype
    """

    __slots__ = 'shape', 'length'

    def __init__(
        self, 
        v: Union[int, gmpy2.mpz, gmpy2.xmpz], 
        x: Union[int, gmpy2.mpz, gmpy2.xmpz],
        shape: Tuple[int,...]
    ) -> None:
        assert len(shape) == 2, f'currently only support 2d array, not {shape}'
        super().__init__(v, x, shape[0] * shape[1])

        self.shape: Tuple[int, int] = shape  
        # must dump to verilog, using `initial` block
        self._module = self._sess.module
    
    def __len__(self):
        return self.length

    def __str__(self):
        ret = utils.logic2str(self.v, self.x, self.length)
        if len(ret) > 400:
            return f'MemData{self.shape}({ret[:400]})'
        else:
            return f'MemData{self.shape}({ret})'
        
    def _dump_sv(self, builder: sv.ModuleSV):
        """ dump declaration to builder. if necessary, also dump constant assignment
        """
        if self not in builder.signals:
            n_unpacked, n_packed = self.shape
            builder.signals[self] = f'logic [{n_packed-1}:0] {{}} [0:{n_unpacked-1}]'

        if self.writer is None:         # initial rom/ram
            _builder = self._module._module 
            if self not in _builder.gates:
                name = _builder.get_name(self)
                ret = ['initial begin']
                for i, v in enumerate(utils.mem2str(self.v, self.x, self.shape)):
                    ret.append(f'  {name}[{i}] = {v};')
                ret.append('end')
                _builder.gates[self] = '\n'.join(ret)

    def _dump_sv_assign(
            self, 
            low_key: Optional[Tuple],
            value: Union[Reader, int, Logic], 
            builder: sv.ModuleSV
        ) -> Tuple[str, str]:
        """ ex. mem = '{1,2,3}; x[idx +: 8] <= x_next; 
        return (left,right)
        """
        left = builder.get_name(self)

        if isinstance(value, Reader):
            right = builder.get_name(value) 
        elif isinstance(value, Logic):
            l = utils.mem2str(value.v, value.x, self.shape)
            right = f"'{{{','.join(l)}}}"
        else:
            raise TypeError(value)

        if low_key is None:
            return left, right 
        else:
            assert len(low_key) == 1
            return f'{left}[{builder.get_name(low_key[0])}]', right

    def _getval_py(self, low_key: tuple = None) -> Logic:
        """ low_key: (idx,)
        """
        if low_key is None:
            return Logic(self.v, self.x)
        else:
            idx = low_key[0]
            if isinstance(idx, Reader):
                idx = idx._data._getval_py() 

        n, w = self.shape
        if idx.x or (idx.v >= n):
            return Logic(gmpy2.mpz(0), gmpy2.bit_mask(w)) 
        else:
            start = idx.v * w
            end = start + w
            ret = Logic(self.v[start:end], self.x[start:end])
            return ret

            
    def _setval_py(
        self, 
        value: Union[Logic, List], 
        need_merge: bool = False,
        dt: int = 1, 
        trace: Union[None, Gate, str] = None,
    ):
        """ key is always required, so value is a list

        value: List[(cond, key, value)]

        TODO unknown state
        """
        if not need_merge:
            data = value
        else: 
            data = Logic(0,0) 
            mask_data = gmpy2.bit_mask(self.shape[-1])
            mask_full = gmpy2.bit_mask(self.length) 
            mask_curr = mask_full
            for cond, key, new in value:
                if isinstance(new, Reader):
                    new: Logic = new._data._getval_py()
                if cond is not None:
                    cond: Logic = cond._data._getval_py()
                    if cond.v == 0 and cond.x == 0:
                        continue
            
                if key is None:
                    new = new 
                    mask_curr = mask_full
                else:
                    idx = key[0]
                    if isinstance(idx, Reader):
                        idx = idx._data._getval_py()
                    else:   # int 
                        idx = Logic(idx, 0) 
                    
                    if idx.x:
                        mask_curr = mask_full 
                        new = Logic(0, mask_full)
                    else: 
                        _n, _w = self.shape 
                        _i = idx.v 
                        if _i >= _n:   # out of range, set nothing
                            continue
                        lshift = _i * _w
                        mask_curr = (mask_data << lshift) & mask_full
                        new = Logic(
                            (new.v << lshift) & mask_curr, 
                            (new.x << lshift) & mask_curr,
                        ) 
                data = data._merge(False, mask_curr, new)
        self._sess.sim_py.insert_signal_event(dt, self, data.v, sel=0)
        self._sess.sim_py.insert_signal_event(dt, self, data.x, sel=1)
        # self._sess.sim_py.insert_signal_event(dt, self, data, sel=2)
            
                
        

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

#-----------------------------------------
# Signal: in/out edge of SignalData
#-----------------------------------------
    

class Writer(HGL):
    """ a writer is an outedge of specific gate 
    """

    _sess: session.Session

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
        new_data._module = self._data._module
        # odd data
        self._data.writer = None  
        self._data._module = None
        
        ret, self._data = self._data, new_data
        return ret  
    
    def _delete(self) -> None:
        """ when deleting a gate
        """
        self._data.writer = None 
        self._data._module = None
        self._data = None
        self._driver = None
        self._type = None
        
    def __len__(self):
        return len(self._type)
    
    def __str__(self):
        return f'{self._driver}.Writer({self._data._module}.{self._data._name})'
    


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
        self._data.writer._driver.__partial_assign__(acitve_signals,cond_stacks,value,high_key)  

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
        return f'{self._name}={self._type.__str__(self._data)}'
    
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

    _sess: session.Session 
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
@vectorize
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
        
        keys: high-level keys, ex. int, signal, slice, str
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

        if isinstance(high_key, (int, str, Logic, Reader)):
            high_key = slice(high_key, None, 1)
        
        if isinstance(high_key, slice):
            start, stop, step = high_key.start, high_key.stop, high_key.step 
            if start is not None and stop is None and step is not None:
                # [signal::8] 
                assert isinstance(step, int) and step > 0 
                assert isinstance(start, (int, str, Logic, Reader))
                if isinstance(start, (int, str)):
                    start = Logic(start).v
                    if start < 0:
                        start = start + self._width
                return (start, step), UInt[step] 
            elif step is not None:
                raise KeyError(f'invalid slicing {high_key}')
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
    def split(self: Reader, n: int = 1) -> Array:
        return Split(self, n)
    
    def zext(self: Reader, w: int):
        return Zext(self, w) 

    def oext(self: Reader, w: int):
        return Oext(self, w)  

    def sext(self: Reader, w: int):
        return Sext(self, w)    
    
    def inside(self: Reader, *args: Any):
        """ reduced equal, x == a || x == b || x == c
        """
        if len(args) == 1:
            return Eq(self, args[0])
        else:
            return Or(Eq(self, ToArray(args))._flat)



class UIntType(LogicType):

    # reduce memory cost
    _cache = {}

    def __init__(self, width: int = 0): 
        assert width > 0
        super().__init__(width)

    def _eval(self, v: Union[int, str, Logic, BitPat]) -> Union[Logic, BitPat]:
        """ 
        called when:
            - `signal <== immd `
            - `signal == immd `
            - `UInt(immd)`

        return a logic value within bit length

        - overflow:
            BitPat not allowed 
            others cut off
        - underflow:
            always zero extended
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
            # _w = max(utils.width_infer(v.v), utils.width_infer(v.x))
        else:
            _v = v
            _x = 0
            # _w = utils.width_infer(_v) 
        # overflow not allowed
        mask = gmpy2.bit_mask(self._width)
        return Logic(_v & mask, _x & mask)

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
            return Reader(data=LogicData(data.v, data.x, len(self)), type=self, name=name)

    def zeros(self, *shape) -> Array:
        """ return array of signals
        """
        if isinstance(shape[0], int):   # zeros(2,3,4)
            shape = shape 
        else:                           # zeros((2,3,4))
            assert len(shape) == 1 
            shape = shape[0]
        return Map(lambda _: self(0), Array(np.zeros(shape), recursive=True))
    
    def __str__(self, data: LogicData = None):
        if data is None:
            return f"UInt[{len(self)}]" 
        else:
            return f"u{utils.logic2str(data.v, data.x, width=len(self))}"

@singleton
class UInt(HGLFunction):
    
    def __getitem__(self, key: int):
        """
        ex. UInt[8] -> UIntType(8), 
        """   
        assert isinstance(key, int) and key > 0
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


@vectorize
def Split(signal: Reader, n: int = 1) -> Array:
    assert signal._type._storage == 'packed' 
    assert isinstance(signal._data, LogicData) 
    signal = UInt(signal)
    ret = []
    for idx in range( (len(signal)+n-1)//n ): 
        ret.append(signal[idx*n:(idx+1)*n]) 
    return Array(ret)

@vectorize
def Zext(signal: Reader, w: int) -> Reader:
    assert signal._type._storage == 'packed'
    _w = len(signal)
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(signal, UInt[w-_w](0))

@vectorize
def Oext(signal: Reader, w: int) -> Reader:
    assert signal._type._storage == 'packed'
    _w = len(signal)
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(signal, UInt[w-_w]((1<<(w-_w))-1)) 

@vectorize
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
            return Reader(data=LogicData(_v.v, _v.x, len(self)), type=self, name=name)
    
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


@singleton
class Vec(HGLFunction):
    
    def __getitem__(self, key: tuple):
        """
        Vec[3,8] -> Vec[3,UInt[8]] 
        Vec[2,3,4]
        """   
        assert isinstance(key, tuple) and len(key) > 1
        T = key[-1]
        if not isinstance(T, SignalType):
            T = UInt[T]
        return T * key[:-1]



    
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
            return Reader(data=LogicData(_v.v, _v.x, len(self)), type=self, name=name)
        
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

    _storage = 'unpacked'

    def __init__(self, *shapes):
        """ 
        ex. self._shape = (1024,8) -> 1024 x UInt[8]
        """
        super().__init__() 
        for i in shapes:
            assert i > 0

        self._shape = tuple(shapes)
        self._length = 1 
        for i in self._shape:
            self._length *= i  
        
    def __len__(self) -> int:
        return self._length 

    def _eval(self, v: Union[int, str, Iterable, Logic]) -> Logic: 
        """ if v is array, zero initialized if cannot fill 
        """
        v = ToArray(v)
        if isinstance(v, Array): 
            _v = gmpy2.mpz(0) 
            _x = gmpy2.mpz(0)
            _w = self._shape[-1]
            _T: UIntType = UInt[_w]
            for i, x in enumerate(v): 
                temp: Logic = _T._eval(x)
                _v |= (temp.v) << (i*_w) 
                _x |= (temp.x) << (i*_w)
            return Logic(_v, _x)
        # regard as UInt
        else:
            return UInt[len(self)]._eval(v) 
 
        
    def _slice(self, high_key) -> Tuple[Tuple[Logic], SignalType]:
        if high_key is None:
            raise TypeError('key required for MemType') 
        elif isinstance(high_key, (int, str, Logic)):
            key = Logic(high_key) 
            assert key.x == 0 
            key = key.v
            if key < 0:
                key += self._shape[0]
            assert 0 <= key < self._shape[0], 'index overflow'
            return (Logic(key), ), UInt[self._shape[-1]]
        elif isinstance(high_key, Reader):
            return (high_key, ), UInt[self._shape[-1]]
        else:
            raise KeyError(high_key)


    def __call__(self, v: Union[int, Array, list] = 0, *, name: str = 'mem') -> Reader:
        _v = self._eval(v)
        return Reader(data=MemData(_v.v, _v.x, self._shape), type=self, name=name)
    

    def __str__(self, data: LogicData = None):
        T = f"Mem({self._shape})"
        if data is None:
            return T 
        else:
            return f"{T}({data})" 

@singleton 
class MemArray(HGLFunction):
    def __getitem__(self, key: tuple):
        assert len(key) == 2, 'only support 2-D unpacked array'
        return MemType(*key)
        


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
    _sess: session.Session 

    _netlist: str = 'logic'   # verilog netlist type of output signal

    # number of unknown values of inputs 
    # if always need to calculate unknown output, set init > 0 
    """
    if (update_v):
        for i in v.driven:
            changed_gates.append(i)
    elif (update_x):
        for i in x.driven:
            i.update_x_count()
            i.x_need = True 
            changed_gates.append(i)
    for i in changed_gates:
        if i.x_count:
            i.forward_vx()
        elif i.x_need:
            i.forward_vx()
        else:
            i.forward_v()
    """
     
    sim_x_count: int            # number of unknown values of input signals 
    sim_x_changed: bool         # input unknown changed at current time slot 
    sim_waiting: bool           # need to execute at current time slot 

    sim_forward_v: Generator    # only update v of outputs 
    sim_forward_vx: Generator   # update v & x of outputs

    __slots__ = 'sim_x_count','sim_x_changed','sim_waiting','sim_forward_v','sim_forward_vx','__dict__'
    
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
        self.sim_x_count = 0 
        self.sim_x_changed = True 
        self.sim_waiting = True 
        self.sim_forward_v = None 
        self.sim_forward_vx = None
        ret = self.__head__(*args, **kwargs)
        # get timing config
        # TODO get in sim_init
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
    
    def sim_init(self) -> None:
        """ get timing, count x, set x_changed, init generator
        """
        # get timing using self.id
        self.timing = self._sess.timing.get(self.id) or self.timing 
        # count x
        for s in self.iports:
            if s._data.x:
                self.sim_x_count += 1 
        self.sim_forward_v = self.sim_v(); next(self.sim_forward_v)
        self.sim_forward_vx = self.sim_vx(); next(self.sim_forward_vx)
        self.sim_x_count += 1000


    def sim_v(self):
        """ called by simulator 
        """
        while 1:
            yield

    def sim_vx(self):
        while 1:
            yield 
            self.forward()
    

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
                left, right = target._data._dump_sv_assign(i.key, i.value, builder=builder) 
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
        
        
@dispatch('Slice', Any) 
class _Slice(Gate):
    
    id = 'Slice'
    
    def __head__(self, signal: Reader, key: Any = None, id='') -> Reader:
        assert isinstance(signal._data, LogicData) and signal._type._storage == 'packed'
        self.id = id or self.id
        self.input = self.read(signal) 
        if self.input._data._module is None:
            self.input._data._module = self._sess.module

        low_key, T = signal._type._slice(key)
        self.low_key = low_key  
        if low_key is not None:
            for i in low_key: 
                if isinstance(i, Reader):
                    self.read(i)   # sensitive
        ret = T()
        self.output = self.write(ret)
        return ret 
    
    def forward(self):
        out = self.input._data._getval_py(self.low_key)
        self.output._data._setval_py(out, dt = self.delay, trace=self)

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV): 
        x, _ = self.input._data._dump_sv_assign(self.low_key, Logic(0,0), builder)
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 


@dispatch('Slice', MemType) 
class _Slice(Gate):
    
    id = 'Slice'
    
    def __head__(self, signal: Reader, key: Any = None, id='') -> Reader:
        assert isinstance(signal._data, MemData)
        self.id = id or self.id
        self.input = self.read(signal) 
        if self.input._data._module is None:
            self.input._data._module = self._sess.module

        low_key, T = signal._type._slice(key)
        self.low_key = low_key  
        if low_key is not None:
            for i in low_key: 
                if isinstance(i, Reader):
                    self.read(i)   # sensitive
        ret = T()
        self.output = self.write(ret)
        return ret 
    
    def forward(self):
        out = self.input._data._getval_py(self.low_key)
        self.output._data._setval_py(out, dt = self.delay, trace=self)

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV): 
        x, _ = self.input._data._dump_sv_assign(self.low_key, Logic(0,0), builder)
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
        acitve_signals: List[Reader],       # [None, signal, signal, ...]
        cond_stacks: List[hgl_assign.CaseGate],        
        value: Union[Reader, int, Any], 
        high_key: Optional[Tuple[Union[Reader, int],int]], 
    ) -> None: 
        # get simplified_key 
        low_key ,T = self.output._type._slice(high_key) 
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

    # def _sv_assignment(self, assignment: Tuple, builder: sv.ModuleSV, op='=') -> str:
    #     """ conditioanl dynamic partial assignment  
    #     assignment:
    #         (cond, key, value)
    #     return: 
    #         if (cond) begin
    #             x[idx +: 2] = y;
    #         end
    #     """
    #     cond, key, value = assignment  
    #     target = self.output._data._dump_sv_assign(key, builder=builder) 
    #     if cond is None:
    #         return f'{target} {op} {builder.get_name(value)};'  
    #     else:
    #         return f'if({builder.get_name(cond)}) {target}{op} {builder.get_name(value)};'

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
            self.output._data._setval_py(self.branches, need_merge=True, dt=self.delay, trace=self)
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
            reset: Optional[Tuple[Reader, int]] = ...,
            reset_value = None,
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
        if reset_value is None:
            self.reset_value: Logic = x._data._getval_py()
        else:
            self.reset_value: Logic = x._type._eval(reset_value)
        
        if clock is ...:   clock = config.conf.clock 
        if reset is ...:   reset = config.conf.reset

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
            # XXX self connect not necessary
            self.branches.append((None, None, x)) 
            # self.condtree.insert([], None, x)
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
            self.output._data._setval_py(self.branches, need_merge=True, dt=self.delay, trace=self)

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
        
        body = self.condtree._dump_sv(self.output, builder=builder, op='<=')
        body = '    ' + '    '.join(body.splitlines(keepends=True))
        if has_reset:
            left, right = self.output._data._dump_sv_assign(None, self.reset_value, builder=builder)
            body = f'  if ({reset}) {left} <= {right}; else begin\n{body}\n  end'

        ret = '\n'.join([
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
    
    def __head__(
            self, 
            x: Reader, 
            *, 
            id: str = '', 
            enable: Tuple[Reader, int] = ...,
            reset: Optional[Tuple[Reader, int]] = ...
        ) -> Reader:
        self.id = id or self.id
        self.branches: List[Tuple[
            Union[Reader, None], 
            Union[None,Tuple],
            Union[Reader, Logic]
        ]] = []   
        self.condtree = CondTreeNode()
        # enable signal, level sensitive 
        self.enable = self.read(enable[0])
        self.enable_level = int(enable[1]) 
        assert len(self.enable) == 1, 'not 1-bit signal'
        assert self.enable_level in [0,1]

        self.pos_rst: Reader = None
        self.neg_rst: Reader = None
        self.reset_value: Logic = x._data._getval_py()
        # reset signal
        if reset is ...:   reset = config.conf.reset
        if reset is None:
            pass 
        elif reset[1]:
            self.pos_rst = self.read(reset[0])
        else:
            self.neg_rst = self.read(reset[0])

        # has default output
        assert x._data.writer is None, f'Signal {x} already has driver'  
        self.output = self.write(x)  
        self.branches.append((None, None, x._data._getval_py())) 
        self.condtree.insert([], None, x._data._getval_py())
        return x 
        

    def forward(self): 
        if (rst:=self.pos_rst) is not None: 
            if rst._data._getval_py() == Logic(1,0):
                self.output._data._setval_py(self.reset_value, dt = self.delay, trace=self) 
                return
        if (rst:=self.neg_rst) is not None:
            if rst._data._getval_py() == Logic(0,0):
                self.output._data._setval_py(self.reset_value, dt = self.delay, trace=self) 
                return 
        
        if self.enable._data._getval_py() == self.enable_level:
            if len(self.branches) == 1:
                data = self.branches[-1][-1]  
                if isinstance(data, Reader):
                    data = data._data._getval_py()
                self.output._data._setval_py(data, dt = self.delay, trace=self)
            else:
                self.output._data._setval_py(self.branches, need_merge=True, dt=self.delay, trace=self)

                
    def dump_sv(self, builder: sv.ModuleSV):
        """  
        always_latch begin 
            if (reset) out = init;
            else if (enable) begin
                latch = a;
                latch = b;
                ...
            end
        end
        """
        enable = builder.get_name(self.enable) 
        if self.enable_level == 0: 
            enable = f'!{enable}' 
        has_reset = False
        if (rst:=self.pos_rst) is not None: 
            reset = builder.get_name(rst)
            has_reset = True 
        elif (rst:=self.neg_rst) is not None:
            reset = f'!{builder.get_name(rst)}'
            has_reset = True   
        body = self.condtree._dump_sv(self.output, builder=builder, op='=')
        body = '    ' + '    '.join(body.splitlines(keepends=True))
        if has_reset:
            left, right = self.output._data._dump_sv_assign(None, self.reset_value, builder=builder)
            body = f'  if ({reset}) {left} = {right}; else if ({enable}) begin\n{body}\n  end'
        else:
            body = f'  if ({enable}) begin\n{body}\n  end' 

        ret = '\n'.join(['always_latch begin', body , 'end'])
        builder.Block(self, ret)


    def __str__(self):
        return f'Latch(name={self.output._data._name})'



            


class Analog(Assignable): 
    pass



class _WireTri(Analog):  
    
    id = 'Wtri'         # id 
    _netlist = 'wire'
  
    def __head__(self, x: Reader, *, id: str) -> Reader: 

        self.id = id or self.id  
        assert x._data.writer is None and x._type._storage == 'packed' 
        self.branches: List[Tuple[
            Reader,                 # condition signal, 1-bit 
            None,                   # no key is allowed
            Union[Reader, Logic]    # value
        ]] = []   
        self.output = self.write(x) 
        self.first_branch = (None, None, Logic(0, gmpy2.bit_mask(len(x))))
        return x 
        
    def __merge__(self, other: Assignable): 
        assert isinstance(other, _WireTri) and len(other.output._type) == len(self.output._type)
        # merge branches
        for cond, key, value in other.branches:
            self.read(cond)
            if isinstance(value, Reader):
                self.read(value)
            self.branches.append((cond, key, value))
        for reader in list(other.output._data.reader.keys()):
            reader._exchange(self.output._data)
        other._delete()

    def _delete(self): 
        for i in self.iports:
            i._driven.pop(self)
        # delete output
        for o in self.oports: 
            o._delete() 
        self.iports.clear()
        self.oports.clear()
        self._sess._remove_gate(self)


    def __partial_assign__(
        self, 
        acitve_signals: List[Reader],
        cond_stacks: List[hgl_assign.CaseGate],        
        value: Union[Reader, int, Any], 
        high_key: Optional[Tuple],
    ) -> None: 
        """ 
        TODO casted type bit-width mismatch
        """  
        assert len(cond_stacks) == 1, 'inout assign only valid in first `when` stmt'
        cond = cond_stacks[0].inout_valid 
        assert isinstance(cond, Reader), 'inout assign only valid in first `when` stmt'
        self.read(cond) 
        assert high_key is None, 'inout does not support partial/sliced assignment'
        if isinstance(value, Reader):
            self.read(value)
        else:
            value = self.output._type._eval(value)
        self.branches.append((cond, None, value))
        

    def forward(self):
        """ simulation function 
        """
        l = [self.first_branch, *self.branches]
        self.output._data._setval_py(l, need_merge=True, dt=self.delay, trace=self)

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        """
        assign out = cond1 ? a : 'z;
        assign out = cond2 ? b : 'z;
        assign out = cond3 ? c : 'z;
        """
        # body = [self._sv_assignment(i, builder=builder) for i in self.branches] 
        # body.insert(0, 'always_comb begin')
        # body.append('end')
        # builder.gates[self] = '\n'.join(body) 
        ret = []
        out = builder.get_name(self.output)
        for cond, _, value in self.branches:
            ret.append(f"assign {out} = {builder.get_name(cond)} ? {builder.get_name(value)} : 'z;")
        ret = '\n'.join(ret)
        builder.Block(self, ret)
            
    def __str__(self):
        return f'Tri(name={self.output._data._name})'


@dispatch('Wire', Any)
def _wire(x: Reader, *, id: str = '') -> Reader:
    return _Wire(x, id=id, next=False) 


@dispatch('WireNext', Any)
def _wirenext(x: Reader, *, id: str = '') -> Reader:
    return _Wire(x, id=id, next=True) 


@dispatch('Reg', Any)
def _reg(x: Reader, *, id = '', clock = ..., reset = ..., reset_value = None) -> Reader: 
    return _Reg(x, id=id, next=False, clock=clock, reset=reset, reset_value = reset_value)


@dispatch('RegNext', Any)
def _regnext(x: Reader,  *, id = '', clock = ..., reset = ..., reset_value = None) -> Reader: 
    return _Reg(x, id=id, next=True, clock=clock, reset=reset, reset_value = reset_value)


@dispatch('Latch', Any)
def _latch(x: Reader, *, id: str = '', enable=...) -> Reader:
    return _Latch(x, enable = enable, id=id)

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
        return ret


    def sim_init(self):
        super().sim_init() 
        self.sim_x_count += 1000

    def sim_vx(self):
        low = self.timing['low']
        high = self.timing['high']
        phase = self.timing['phase'] 
        clk = self.clk._data
        simulator = self._sess.sim_py  
        yield 
        simulator.update_v(low+phase-1, clk, gmpy2.mpz(1))
        while 1:
            yield 
            if clk.v:
                simulator.update_v(high, clk, gmpy2.mpz(0))
            else:
                 simulator.update_v(low, clk, gmpy2.mpz(1))
    
    def sim_v(self): 
        while 1:
            yield


    def dump_cpp(self):
        raise NotImplementedError()
    
    def dump_sv(self, builder: sv.ModuleSV) -> None:
        clk = builder.get_name(self.clk)
        low = self.timing['low']
        high = self.timing['high']
        builder.Task(self, f'always begin {clk} = 0; # {low}; {clk} = 1; # {high}; end')
    



class ClockDomain(HGL):
    
    _sess: session.Session
    
    def __init__(self, clock = ..., reset = ...):
        if clock is ...:
            clock = config.conf.clock 
        if reset is ...:
            reset = config.conf.reset 
            
        clk, edge = clock 
        assert isinstance(clk, Reader) and isinstance(edge, (int, bool))
        self.clock: Tuple[Reader, int] = clock
        assert reset is None or isinstance(reset[0], Reader)
        self.reset: Optional[Tuple[Reader, int]] = reset  
        
        self._clk_restore = []
        self._rst_restore = []
        
    def __enter__(self):
        self._clk_restore.append(config.conf.clock)
        self._rst_restore.append(config.conf.reset)
        config.conf.clock = self.clock 
        config.conf.reset = self.reset

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        config.conf.reset = self._clk_restore.pop()
        config.conf.clock = self._rst_restore.pop()
    
    
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


random.seed(42)

@vectorize
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
        left._data._setval_py([(None, low_key, immd)], need_merge=True, dt=dt, trace=trace)
    
    
    
@vectorize
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

@vectorize
def getr(left: Reader) -> Logic: 
    """ not set, just return 
    """
    T = left._type 
    v = T._eval(random.randrange(1 << len(T))) 
    return v 

@vectorize
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





import pyhgl.logic.hgl_assign as hgl_assign
