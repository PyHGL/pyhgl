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
import pyhgl.logic.hglmodule as hglmodule
import pyhgl.logic._session as _session  
import pyhgl.logic.verilogmodule as verilogmodule
import pyhgl.logic.utils as utils
import pyhgl.logic.assign as assign 



#--------------- 
# inner datatype 
#--------------- 
    
class _Data(HGL):
    __slots__ = () 


class BitPat(_Data):
    """ zero extended to inf. ex. ??? -> 0...0???
    """
    def __init__(self, v: str) -> None:
        """ 0101??11  -> mask: 111....11110011, value: 01010011
        """
        self._v: str = utils.str2bitpat(v)  # 0 is stripped
        mask = []
        value =[]
        for i in v:
            if i == '0':
                mask.append('1')
                value.append('0')
            elif i == '1':
                mask.append('1') 
                value.append('1')
            elif i == '?':
                mask.append('0') 
                value.append('0') 
        self.mask = gmpy2.mpz(int(''.join(mask), base=2)) | (~ gmpy2.bit_mask(len(v)))
        self.value = gmpy2.mpz(int(''.join(value), base=2))
    
    def __str__(self):
        return f'BitPat({self._v})'  

    def __len__(self):
        return len(self._v)

    def __eq__(self, x: Union[int, gmpy2.mpz]) -> bool:
        return (x & self.mask) == self.value

        

class Logic(_Data):
    """ arbitrary length of 3-state value: 0, 1, x 

    v   x    state 
    0   0     0 
    1   0     1 
    0   1     x 
    1   1     z
    """

    __slots__ = 'v', 'x'

    def __init__(self, v: Union[str, int, gmpy2.mpz, Logic] = 0) -> None: 
        if isinstance(v, (int, gmpy2.mpz, gmpy2.xmpz)):
            assert v >= 0
            self.v = gmpy2.xmpz(0)  
            self.x = gmpy2.xmpz(0)
            self.v[:] = v 
        elif isinstance(v, Logic):
            self.v = gmpy2.xmpz(0)  
            self.x = gmpy2.xmpz(0) 
            self.v[:] = v.v 
            self.x[:] = v.x
        elif isinstance(v, str):
            self.v, self.x, _width = utils.str2logic(v) 
        else:
            raise ValueError(v)
            
    def __len__(self):
        return max(utils.width_infer(self.v), utils.width_infer(self.x))

    def __eq__(self, other: Logic) -> bool:
        if self.x or other.x:
            return False 
        return self.v == other.v  

    def __and__(self, other: Logic) -> Logic:
        ...

    def __getitem__(self):
        ... 

    def __setitem__(self):
        ...

    def __ne__(self, other):
        ...
            
    def __str__(self): 
        ret = []
        for i in range(len(self)):
            if self.x[i]:
                ret.append('x')
            else:
                ret.append(str(self.v[i]))
        return ''.join(reversed(ret))
        
#--------------------------------------
# data container, updated in simulation
#--------------------------------------

class SignalData(HGL):
    """ data container 
    
    Attribute:
        - _t: time of latest update
        - writer: None | Writer 
            - Writer: out-edge of Gate
            - zero or one writer
        - reader: dict of Reader
            - Reader: in-edge of Gate 
            - arbitrary number of readers
    Method:
        - __init__: instantiation from given Immd
        - copy: return a copy with current data, but without writer/reader
        - __eq__: whether current value is equal to Immd
        - _name: return a prefered name 
    """
    _sess: _session.Session 
    __slots__ = 'writer', 'reader', '_t'
    
    def __init__(self) -> None:
        # time stamp of the latest update
        self._t: int = 0         
        # 0 or 1 writer
        self.writer: Optional[Writer] = None 
        # any number of reader
        self.reader: Dict[Reader, None] = {}
    
    def copy(self) -> SignalData:
        """ return an instance with same value, do not copy writer and reader
        """
        raise NotImplementedError() 


    def __hash__(self):             
        return id(self) 
    
    @property
    def _name(self) -> str:
        # prefered name
        if self.reader:
            return next(iter(self.reader.keys()))._name
        else:
            return 'temp'
        
    def __str__(self):
        """ return current value
        """
        raise NotImplementedError() 
    
    

class LogicData(SignalData):
    """ 2-valued signal value
    """
    
    __slots__ = 'v'
    
    def __init__(self, v: Union[int, gmpy2.mpz, gmpy2.xmpz] = 0) -> None:
        super().__init__()
        self.v: gmpy2.xmpz = gmpy2.xmpz(gmpy2.mpz(v))   
        
    def copy(self) -> LogicData:
        return LogicData(self.v)
    
    def __str__(self):
        ret = bin(self.v)[1:]
        if len(ret) > 35:
            return f'b...{ret[-32:]}'
        else:
            return ret


class MemData(SignalData):
    """ unpacked array
    """

    __slots__ = 'v','shape'

    def __init__(self, v: Union[int, gmpy2.mpz, gmpy2.xmpz], shape: Tuple[int,...]) -> None:
        """ shape: (1024, 8)
        """
        super().__init__()
        self.v: gmpy2.xmpz = gmpy2.xmpz(gmpy2.mpz(v))   
        assert len(shape) >= 2
        self.shape: Tuple[int,...] = shape 

    def copy(self) -> MemData:
        return MemData(self.v, self.shape)
         
    def __str__(self):
        ret = hex(self.v)[2:] 
        if len(ret) > 35:
            return f'mem{self.shape}{ret[-32:]}'
        else:
            return f'mem{self.shape}{ret}'
    
#----------------------------------
# Signal: out/in edge of SignalData
#----------------------------------
    
         
class Writer(HGL):
    """ a writer is output of specific gate 
    """

    __slots__ = '_sess', '_data', '_type', '_driver', '_timewheel'
    
    def __init__(self, data: SignalData, type: SignalType, driver: Gate): 
        self._sess: _session.Session = HGL._sess
        self._data: SignalData = data; 
        self._data.writer = self 
        self._type: SignalType = type
        self._driver: Gate = driver 
        # speed up 
        self._timewheel = self._sess.simulator.time_wheel
        
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
        self._sess = None
        self._type = None
        
    def _setval(self, v: Any, dt: int, immd_key = None) -> None:
        """ insert a update event to simulator
        
        simulation stage, called by Gate.forward 
        
        v: Immd, gmpy2.mpz | _Data | str ...
        dt: delay 
        low_key: None | (start, length) | ...
        """
        e = (self._type, self._data, immd_key, v)
        # insert to timewheel
        try:
            self._timewheel[dt][0].append(e) 
        # insert to priority queue
        except:
            sim = self._sess.simulator
            t = sim.t + dt 
            if t not in sim.priority_queue:
                sim.priority_queue[t] = self._sess.simulator.new_events() 
            sim.priority_queue[t][0].append(e)
        
    def __len__(self):
        return len(self._type)
    


class Reader(HGL):
    """ read-only signal 
    
    _data: 
        instance of SignalData 
        
    _type:
        instance of SignalType
        
    __getattr__:
        get attr of _type. if method, bound to self 
    """
    
    __slots__ = ('_sess', '_data', '_type', '_direction', '_driven', 
                 '_events', '_name', '_timestamps', '_values')
    
    def __init__(self, data: SignalData, type: SignalType, name: str = ''):
        self._sess: _session.Session = HGL._sess
        self._data: SignalData = data; self._data.reader[self] = None
        self._type: SignalType = type 
        self._direction: Literal['inner', 'input', 'output', 'inout'] = 'inner'
        # prefered name
        self._name: str = name or type._name

        # simulation
        # driven gates
        self._driven: Dict[Gate, int] = {}
        # driven events, coroutine_events that triggered when signal changing 
        self._events: List[Generator] = []
        # store history values for verification and waveform
        self._timestamps: list[int] = []
        self._values: list[Any] = []
        
    @property 
    def __hgl_type__(self) -> type:
        """ type for dynamic dispatch
        """
        return type(self._type) 
        
    def _exchange(self, data: SignalData) -> SignalData:
        """ read from another data
        """
        self._data.reader.pop(self) 
        ret, self._data = self._data, data 
        self._data.reader[self] = None
        return ret  

    def _disconnect(self): 
        """ read from new data, return a new signal read from odd data
        """
        odd_data = self._exchange(self._data.copy())
        return Reader(odd_data, self._type, name=self._name)

    def copy(self): 
        """ return a new signal with same type, connect to nothing
        """
        return Reader(self._data.copy(), self._type, name=self._name)
    
    def _track(self) -> None:
        """ track signal waveform
        """
        if not self._timestamps:
            self._timestamps.append(self._sess.simulator.t+1)
            self._values.append(self._getval()) 
        
    def _update(self) -> None:
        """ called after _track, store current time and value
        """
        v = self._getval()
        if self._values[-1] == v:
            return 
        else:
            self._timestamps.append(self._sess.simulator.t+1)
            self._values.append(v) 
        
    def _history(self, t:int) -> Any:
        """ called after _track, get value at time t
        """
        idx = bisect.bisect(self._timestamps, t) - 1
        if idx < 0: 
            idx = 0 
        return self._values[idx]
        
    def _getval(self, immd_key = None):
        """ simulation stage, get signal value immediately
        """
        return self._type._getimmd(self._data, immd_key)
    
    def _setval(self, v: Any, dt, immd_key = None) -> None:
        """ simulation stage, insert an event 
        """
        self._sess.simulator.insert_signal_event(dt, (self._type, self._data, immd_key, v))
        
    def __getitem__(self, keys) -> Any:  
        """ building stage, dynamic partial slicing 
        """ 
        if self._data.writer is None or not isinstance(self._data.writer._driver, Assignable):
            return _part_select_wire(self, keys) 
        else:
            return self._data.writer._driver.__part_select__(self, keys)
    
    def __partial_assign__(
        self, 
        conds: List[Union[assign.WhenElseFrame, assign.SwitchOnceFrame]],
        value: Union[int, Reader, list, dict, str, Any], 
        keys: Union[SignalKey, int, Reader, tuple, Any], 
    ) -> None:
        """ building stage. recursively call of partial assign
        
        ex. a[key1, key2, key3,...] <== value 

        conds: 
            condition frames 
        value: 
            signal or immd 
        keys:
            high level key: SignalKey, signal or immd
        """
        # make self assignable 
        if self._data.writer is None or not isinstance(self._data.writer._driver, Assignable):
            Wire(self)  
        self._data.writer._driver.__partial_assign__(self, conds, value, keys)  

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



@dispatch('Signal', Any)
def _signal(obj) -> Reader:
    """ python literal -> Reader
    """
    if isinstance(obj, Reader):
        return obj 
    elif isinstance(obj, str):
        return UInt(obj)
    elif isinstance(obj, SignalType):
        return obj()
    else: 
        return UInt(gmpy2.mpz(obj))

    

#----------------------------------
# Signal Type
#----------------------------------

class SignalType(HGL):
    
    """ signal type, immutable 
    
    - len() return signal bit width. 0 means undetermined width
    - _eval accept python data type such as int, str, list, return Immd
        - Immd: gmpy2.mpz | _Data   
        - overflow is not allowed
    - __call__ create and return a signal. 
        - positional args should have default value
        - first positional arg can be Immd, Signal, SignalData
    - _slice accept one or more keys, return (start, SignalType)
        - start: int | signal
    - _getimmd: return current value of data
    - _setimmd: update current data
        - Args
            - v: Immd
                - Immd: gmpy2.mpz | _Data
            - key: None | (start, length)
                - start, length: int
        - Return: whether value is changed or not 
    """
    __slots__ = ()
    _sess: _session.Session 
    _name: str = 'signal'
    
    def __len__(self) -> int:
        raise NotImplementedError() 
    
    def _eval(self, v: Any) -> Any:
        """ building stage 
        
        v: int, str, list, dict, bitpat 
        return Immd accepted by SignalData.__init__, or Immd for `==` op
        
        overflow will raise exception
        """
        raise NotImplementedError()
    
    def _slice(self, keys) -> Tuple[Optional[tuple], SignalType]: 
        """ building stage. turn high-level keys into low-level key
        
        used in both part-select and partial-assignment 

        low_key: 
            Tuple[Signal|immd]
        immd_key: 
            Tuple[immd]

        return (low_key, SignalType)
        """
        raise NotImplementedError()  

    
    def _getimmd(self, data: SignalData, immd_key: Optional[Tuple[int, int]]) -> Any:
        """ simulation stage, return current value (immutable)
        
        - data: SignalData
        - key: None | (start, length)
            - start: signal|int 
            - length: int
        """
        raise NotImplementedError()  

    
    def _setimmd(self, data: SignalData, immd_key: Optional[Tuple[int, int]], v: Any) -> bool: 
        """ simulation stage, set some bits, reutrn True if value changes
        
        data: SignalData
        key: None | (start, length), may out of range
        v: immd, may out of range
        """
        raise NotImplementedError()

    def _verilog_name(self, x: Union[Reader, SignalData], m: verilogmodule.VerilogModule) -> str: 
        """
        x:
            SignalData: LHS, return a new defination name
            Reader: RHS, if casted type, return a new signal name  
        """
        if len(self) == 1:
            verilog_width = ''
        else:
            verilog_width = f'[{len(self)-1}:0]'

        if isinstance(x, SignalData):
            ret = m.new_name(x, x._name)  
            m.new_signal(x, f'{x.writer._driver._netlist} {verilog_width} {ret}') 
            return ret
        elif isinstance(x, Reader):  
            data = x._data 
            assert isinstance(data, LogicData)
            origin = m.get_name(data) 
            assert origin
            # same bit width
            if len(data.writer._type) == len(self): 
                m.update_name(x, origin)
                return origin
            else:
                ret = m.new_name(x, x._name) 
                m.new_signal(x, f'{data.writer._driver._netlist} {verilog_width} {ret}') 
                m.new_gate(f'assign {ret} = {origin};') 
                return ret
        else:
            raise TypeError(x)

    def _verilog_immd(self, x: Union[int, gmpy2.mpz, BitPat]) -> str: 
        """ python immd to verilog immd 
        """
        width = len(self)
        if isinstance(x, (int, gmpy2.mpz, gmpy2.xmpz)):
            return f"{width}'d{x}"
        elif isinstance(x, BitPat):
            v ='0' * (width-len(x)) + x._v
            return f"{width}'b{v}" 

    def _verilog_key(self, low_key: Optional[tuple], gate: Gate) -> str:
        if low_key is None:
            return ''
        else:
            if isinstance(low_key[0], Reader):
                start, length = gate.get_name(low_key[0]), str(low_key[1])
            else:
                start, length = str(low_key[0]), str(low_key[1])
            return f'[{start} +: {length}]'

    
    def __call__(self) -> Reader:
        """ create a signal with/without arguments
        """
        raise NotImplementedError()  
        
    def __str__(self, data: SignalData=None):
        raise NotImplementedError() 
    
        
        
class LogicType(SignalType):
    """ UInt, SInt, Vector, Struct, ...
    """
    __slots__ = '_width' 

    def __init__(self, width: int = 0):
        self._width: int = width  
        
    def __len__(self) -> int:
        return self._width
        
    def __str__(self, data: LogicData = None):
        T = f"{self.__class__.__name__}[{len(self)}]" 
        if data is None:
            return T 
        else:
            return f"{T}({data})"
    
    def _getimmd(self, data: LogicData, immd_key: Optional[Tuple[int, int]]) -> Any:
        """ get integer value. out of range bits are 0 
        
        return: gmpy2.mpz 
        """
        if immd_key is None: 
            return data.v[0:self._width] 
        else:
            return data.v[immd_key[0]:min(immd_key[0]+immd_key[1], self._width)] 

        
    def _setimmd(self, data: LogicData, immd_key: Optional[Tuple[int, int]], v: gmpy2.mpz) -> bool: 
        """ set some bits, reutrn True if value changes
        
        v: gmpy2.mpz, may overflow
        key: None | (start, length), may out of range
        
        XXX: slicing negative gmpy2.mpz return wrong value
        XXX: slicing key for gmpy2.xmpz cannot be mpz
        """ 
        target: gmpy2.xmpz = data.v 
        if immd_key is None:
            odd = target.copy()
            target[0:self._width] = v  
            return target != odd

        w = self._width
        if v < 0:
            v = v & gmpy2.bit_mask(w) 
        start = int(immd_key[0])
        stop = min(start + immd_key[1], w)
        if start >= stop:
            return False 
        if target[start:stop] == v[0:stop-start]:
            return False  
        target[start:stop] = v
        return True  

    
    def _slice(self, keys) -> Tuple[Tuple[Union[int, Reader], int], SignalType]:
        """ uint slicing
        
        keys: high-level keys, int, signal, slice 
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
        if self._width == 0:
            raise TypeError(f'{self} is not complete signal type')
        
        if isinstance(keys, slice):
            start, stop, step = keys.start, keys.stop, keys.step 
            if start is not None and stop is None and step is not None:
                # [signal::8]
                return (start, step), UInt[step] 
            elif step is not None:
                # [:-1]
                raise KeyError(f'invalid slicing {keys}')
        elif isinstance(keys, int):
            if keys < 0:
                keys += self._width
            start, stop = keys, keys+1 
        elif isinstance(keys, Reader):
            return (keys, 1), UInt[1]
        else:
            raise KeyError(keys) 
        
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

    def extz(self: Reader, w: int):
        return Extz(self, w)        

    def exto(self: Reader, w: int):
        return Exto(self, w)   
    


class UIntType(LogicType):

    _cache = {}
    _name = 'uint'

    def __init__(self, width: int = 0): 
        assert width > 0
        super().__init__(width)

    def _eval(self, v: Union[int, str, float]) -> Union[gmpy2.mpz, BitPat]:
        """ 
        called when:
            - signal <== immd 
            - signal == immd 
            - UInt(immd) 
            
        Exception if v overflowed
        """
        if isinstance(v, str) and '?' in v:
            v = BitPat(v)  
        if isinstance(v, BitPat):
            assert len(v) <= self._width, 'overflow'
            return v 

        if isinstance(v, str): 
            _v, _w =  utils.str2int(v) 
        else:
            _v = v
            _w = utils.width_infer(_v) 
        # overflow not allowed
        if self._width < _w or _v < 0:
            raise Exception(f'value {v} overflow for {self}') 
        if isinstance(_v, gmpy2.mpz):
            return _v
        else:
            return gmpy2.mpz(_v)

    def __call__(
        self, 
        v: Union[int, str, float, Reader, Iterable]=0, 
        *, 
        name: str = 'uint'
    ) -> Reader: 
        """
        v:
            - int, str, float 
            - signal 
            - Iterable container of above
        """ 
        # array
        v = ToArray(v)
        if isinstance(v, Array):
            return Map(self, v, name=name)
        # is type cast
        if isinstance(v, Reader):    
            data = v._data  
            if not isinstance(data, LogicData):
                raise ValueError(f'signal {v} is not logic signal')
            return Reader(data=data, type=self, name=name) 
        else:
            _v = self._eval(v) 
            return Reader(data=LogicData(_v), type=self, name=name) 
    

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
        v: Union[int, str, float, Reader, Iterable]=0, 
        w: int = None,
        name: str = 'uint'
    ) -> Reader: 

        # array
        v = ToArray(v) 
        w = ToArray(w)
        if isinstance(v, Array) or isinstance(w, Array):
            return Map(self, v, w, name=name)

        if w is not None:
            return UInt[w](v, name=name)
        # is type cast
        if isinstance(v, Reader):    
            return UInt[len(v)](v, name=name)  
        else:
            if isinstance(v, str):
                _, _w = utils.str2int(v) 
            else:
                _w = utils.width_infer(v)
            return UInt[_w](v, name=name)


def Split(signal: Reader) -> Array:
    assert len(signal) > 0
    return Array(signal[i] for i in range(len(signal)))


def Zext(signal: Reader, w: int) -> Reader:
    _w = len(signal)
    assert _w > 0
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(signal, UInt[w-_w](0))


def Oext(signal: Reader, w: int) -> Reader:
    _w = len(signal)
    assert _w > 0
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(signal, UInt[w-_w]((1<<(w-_w))-1)) 


def Sext(signal: Reader, w: int) -> Reader:
    _w = len(signal)
    assert _w > 0
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        msb = signal[-1] 
        return Cat([signal] + [msb] * (w-_w))


def Extz(signal: Reader, w: int) -> Reader:
    _w = len(signal)
    assert _w > 0
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(UInt[w-_w](0), signal)
 

def Exto(signal: Reader, w: int) -> Reader:
    _w = len(signal)
    assert _w > 0
    if _w > w:
        raise Exception(f'overflow') 
    elif _w == w:
        return Wire(signal) 
    else:
        return Cat(UInt[w-_w]((1<<(w-_w))-1), signal)




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
    
    
    def _eval(self, v: Union[int, str, Iterable]) -> gmpy2.mpz:
        """ if not iterable, set UInt(v) as whole; if iterable, set each value 
        """
        v = ToArray(v) 
        
        if isinstance(v, Array): 
            assert len(v) == self._length, 'shape mismatch' 
            _v = gmpy2.mpz(0)
            _w = len(self._T)
            for i, x in enumerate(v):
                _v |= (self._T._eval(x)) << (i*_w)
            return _v
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
            return Reader(data=v, type=self, name=name)
        else:
            _v = self._eval(v)
            return Reader(data=LogicData(_v), type=self, name=name)
    
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

     
    def _eval(self, v: Union[int, str, Array, dict]) -> gmpy2.mpz:
                
        if not isinstance(v, (Array, dict)):
            return UInt[len(self)]._eval(v)
        else:
            v = Array(v)
            ret = gmpy2.mpz(0)
            for key, value in v._items():
                T, start = self._fields[key]
                _v = T._eval(value) 
                _w = len(T)
                ret &= ~(((1 << _w) - 1) << start)
                ret |= _v << start
                
            return ret 
    
    def __call__(
        self, 
        v: Union[LogicData, Reader, Iterable]=0, 
        *, 
        name: str = 'vector'
    ) -> Reader:
        
        if isinstance(v, Reader):
            v = v._data  
            assert isinstance(v, LogicData)  
            return Reader(data=v, type=self, name=name)
        else:
            _v = self._eval(v)
            _w = len(self)
            return Reader(data=LogicData(_v), type=self, name=name)
        
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
        

    def __len__(self) -> int:
        return self._width 

    def _getimmd(self, data: MemData, immd_key: Optional[Tuple[int]]) -> Any:
        """ get integer value. out of range bits are 0 
        
        return: gmpy2.mpz 
        """
        if immd_key is None: 
            return data.v[0:self._width] 
        else:
            start = 0
            for x, w in zip(immd_key, self._idxes):
                start += x * w  
            return data.v[start:start+self._shape[-1]]

    def _setimmd(self, data: MemData, immd_key: Optional[Tuple[int]], v: gmpy2.mpz) -> bool: 
        """ set some bits, reutrn True if value changes 

        immd_key: maybe mpz, turn into int
        """ 
        target: gmpy2.xmpz = data.v 
        if immd_key is None:
            odd = target.copy()
            target[0:self._width] = v  
            return target != odd

        v = v & gmpy2.bit_mask(self._shape[-1]) 
        start = 0
        for idx, max_idx, w in zip(immd_key, self._shape, self._idxes):
            if idx >= max_idx:
                return False  
            start += idx * w  
        stop = start + self._shape[-1] 

        start = int(start)
        stop = int(stop) 
        print(immd_key, v, self._sess.simulator.t)
        if target[start:stop] == v:
            return False 
        else:
            target[start:stop] = v 
            return True


    
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

    
    def _eval(self, v: Union[int, str, Iterable]) -> gmpy2.mpz:
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

    def _verilog_name(self, x: Union[Reader, SignalData], m: verilogmodule.VerilogModule) -> str: 
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

    id: str = 'Gate'
    timing: Dict[str,int] = {'delay': 1} 
    delay: int

    iports: Dict[Reader, None] 
    oports: Dict[Writer, None]
    
    _netlist: str = 'logic'   # verilog netlist type of output signal
    _sess: _session.Session 
    
    __slots__ = 'iports', 'oports', 'delay', '_sess', '__dict__'
    
    def __head__(self, *args, **kwargs) -> Union[Reader, Tuple[Reader]]:
        """ initializating,  return Signal or Tuple[Signal]
        """
        pass
        
    def read(self, x: Reader) -> Reader:
        """ add x as input signal
        """ 
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
    
    def get_name(self, obj: Union[Reader, Writer]) -> str:
        """ get object name in verilog module
        """
        m: hglmodule.Module = self._sess.verilog.gates[self] 
        return m._verilog_name(obj)
        

    def get_timing(self):
        timing = self._sess._get_timing(self.id) or self.timing 
        self.delay: int = timing['delay']
        
    def __new__(cls, *args, **kwargs):
        
        self = object.__new__(cls) 
        self._sess: _session.Session = HGL._sess
        self._sess._new_gate(self) 
        self.iports = {}
        self.oports = {}
        ret = self.__head__(*args, **kwargs)
        # get simulation parameters
        self.get_timing()
        
        # ---------------------------------
        if self._sess.verbose_hardware:
            trace = utils.format_hgl_stack(2, 4)
            self._sess.print(f'{self}, {self.id}={self.timing}')
            self._sess.print(trace)
        #----------------------------------
        
        return ret 
    
    
    def forward(self) -> None:
        """ called by simulator when inputs changing or after init 
        """
        pass
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str:
        """ called when emitting verilog 
        
        ex. assign a = b | c;
        """
        return ''
        
    def _emit_graphviz(self, g, iports: List[Reader], oports: List[Writer], body = 'Node'):
        newline = '&#92;n'
        iports_str = '|'.join(f"<i{id(i)}> {i._name}{newline}{i._type.__str__(i._data)}" for i in iports)
        oports_str = '|'.join(f"<o{id(o._data)}> {o._data._name}{newline}{o._type.__str__(o._data)}" for o in oports)
        label = "{{%s} | %s | {%s}}" % ( iports_str, body, oports_str)
        
        curr_gate = str(id(self))  
        g.node(name = curr_gate, label = label, shape='record', color='blue') 
        
        for signal in iports:
            if signal._data.writer is not None:
                source_gate = str(id(signal._data.writer._driver))
                g.edge(f"{source_gate}:o{id(signal._data)}", f"{curr_gate}:i{id(signal)}")
        
    def emitGraph(self, g) -> None:
        self._emit_graphviz(g, self.iports, self.oports, self.id) 

    def __str__(self):
        return f"{self.__class__.__name__}"




#----------------------------------
# Assignable: Wire & Reg
#----------------------------------

class _Assignment(HGL):
    
    
    __slots__ = 'target', 'key', 'value', 'value_key'
    
    def __init__(
        self, 
        key: Optional[tuple], 
        value: Union[Reader, Any],
        value_key: Optional[tuple],
    ):
        """ target[key] <== value[value_key]
        """
        self.key = key
        self.value = value 
        self.value_key = value_key 
        
        if value_key is not None:
            assert isinstance(value, Reader), f'xxx[{key}] <== {value}[{value_key}]'
    
    def eval(self) -> Tuple[Optional[Any],Any]: 
        """ return immd_key and immd of value[value_key] 

        low_key -> immd_key
        """  
        if self.key is not None:
            key = [i._getval() if isinstance(i, Reader) else i for i in self.key]
        else:
            key = None

        value, value_key = self.value, self.value_key 
        if value_key is not None: 
            value_key = [i._getval() if isinstance(i, Reader) else i for i in value_key] 
            return key, value._getval(value_key)
        else:
            if isinstance(value, Reader):
                value = value._getval()
            return key, value      

    def __str__(self):
        return f'(key={self.key}, value={self.value})'
        

class CondTreeNode:
    """ store conditions as a tree 
    
    - frames: list of condition_frame | _Assignment
        - condition_frame: list of (condition, CondTreeNode)
        - condition: None | tuple(signal, immd, ...)
            - immd: gmpy2.mpz, BitPat 
    - _Assignment: (target, target_key, value, value_key)
        - target_key: low_level key, maybe signal
        - value: immd | signal   
        - value_key: out[key] <== value[value_key] 
        
    level of tree is level of condition frame
    
    """ 
    
    __slots__ = 'gate', 'is_analog', 'is_reg', 'curr_frame', 'frames'
    
    def __init__(
        self, 
        gate: Assignable, 
        is_analog = False,
        is_reg = False
    ):
        
        self.gate = gate            # target gate that condition tree belongs to
        self.is_analog = is_analog  # if True, store all assignments 
        self.is_reg = is_reg        # if True, do not trigger on inputs  
        
        # whether use existing frame or append a new frame
        self.curr_frame: Union[assign.WhenElseFrame, assign.SwitchOnceFrame] = None 

        self.frames: List[Union[_Assignment, List[Tuple[Optional[tuple], CondTreeNode]]]] = [] 
    
    def insert(
        self, 
        conds: List[Union[assign.WhenElseFrame, assign.SwitchOnceFrame]], 
        assignment: _Assignment,
    ) -> None:
        """ 
        assignment: (key, value, value_key)
        is_analog: True for wor|wand|tri
        is_reg: True for reg
        """
        assert isinstance(assignment, _Assignment)
        # direct assign in current frame 
        if not conds: 
            if isinstance(assignment.value, Reader):
                if not self.is_reg:
                    self.gate.read(assignment.value) 
            # key = (start, width)
            if assignment.key is not None: 
                for i in assignment.key:
                    if isinstance(i, Reader):
                        if not self.is_reg:
                            self.gate.read(i) 
            # key = None, full assignment
            else:
                if not self.is_analog:
                    self.frames.clear() 
            if assignment.value_key is not None:
                for i in assignment.value_key:
                    if isinstance(i, Reader):
                        if not self.is_reg:
                            self.gate.read(i)                    
            self.frames.append(assignment)
            self.curr_frame = None
            
        else:
            curr_frame = conds[0]
            # append a new frame 
            if self.curr_frame is not curr_frame:
                self.frames.append([]) 
                self.curr_frame = curr_frame 
            # list of condition
            curr_branches = curr_frame.branches 
            # list of (condition, CondTreeNode)
            last_frame = self.frames[-1]
            # update frame to catch previous conditions 
            if len(curr_branches) > len(last_frame):
                for i in range(len(last_frame), len(curr_branches)):
                    cond: Optional[Tuple[Reader, Any]] = curr_branches[i]
                    last_frame.append(
                        (cond, CondTreeNode(self.gate, self.is_analog, self.is_reg))
                    ) 
                    # record cond signal
                    if cond is not None:
                        if not self.is_reg:
                            self.gate.read(cond[0])
                        
            # recursive
            _, next_tree_node = last_frame[-1]
            next_tree_node.insert(conds[1:], assignment)  

    def merge(self, other: CondTreeNode):
        self.frames = other.frames + self.frames 
        self.read(other)
            
    def read(self, node: CondTreeNode):
        for frame in node.frames:
            if isinstance(frame, _Assignment):
                if isinstance(frame.value, Reader):
                    self.gate.read(frame.value)
        # TODO unread
    
    def eval(self, ret: List[_Assignment]) -> bool:
        """ return reversed order
        evaluate conds at simulation time and return list of assignments
        
        return True if early break
        """  
        # check from tail to head
        for frame in reversed(self.frames):
            if isinstance(frame, _Assignment):
                ret.append(frame)
                # key = None, full assignment, ignore rest and return
                if frame.key is None and not self.is_analog:
                    return True
            # conditions in one frame are mutually exclusive
            else:
                for cond, node in frame: 
                    # break if one branch is true
                    if self._eval_cond(cond): 
                        if node.eval(ret):
                            return True 
                        else:
                            break  
        return False
        
    def _eval_cond(self, cond: Optional[Tuple[Reader, Any]]) -> bool:
        if cond is None:
            return True 
        else:
            signal_data = cond[0]._getval()
            return any(i == signal_data for i in cond[1:])
        
    def __str__(self) -> str: 
        def cond_str(cond):
            if cond is None:
                return '_:'
            else:
                return f"{cond[0]._name} == {'|'.join(str(x) for x in cond[1:])}:"
        body = []
        body.append('?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????')
        for i, branches in enumerate(self.frames): 
            if isinstance(branches, _Assignment):
                body.append(f'?????? {self.gate}[{branches.key}] <- {branches.value}')
            else:
                for cond, node in branches:
                    body.append('???'+cond_str(cond))
                    body.append('???  '+'???  '.join(str(node).splitlines(keepends=True)))
                if i < len(self.frames) - 1:
                    body.append('????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????')
        body.append('?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????')
        return '\n'.join(body) 
    
    def emitVerilog(self, op: str = '=') -> str:
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
        for i in self.frames:
            if isinstance(i, _Assignment):
                ret.append(self._verilog_assign(i, self.gate.output, op))
            else:
                for idx, t in enumerate(i):
                    cond, node = t  
                    ret.append(self._verilog_cond(idx, cond) + ' begin')
                    ret.extend('  ' + x for x in node.emitVerilog(op).splitlines())
                    ret.append('end') 
        return '\n'.join(ret)
        
        
    def _verilog_cond(self, idx: int, cond: Optional[Tuple[Reader, Any]]) -> str:
        """ ex. else if (x==1)
        """
        if cond is None:
            return 'else'
        
        signal =cond[0]
        x = self.gate.get_name(signal) 
        ret = []
        for i in cond[1:]:
            if isinstance(i, BitPat): 
                ret.append(f'{x} ==? {signal._type._verilog_immd(i)}') 
            else:   # mpz
                ret.append(f'{x}=={signal._type._verilog_immd(i)}')

        ret = ' && '.join(ret)
        if idx == 0:
            return f'if({ret})'
        else:
            return f'else if({ret})'
            
        
    def _verilog_assign(
        self, 
        assignment: _Assignment,
        out: Writer,
        op: str='='
    ):
        """ ex. x[waddr +: 8] = y[raddr +: 8];
        """
        key, value, value_key = assignment.key, assignment.value, assignment.value_key
        if isinstance(value, Reader):
            value = self.gate.get_name(value) + value._type._verilog_key(value_key, self.gate)
        else:
            assert value_key is None
            value = out._type._verilog_immd(value)
        
        target = self.gate.get_name(out) + out._type._verilog_key(key, self.gate)
        
        return f'{target} {op} {value};'
        


def _part_select_wire(signal: Reader, keys: Any) -> Reader:
    if type(keys) is SignalKey:
        return _Wire(signal)
    else:
        low_key ,T = signal._type._slice(keys)
        return _Wire(signal, next=True, input_key=low_key, output_type = T)


class Assignable(Gate):
    """ wire, reg, ..., multiple inputs, single output  
    """
    
    condtree: CondTreeNode 
    output: Writer 

    def __part_select__(self, signal: Reader, keys: Any) -> Reader:  
        return _part_select_wire(signal, keys)

    def __partial_assign__(
        self, 
        signal: Reader,
        conds: List[Union[assign.WhenElseFrame, assign.SwitchOnceFrame]],
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
            low_key ,T = signal._type._slice(keys) 
        # get valid immd
        if not isinstance(value, Reader):
            value = T._eval(value)

        self.condtree.insert(conds, _Assignment(low_key, value, None)) 

    def __merge__(self, other: Assignable):
        raise Exception()
               

        
class _Wire(Assignable): 
    
    id = 'Wire'         # id
    delay: int          # timing
    condtree: CondTreeNode 
    output: Writer 
    _netlist = 'logic'   # netlist type
  
    def __head__(
        self, 
        x: Reader, 
        *, 
        id: str = '', 
        next: bool = False, 
        input_key = None, 
        output_type = None
    ): 
        """ turn into wire
        
        if next, return new signal, width=len(x); 
        else insert a wire between input x and its writer, width=len(x._data)
        
        input_key: low_key
        """
        self.id = id or self.id    # timing config id   
        self.condtree =  CondTreeNode(gate=self, is_analog=False, is_reg=False)
        
        # turn into assignable
        if not next:
            if x._data.writer is None:
                self.output = self.write(x)
                self.condtree.insert(conds=[], assignment=_Assignment(None, x._getval(), None)) 
                return x 
            elif not isinstance(x._data.writer._driver, Assignable):
                # driver -> new_x -> self -> x
                new_data = x._data.copy() 
                odd_type = x._data.writer._type
                x._data.writer._exchange(new_data)
                self.condtree.insert(conds=[], assignment=_Assignment(None, Reader(new_data, odd_type), None)) 
                self.output = self.write(Writer(x._data, odd_type, self))
                return x 
            else:
                raise ValueError(f'singal {x} already assignable')
        else:
            ret_type = output_type if output_type is not None else x._type
            ret = ret_type()
            self.output = self.write(ret) 
            self.condtree.insert(conds=[],  assignment=_Assignment(None, x, input_key)) 
            return ret
            
            
    def forward(self):
        """ simulation function 
        """
        assignments: List[_Assignment] = [] 
        self.condtree.eval(assignments)
        if len(assignments) == 1:
            key, value = assignments[0].eval()
            self.output._setval(value, self.delay, key)  
        else:
            _data = self.output._data.copy()
            _type = self.output._type
            for assignment in reversed(assignments):
                key, value = assignment.eval()
                _type._setimmd(_data, key, value) 
            value = _type._getimmd(_data, None) 
            self.output._setval(value, self.delay, None)          
    
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str: 
        body = self.condtree.emitVerilog('=') 
        body = '  ' + '  '.join(body.splitlines(keepends=True))
        return '\n'.join(['always_comb begin', body , 'end'])
            
        
    def __str__(self):
        return f'Wire(name={self.output._data._name})'



class _Reg(Assignable): 
  
    id = 'Reg'         # id
    delay: int          # get_timing
    condtree: CondTreeNode 
    output: Writer 
    _netlist = 'logic'   # netlist type
    _sess: _session.Session
  
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
        self.condtree =  CondTreeNode(gate=self, is_analog=False, is_reg=True)
        
        self.posedge_clk: List[Reader, gmpy2.mpz] = []
        self.negedge_clk: List[Reader, gmpy2.mpz] = [] 
        self.pos_rst: Reader = None
        self.neg_rst: Reader = None
        self.reset_value: gmpy2.mpz = x._getval()
        
        if clock is ...:   clock = self._sess.module.clock 
        if reset is ...:   reset = self._sess.module.reset

        # record initial value of clock
        if clock[1]:
            self.posedge_clk = [self.read(clock[0]), clock[0]._getval()] 
        else:
            self.negedge_clk = [self.read(clock[0]), clock[0]._getval()] 
            
        # record initial value of reset 
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
            self.condtree.insert(conds=[], assignment=_Assignment(None, x, None)) 
            return x  
        # reg next
        else:
            ret = Reader(x._data.copy(), x._type)
            self.output = self.write(ret)
            self.condtree.insert(conds=[], assignment=_Assignment(None, x, None)) 
            return ret 

    def forward(self):
        if (rst:=self.pos_rst) is not None: 
            if rst._getval():
                self.output._setval(self.reset_value, self.delay) 
                return
        if (rst:=self.neg_rst) is not None:
            if not rst._getval():
                self.output._setval(self.reset_value, self.delay) 
                return 
        if s:=self.posedge_clk: 
            clk, odd = s 
            new = clk._getval()
            s[1] = new 
            if not (odd==0 and new != 0):
                return    
        if s:=self.negedge_clk:
            clk, odd = s 
            new = clk._getval() 
            s[1] = new  
            if not (odd != 0 and new == 0):
                return  
            
        assignments: List[_Assignment] = []  
        self.condtree.eval(assignments)   # 800ms  

        if len(assignments) == 1:
            key, value = assignments[0].eval()
            self.output._setval(value, self.delay, key) 
        else: 
            _data = self.output._data.copy()
            _type = self.output._type  
            for assignment in reversed(assignments):
                key, value = assignment.eval() 
                _type._setimmd(_data, key, value)
            value = _type._getimmd(_data, None)
            self.output._setval(value, self.delay, None)
            
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str: 
        triggers = [] 
        has_reset = False
        if (rst:=self.pos_rst) is not None: 
            triggers.append(f'posedge {self.get_name(rst)}')  
            reset = self.get_name(rst)
            has_reset = True 
        elif (rst:=self.neg_rst) is not None:
            triggers.append(f'negedge {self.get_name(rst)}') 
            reset = f'!{self.get_name(rst)}'
            has_reset = True  
            
        if self.posedge_clk:
            triggers.append(f'posedge {self.get_name(self.posedge_clk[0])}')
        else:
            triggers.append(f'negedge {self.get_name(self.negedge_clk[0])}')
        triggers = ' or '.join(triggers)
        
        out = self.get_name(self.output)
        body = self.condtree.emitVerilog('<=')
        body = '    ' + '    '.join(body.splitlines(keepends=True))
        if has_reset:
            body = f'  if ({reset}) {out} <= {utils.const_int(self.reset_value)}; else begin\n{body}\n  end'

        return '\n'.join([
            # FIXME modelsim error
            # f'initial begin {out} = {self.reset_value}; end',
            f'always_ff @({triggers}) begin',
            body, 
            'end'
        ])
            

    def __str__(self):
        return f'Reg(name={self.output._data._name})'



class _Latch(Assignable):
    
    id = 'Latch'
    delay: int 
    condtree: CondTreeNode 
    output: Writer 
    _netlist = 'logic'
    
    def __head__(self, x: Reader, *, id: str = '') -> Reader:
        """ no LatchNext, turn x into latch

        TODO reset
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
            self.output._setval(value, self.delay, key)  
        else:
            _data = self.output._data.copy()
            _type = self.output._type
            for assignment in reversed(assignments):
                key, value = assignment.eval()
                _type._setimmd(_data, key, value) 
            value = _type._getimmd(_data, None) 
            self.output._setval(value, self.delay, None)      

                
    def emitVerilog(self, v: verilogmodule.Verilog) -> str:
        body = self.condtree.emitVerilog('=') 
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
            self.posedge_clk = [self.read(clock[0]), clock[0]._getval()] 
        else:
            self.negedge_clk = [self.read(clock[0]), clock[0]._getval()] 

        self.output = self.write(x)
        return x 


    def __part_select__(self, signal: Reader, keys: Any) -> Reader:  
        return _part_select_wire(signal, keys)


    def __partial_assign__(
        self, 
        signal: Reader,
        conds: List[Union[assign.WhenElseFrame, assign.SwitchOnceFrame]],
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
            new = clk._getval()
            s[1] = new 
            if not (odd==0 and new != 0):
                return    
        if s:=self.negedge_clk:
            clk, odd = s 
            new = clk._getval() 
            s[1] = new  
            if not (odd != 0 and new == 0):
                return  
            
        assignments: List[_Assignment] = []  
        self.condtree.eval(assignments)   # 800ms   
        if not assignments: 
            return
        # TODO simulation error if more than one assignments 
        key, value = assignments[0].eval()
        self.output._setval(value, self.delay, key) 
            
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str: 
        triggers = [] 
            
        if self.posedge_clk:
            triggers.append(f'posedge {self.get_name(self.posedge_clk[0])}')
        else:
            triggers.append(f'negedge {self.get_name(self.negedge_clk[0])}')
        triggers = ' or '.join(triggers)
        
        out = self.get_name(self.output)
        body = self.condtree.emitVerilog('<=')
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
        self.default = x._getval()
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
        conds: List[Union[assign.WhenElseFrame, assign.SwitchOnceFrame]],
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
            self.output._setval(self.default, self.delay, None)
        else:
            key, value = assignments[0].eval()
            self.output._setval(value, self.delay, key)  
    
    
    def emitVerilog(self, v) -> str: 
        body = self.condtree.emitVerilog('=')  
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
        self.clk: Reader = UInt[1](0) 
        self.write(self.clk)
        self.clk._name = id.lower() 
        # get timing
        timing = self._sess._get_timing(self.id) or self.timing 
        self.low: int = timing['low'] 
        self.high: int = timing['high']
        self.phase: int = timing['phase']
        # insert an event
        self._sess.simulator.insert_coroutine_event(0, self.clock_event())
        return self.clk

    def get_timing(self):
        pass
    
    def clock_event(self):
        clk = self.clk
        low = self.low 
        high = self.high 
        phase = self.phase
        zero = gmpy2.mpz(0) 
        one = gmpy2.mpz(1)

        yield (phase+low-1) 
        clk._setval(one, 0)
        while 1:
            yield high
            clk._setval(zero, 0)
            yield low
            clk._setval(one, 0)
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str: 
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
    
    
        

    
class DummyGate(Assignable):

    id = 'DummyGate'

    def __head__(self, inputs = [], outputs = []): 
        self.inputs: List[Reader] = []
        self.outputs: List[Writer] = []
        for i in inputs:
            self.inputs.append(self.read(i))
        for i in outputs:
            self.outputs.append(self.write(i))
        self.verilog: List[str] = []
        return self 

    def __partial_assign__(self, *args, **kwargs) -> None: 
        raise Exception()


    def delete(self): 
        for i in self.iports:
            i._driven.pop(self)
        for o in self.oports: 
            o._delete() 
        self.iports.clear()
        self.oports.clear()
        self._sess._remove_gate(self)
    
    def append(self, v: str):
        self.verilog.append(v)
        
    def assign(self, s: Reader, v: Any):
        self.verilog.append(f'{self.get_name(s)} = {v};')
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str:
        self.verilog.clear()
        self.__body__()
        return '\n'.join(self.verilog)

    def __body__(self):
        return 


def inline_verilog(f: Callable):
    assert len(inspect.signature(f).parameters) == 1
    type(f.__name__, (DummyGate,),{'__body__':f})()

"""
@blackbox
def emit(self, verilog):
    name = v.insert_module('module adder(x,y); endmodule')

    x = self.get_name(dut.x)
    y = self.get_name(dut.y) 
    return f'''
initial begin
    # 3 
    {x} = 1 
end
    '''

"""
# TODO gate __slots__ 
    

@vectorize_first
def setv(left: Reader, right: Any, key = None, dt=0): 
    assert dt >= 0
    t = left._type 
    if key is not None:
        low_key, T = t._slice(key)  
    else:
        low_key = None
    left._setval(t._eval(right), dt, low_key)
    
    
@vectorize_first
def setr(left: Reader): 
    t = left._type 
    ret = random.randrange(1 << len(t))
    left._setval(t._eval(ret), 0) 
    return ret

@vectorize 
def getv(signal: Reader, key = None, format=None):
    if key is None:
        ret = signal._getval()
    else:
        low_key, T = signal._type._slice(key)
        ret = signal._getval(low_key)
    
    if format is None:
        return ret
    else:
        raise ValueError('unsupported format')




