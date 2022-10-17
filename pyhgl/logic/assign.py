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
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import builtins
import sys
import gmpy2

from pyhgl.array import *
import pyhgl.logic.hardware as hardware
import pyhgl.logic._session as _session
import pyhgl.logic.hglmodule as hglmodule



class _Ports(HGLFunction):
    
    def __init__(self, direction: str) -> None:
        self.direction = direction 
    
    def __call__(self, x) -> Any: 
        x = Signal(x) 
        def f(s: hardware.Reader):
            s._direction = self.direction
        Map(f, x)  
        return x
    
    def __matmul__(self, x):
        return self.__call__(x)
    
    def __rmatmul__(self, x):
        return self.__call__(x)
    
Input = _Ports('input')
Output = _Ports('output')
InOut = _Ports('inout')
Inner = _Ports('inner')
        


@singleton 
class Copy(HGLFunction):
    """
    copy signals with direction
    """
    _sess: _session.Session
    
    def __call__(self, x: Union[hardware.Reader, Array]) -> Any:   
        x = Signal(x)
        def f(s: hardware.Reader):
            ret = hardware.Reader(s._data.copy(), s._type)
            ret._direction = s._direction 
            return ret
        return Map(f, x) 
    
    
@singleton 
class Flip(HGLFunction):
    """
    copy signals with flipped direction
    """
    
    _sess: _session.Session
    
    def __call__(self, x: Union[hardware.Reader, Array]) -> Any:  
        x = Signal(x)
        def f(s: hardware.Reader):
            ret = hardware.Reader(s._data.copy(), s._type)
            if s._direction == 'input':
                ret._direction = 'output'
            elif s._direction == 'output':
                ret._direction = 'input' 
            elif s._direction == 'inner':
                ret._direction = 'input'
            else:
                ret._direction = s._direction 
            return ret
        return Map(f, x)
    

@singleton 
class IO(HGLFunction):
    """
    record io to current module
    insert a DummyGate to each input signal
    """
    _sess: _session.Session
    
    def __call__(self, x: Array) -> Array:  
        module = self._sess.module 
        assert isinstance(x, Array)
        assert len(module._position) > 1, 'IO should be called inside a module'

        def make_io(x):
            assert isinstance(x, hardware.Reader)
            if x._direction == 'input':
                assert x._data.writer is None  
                hardware.DummyGate(outputs = [x]) 
                module._temp_inputs.append(x)
            elif x._direction == 'inner':
                x._direction = 'output' 
        Map(make_io, x) 
        hglmodule._rename_array(x, 'io')
        return x





@dispatch('Matmul', Any, [HGLFunction, type(lambda _:_), type])
def _call(x, f):
    """ Signal|Array @ Callable
    """
    return f(x) 




    

# TODO inouts
# @vectorize 
# def InOut(x: Union[int, str, hardware.Reader]) -> hardware.Reader:
#     x = Signal(x) 
#     assert isinstance(x._data.writer._driver, hardware.Analog)
#     assert x._data.writer is None  
#     x = hardware.Wire(x)
#     x._sess.module._module.inou[x] = None  
#     return x 

# def InOut(
#     *args: Union[int, str, hardware.Reader, list, Array, Iterable]
# ) -> Union[hardware.Reader, Array]:

#     if len(args) > 1:
#         args = Array(args) 
#     else:
#         args = args[0]
        
#     def _inout(signal: hardware.Reader):
#         assert signal._data.writer is None, 'InOut only accept constant'
#         return hardware.Wtri(signal)

#     return Map(_inout, hardware.Signal(args))



    
def _register_builtins(cls):
    if hasattr(cls, '__hgl_wrapped__'):
        name = cls.__hgl_wrapped__.__name__
    else:
        name = cls.__name__ 
    setattr(builtins, name, cls)
    return cls



@_register_builtins 
def  __hgl_partial_assign__(left: Union[hardware.Reader, Array], right, keys=SignalKey()) -> None:  
    """ PyHGL operator, ex. left[key] <== right
    """
    # turn iterable into array
    left, right = ToArray(left), ToArray(right)
    # get conds
    f_locals=sys._getframe(1).f_locals
    cond_stack: CondStack = f_locals.get('__hgl_condition__')
    if isinstance(cond_stack, CondStack):
        conds = cond_stack.stacks 
    else:
        conds = [] 
        
    assert hasattr(left, '__partial_assign__'), 'require Signal or Array'
    left.__partial_assign__(conds, right, keys)



@_register_builtins
@vectorize
def __hgl_connect__(left: hardware.Reader, right: hardware.Reader) -> None:
    """ PyHGL operator <=>
    """
    assert isinstance(left, hardware.Reader) and isinstance(right, hardware.Reader)
        
    if left._data.writer is None and right._data.writer is not None:
        assert type(left._data) is type(right._data) 
        left._exchange(right._data)
    elif left._data.writer is not None and right._data.writer is None:
        assert type(left._data) is type(right._data)   
        right._exchange(left._data)
    elif left._data.writer is None and right._data.writer is None:
        assert type(left._data) is type(right._data) 
        left._exchange(right._data)
    else:
        left_gate: hardware.Assignable = left._data.writer._driver
        right_gate: hardware.Assignable = right._data.writer._driver 
        left_gate.__merge__(right_gate)
    


"""
store conditions in python local frame: locals()['__hgl_condition__'] 
"""


class WhenElseFrame(HGL):
    """
    'when' stmt
    
    when a: ... 
    elsewhen b: ... 
    elsewhen c: ... 
    otherwise: ...
    """
    def __init__(self):
        """ list of condition 
        
        - condition: None | (signal, 0|1)
            
        ex. [(a, 0), (b,1), (c,1), (d,2), None]
        """
        # list of (signal, value), ex. [(a, 1), (b,1), (c,1), None]
        self.branches: List[Optional[Tuple[hardware.Reader, gmpy2.mpz]]] = []    

    def when(self, signal:hardware.Reader):
        # accept one bit signal
        assert len(self.branches) == 0  
        self.branches.append((signal, gmpy2.mpz(1)))

    def elsewhen(self, signal: hardware.Reader):
        assert len(self.branches) > 0  
        self.branches.append((signal, gmpy2.mpz(1)))

    def otherwise(self):
        assert len(self.branches) > 0
        self.branches.append(None)  
        
    def __str__(self):
        ret = []
        for i in self.branches:
            if i is None:
                ret.append('None')
            else:
                ret.append(f"{i[0]._name} == {'|'.join(str(x) for x in i[1:])}")
        return ', '.join(ret)


class SwitchOnceFrame(HGL):
    """
    stack for each 'switch' stmt
    switch x:
        once 1: ... 
        once 2: ... 
        once 3,4,5: ...
        once Bitpat('??'): ... 
        once ...: ...
    note: only support immd for case items
    """
    def __init__(self, sel: hardware.Reader):
        """  
        - condition: None | (signal, immd, ...)
            - immd: gmpy2.mpz | state-string | BitPat
            
        ex. [(a, 0), (b,1,2,3), (c,BitPat('1?')), (d,'idle'), None]
        """
        self.branches: List[Optional[Tuple[hardware.Reader, int]]] = []    
        self.sel: hardware.Reader = sel    
        self.complete: bool = False

    def case(self, right: Tuple[Union[int, hardware.Reader, hardware.BitPat, dict, list]]):
        if self.complete:
            raise Exception("case item after default")
        elif right[0] is ...:
            assert self.branches, "no case items before default"
            self.branches.append(None) 
            self.complete = True
            return          
        else: 
            items = [self.sel._type._eval(i) for i in right]
            self.branches.append((self.sel, *items))    

    def __str__(self):
        ret = []
        for i in self.branches:
            if i is None:
                ret.append('None')
            else:
                ret.append(f"{i[0]._name} == {'|'.join(str(x) for x in i[1:])}")
        return ', '.join(ret)
     


class CondStack:
    """ stack of when/switch frames; stored in local scope of functions
    """
    def __init__(self):

        # grow for nested 'when', 'switch'
        self.stacks: List[Union[WhenElseFrame, SwitchOnceFrame]] = []  
        # exit 'elsewhen' and 'otherwise' stmt
        self.prev_stack: WhenElseFrame = None     
        

    def new(self, stack: Union[WhenElseFrame, SwitchOnceFrame]):
        """ new 'when' stmt or 'switch' stmt 
        """
        self.stacks.append(stack)

    def pop_store(self):
        """ for 'when', 'elsewhen' stmt
        """
        self.prev_stack = self.stacks.pop()

    def pop_nostore(self):
        """ for 'otherwise'  stmt
        """
        self.stacks.pop()

    def restore(self): 
        """ only for 'elsewhen' and 'otherwise' stmt 
        """
        if (self.prev_stack is None) or (not isinstance(self.prev_stack, WhenElseFrame)):
            raise Exception("'No 'when' stmt before 'elsewhen' and 'otherwise' stmt") 
        else: 
            self.stacks.append(self.prev_stack)
            self.prev_stack = None  

    @property
    def tail(self) -> Union[WhenElseFrame, SwitchOnceFrame]:
        return self.stacks[-1]

    
    def __str__(self):
        ret = ['>>> Condition frames:']
        for i, j in enumerate(self.stacks):
            ret.append(f'level{i+1:>2}: {j}')
        return '\n'.join(ret)
    
    
    
    
    
@_register_builtins
class __hgl_when__(HGL):
    def __init__(self, signal: hardware.Reader): 
        """
        when expr: 
            ... 
        """
        f_locals = sys._getframe(1).f_locals
        if '__hgl_condition__' not in f_locals: 
            f_locals['__hgl_condition__'] = CondStack()
        self.condframe: CondStack = f_locals['__hgl_condition__']
        
        assert isinstance(signal, hardware.Reader)
        self.signal = Bool(signal)  

    def __enter__(self):
        self.condframe.new(WhenElseFrame())         # new frame 
        self.condframe.tail.when(self.signal)

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.pop_store()                  # store for potential 'elsewhen' 'otherwise'
    

@_register_builtins
class __hgl_elsewhen__(HGL):
    def __init__(self, signal: Union[hardware.Reader, Tuple[hardware.Reader]]):

        f_locals = sys._getframe(1).f_locals 
        assert '__hgl_condition__' in f_locals, "no 'when' stmt before 'elsewhen' stmt"
        self.condframe: CondStack = f_locals['__hgl_condition__']

        assert isinstance(signal, hardware.Reader)
        self.signal = Bool(signal)

    def __enter__(self):
        self.condframe.restore()
        self.condframe.tail.elsewhen(self.signal)

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.pop_store()
    

@_register_builtins
class __hgl_otherwise__(HGL):
    """
    when expr:
        ... 
    otherwise:
        ... 
    """
    def __init__(self):
        f_locals = sys._getframe(1).f_locals 
        assert '__hgl_condition__' in f_locals, "no 'when' stmt before 'elsewhen' stmt"
        self.condframe: CondStack = f_locals['__hgl_condition__']

    def __enter__(self):
        self.condframe.restore()            
        self.condframe.tail.otherwise()

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.pop_nostore()
    
    


@_register_builtins
class __hgl_switch__(HGL):
    def __init__(self, signal: Union[hardware.Reader, Tuple[hardware.Reader]]): 
        """
        switch expr:
            case const: ... 
            case const: ... 
            ...: ...
        """
        f_locals = sys._getframe(1).f_locals
        if '__hgl_condition__' not in f_locals: 
            f_locals['__hgl_condition__'] = CondStack()
        self.condframe: CondStack = f_locals['__hgl_condition__']
        
        if not isinstance(signal, hardware.Reader):
            signal = Cat(signal) 
        self.signal = signal

    def __enter__(self):
        self.condframe.new(SwitchOnceFrame(self.signal))         # new frame 

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.pop_nostore()


@_register_builtins
class __hgl_once__(HGL):
    def __init__(self, right: Tuple[Union[int, hardware.Reader, hardware.BitPat, dict, list]]):
        condframe: CondStack = sys._getframe(1).f_locals['__hgl_condition__']
        self.switch_frame: SwitchOnceFrame = condframe.tail
        self.right = right
        
    def __enter__(self):
        self.switch_frame.case(self.right)

    def __exit__(self, exc_type, exc_value, traceback):
        pass        









