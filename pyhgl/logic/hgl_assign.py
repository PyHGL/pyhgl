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
import sys
import gmpy2

from pyhgl.array import *
import pyhgl.logic._session as _session
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic.hgl_basic as hgl



class _Ports(HGLFunction):
    
    def __init__(self, direction: str) -> None:
        self.direction = direction 
    
    def __call__(self, x) -> Any: 
        x = Signal(x) 
        def f(s: hgl.Reader):
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
    copy signals with direction. does not copy default value
    """
    _sess: _session.Session
    
    def __call__(self, x: Union[hgl.Reader, Array]) -> Any:   
        x = Signal(x)
        def f(s: hgl.Reader):
            ret = s._type()
            ret._direction = s._direction 
            return ret
        return Map(f, x) 
    
    
@singleton 
class Flip(HGLFunction):
    """
    copy signals with flipped direction.  does not copy default value
    """
    
    _sess: _session.Session
    
    def __call__(self, x: Union[hgl.Reader, Array]) -> Any:  
        x = Signal(x)
        def f(s: hgl.Reader):
            ret = s._type()
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
            assert isinstance(x, hgl.Reader)
            if x._direction == 'input':
                assert x._data.writer is None  
                hgl.DummyGate(outputs = [x]) 
                module._temp_inputs.append(x)
            elif x._direction == 'inner':
                x._direction = 'output' 
        Map(make_io, x) 
        module_hgl._rename_array(x, 'io')
        return x





@dispatch('Matmul', Any, [HGLFunction, type(lambda _:_), type])
def _call(x, f):
    """ Signal|Array @ Callable
    """
    return f(x) 







@register_builtins 
def  __hgl_partial_assign__(left: Union[hgl.Reader, Array], right, keys=SignalKey()) -> None:  
    """ PyHGL operator, ex. left[key] <== right
    """
    # turn iterable into array
    left, right = ToArray(left), ToArray(right)
    # get conditions
    f_locals=sys._getframe(1).f_locals
    cond_stack: CondStack = f_locals.get('__hgl_condition__')
    if isinstance(cond_stack, CondStack):
        conds = cond_stack.active_signals
    else:
        conds = [None] 
        
    assert hasattr(left, '__partial_assign__'), 'require Signal or Array'
    left.__partial_assign__(conds, right, keys)



@register_builtins('__hgl_connect__')
@vectorize
def __hgl_connect__(left: hgl.Reader, right: hgl.Reader) -> None:
    """ PyHGL operator <=>
    """
    assert isinstance(left, hgl.Reader) and isinstance(right, hgl.Reader)
        
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
        left_gate: hgl.Assignable = left._data.writer._driver
        right_gate: hgl.Assignable = right._data.writer._driver 
        left_gate.__merge__(right_gate)
    


"""
store nested conditions in python local frame: locals()['__hgl_condition__'] 

Functional dependency: Bool, UInt, SignalType._eval
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
        self.branches: List[Optional[Tuple[hgl.Reader, gmpy2.mpz]]] = []    

    def when(self, signal:hgl.Reader):
        # accept one bit signal
        assert len(self.branches) == 0  
        self.branches.append((signal, gmpy2.mpz(1)))

    def elsewhen(self, signal: hgl.Reader):
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


class CaseFrame(hgl.Gate):
    """
    switch (x, 'unique'):
        once 1: ... 
        once 'idle': ... 
        once 3,4,5: ...
        once Bitpat('??'): ... 
        once ...: ...

    when a: ... 
    elsewhen b: ... 
    elsewhen c: ... 
    otherwise: ...

    - `when` statement is converted to `switch(True)` statement 
    - case expression: single signal|immd, not BitPat or Bundle
    - case items: single signal|immd|BitPat
    - do bitwise comparation, which may differ from `==`
    """
    def __head__(self, sel: Union[hgl.Reader, Tuple], active_signal: hgl.Reader = None):
        """ sel: signal | (signal, flag)
        """
        self.type: hgl.SignalType = None # TODO 
        # unique case flag
        if isinstance(sel, (tuple, list)):
            assert len(sel) == 2
            sel_signal = sel[0]
            assert sel[1] == 'unique', 'unknown flag'
            self.flag_unique: bool = True
        else:
            sel_signal = sel 
            self.flag_unique: bool = False
        # whether has the default branch
        self.complete: bool = False
        # init signals
        self.sel_signal: hgl.Reader = self._to_single_signal(sel_signal)
        self.active_signal = active_signal
        self.read(self.sel_signal) 
        if active_signal is not None:
            self.read(active_signal)
        # case items, ex. [(1, a), (2, a), (signal, b), (None, c)]
        self.items: List[Tuple[Union[int, hgl.Reader, hgl.BitPat, None], hgl.Reader]] = []

        return self 

    def _to_single_signal(self, v: Union[hgl.Reader, int, str]) -> hgl.Reader:
        if isinstance(v, hgl.Reader):
            return v 
        v = Signal(v)
        assert isinstance(v, hgl.Reader), 'requires sigle signal'
        assert v._type._storage in ['packed', 'packed variable'], 'unsupported signal type'
        return v


    def when(self, signal: Union[hgl.Reader, int, str]) -> hgl.Reader:
        """ return a 1-bit active signal 
        """ 
        assert len(self.items) == 0
        signal = self._to_single_signal(signal)
        ret = hgl.UInt[1](0)
        self.items.append((self.read(signal), self.write(ret)))
        return ret 
    
    def elsewhen(self, signal: Union[hgl.Reader, int, str]) -> hgl.Reader:
        assert len(self.items) > 0, 'no `when` statement before `elsewhen`'
        signal = self._to_single_signal(signal)
        ret = hgl.UInt[1](0)
        self.items.append((self.read(signal), self.write(ret)))
        return ret 
        
    def otherwise(self) -> hgl.Reader:
        assert len(self.items) > 0, 'no `when` statement before `otherwise`'
        ret = hgl.UInt[1](0)
        self.items.append((None, self.write(ret)))
        return ret 

    def case(self, items:  Tuple[Union[int, hgl.Reader, hgl.BitPat]]) -> hgl.Reader:
        """ 
        accept multiple case items,
        BitPat and special immd is allowed
        """
        assert not self.complete, "case item after default"

        if items[0] is ...:
            assert len(items) == 1, 'invalid syntax'
            assert self.items, "no case items before default"
            self.complete = True
            ret = hgl.UInt[1](0)
            self.items.append((None, self.write(ret)))
            return ret
        else: 
            # single signal, immd, or BitPat
            items_valid = []
            for i in items:
                if isinstance(i, hgl.Reader):  
                    # does not support variable width 
                    assert i._type._storage == 'packed', 'case item must be fixed width'
                    items_valid.append(self.read(i)) 
                else:
                    items_valid.append(self.sel_signal._eval(i))

            ret = hgl.UInt[1](0)
            _write = self.write(ret)
            for i in items_valid: 
                self.items.append((i, _write))
            return ret

    def forward(self) -> None:
        pass # TODO 


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
    def __init__(self, sel: hgl.Reader):
        """  
        - condition: None | (signal, immd, ...)
            - immd: gmpy2.mpz | state-string | BitPat
            
        ex. [(a, 0), (b,1,2,3), (c,BitPat('1?')), (d,'idle'), None]
        """
        self.branches: List[Optional[Tuple[hgl.Reader, int]]] = []    
        self.complete: bool = False  # has default branch
        self.unique: bool = False   # unique case

    def case(self, items: Tuple[Union[int, hgl.Reader, hgl.BitPat, dict, list]]):
        if self.complete:
            raise Exception("case item after default")
        elif items[0] is ...:
            assert self.branches, "no case items before default"
            self.branches.append(None) 
            self.complete = True
            return          
        else: 
            items = [self.sel._type._eval(i) for i in items]
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
        self.stacks: List[CaseFrame] = []  
        # exit 'elsewhen' and 'otherwise' stmt
        self.prev_stack: CaseFrame = None     
        
        self.active_signals: List[Optional[hgl.Reader]] = [None]

    def new(self, sel: Union[hgl.Reader, Tuple]):
        """ new 'when' stmt or 'switch' stmt 
        """
        self.stacks.append(CaseFrame(sel, self.active_signals[-1]))

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
        if self.prev_stack is None:
            raise Exception("'No 'when' stmt before 'elsewhen' and 'otherwise' stmt") 
        else: 
            self.stacks.append(self.prev_stack)
            self.prev_stack = None  



    @property
    def tail(self) -> CaseFrame:
        return self.stacks[-1]

    
    def __str__(self):
        ret = ['>>> Condition frames:']
        for i, j in enumerate(self.stacks):
            ret.append(f'level{i+1:>2}: {j}')
        return '\n'.join(ret)
    
    
    
    
@register_builtins
class __hgl_when__(HGL):
    def __init__(self, signal): 
        """
        when expr: 
            ... 
        """
        f_locals = sys._getframe(1).f_locals
        if '__hgl_condition__' not in f_locals: 
            f_locals['__hgl_condition__'] = CondStack()
        self.condframe: CondStack = f_locals['__hgl_condition__']
        self.signal = signal

    def __enter__(self):
        self.condframe.new(1)               # new `switch(1) ...` 
        active = self.condframe.tail.when(self.signal)
        self.condframe.active_signals.append(active)

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.active_signals.pop()
        self.condframe.pop_store()                  # store for potential 'elsewhen' 'otherwise'
    

@register_builtins
class __hgl_elsewhen__(HGL):
    def __init__(self, signal):

        f_locals = sys._getframe(1).f_locals 
        assert '__hgl_condition__' in f_locals, "no 'when' stmt before 'elsewhen' stmt"
        self.condframe: CondStack = f_locals['__hgl_condition__']
        self.signal = signal

    def __enter__(self):
        self.condframe.restore()
        active = self.condframe.tail.elsewhen(self.signal)
        self.condframe.active_signals.append(active)

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.active_signals.pop()
        self.condframe.pop_store()
    

@register_builtins
class __hgl_otherwise__(HGL): 
    def __init__(self):
        f_locals = sys._getframe(1).f_locals 
        assert '__hgl_condition__' in f_locals, "no 'when' stmt before 'elsewhen' stmt"
        self.condframe: CondStack = f_locals['__hgl_condition__']

    def __enter__(self):
        self.condframe.restore()            
        active = self.condframe.tail.otherwise()
        self.condframe.active_signals.append(active)

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.active_signals.pop()
        self.condframe.pop_nostore()
    
    


@register_builtins
class __hgl_switch__(HGL):
    def __init__(self, signal): 
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
        self.signal = signal

    def __enter__(self):
        self.condframe.new(self.signal)         # new frame 

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.pop_nostore()


@register_builtins
class __hgl_once__(HGL):
    def __init__(self, signal):
        f_locals = sys._getframe(1).f_locals 
        assert '__hgl_condition__' in f_locals, "no 'when' stmt before 'elsewhen' stmt"
        self.condframe: CondStack = f_locals['__hgl_condition__']
        self.signal = signal
        
    def __enter__(self):
        active = self.condframe.tail.case(self.signal)
        self.condframe.active_signals.append(active)

    def __exit__(self, exc_type, exc_value, traceback):
        self.condframe.active_signals.pop()        









