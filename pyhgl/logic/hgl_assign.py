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
import pyhgl.logic.hgl_core as hgl_core 
import pyhgl.logic.module_sv as sv



@register_builtins 
def  __hgl_partial_assign__(left: Union[hgl_core.Reader, Array], right, keys=SignalKey()) -> None:  
    """ PyHGL operator, ex. left[key] <== right
    """
    # turn iterable into array
    left, right = ToArray(left), ToArray(right)
    # get conditions
    f_locals=sys._getframe(1).f_locals
    cond_stack: CondStack = f_locals.get('__hgl_condition__')
    if isinstance(cond_stack, CondStack):
        active_signals = cond_stack.active_signals 
        cond_stacks = cond_stack.stacks
    else:
        active_signals = [None] 
        cond_stacks = []
        
    assert hasattr(left, '__partial_assign__'), 'require Signal or Array'
    left.__partial_assign__(active_signals, cond_stacks, right, keys)



@register_builtins('__hgl_connect__')
@vectorize
def __hgl_connect__(left: hgl_core.Reader, right: hgl_core.Reader) -> None:
    """ PyHGL operator <=>
    """
    assert isinstance(left, hgl_core.Reader) and isinstance(right, hgl_core.Reader)
        
    if not _has_driver(left) and _has_driver(right):
        assert type(left._data) is type(right._data) 
        __hgl_partial_assign__(left, right)
    elif _has_driver(left) and not _has_driver(right):
        assert type(left._data) is type(right._data)  
        __hgl_partial_assign__(right, left)
    elif not _has_driver(left) and not _has_driver(right):
        raise Exception(f'signals both without driver: {left}, {right}')
    else: 
        assert left._data.writer is not None and right._data.writer is not None
        left_gate: hgl_core.Assignable = left._data.writer._driver
        right_gate: hgl_core.Assignable = right._data.writer._driver 
        left_gate.__merge__(right_gate)   # analog
    
def _has_driver(signal: hgl_core.Reader) -> bool:
    return not (signal._data.writer is None and signal._data._module is None)

"""
store nested conditions in python local frame: locals()['__hgl_condition__'] 

Functional dependency: Bool, UInt, SignalType._eval
"""



class CaseGate(hgl_core.Gate):
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

    id = 'Case'
    
    def __head__(self, sel: Union[hgl_core.Reader, Tuple], active_signal: hgl_core.Reader = None):
        """ sel: signal | (signal, flag)
        """
        # unique case flag
        if isinstance(sel, (tuple, list)):
            assert len(sel) == 2 and sel[1] == 'unique', 'unknown flag'
            self.sel_signal = self.read(self._to_signal(sel[0]))
            self.flag_unique: bool = True
        else:
            self.sel_signal = self.read(self._to_signal(sel))
            self.flag_unique: bool = False
        # whether has the default branch
        self.complete: bool = False
        # frame active signal
        self.active_signal = self.read(active_signal)
        # case items, ex. [((1x1,x00), branch_1), ((signal), branch_2), (None, branch_3)]
        self.branches: List[Tuple[ 
            Optional[Tuple[ 
                hgl_core.Logic,
                hgl_core.BitPat,
                hgl_core.Reader,
            ]],     # case item
            hgl_core.Reader,        # branch active signal
        ]] = []  
        # for inout type ,  cond signal | None
        self.inout_valid: hgl_core.Reader = None
        return self 
    
    def _to_signal(self, s) -> hgl_core.Reader:
        ret = Signal(s)
        assert isinstance(ret, hgl_core.Reader) 
        assert ret._type._storage in ['packed', 'packed variable'] 
        return ret


    def when(self, signal: Union[hgl_core.Reader, int, str]) -> hgl_core.Reader:
        """ return a 1-bit active signal 
        """ 
        assert len(self.branches) == 0
        signal = Bool(self._to_signal(signal))
        ret = hgl_core.UInt[1](0, name='case_when')
        self.branches.append((
            (self.read(signal),), 
            self.write(ret),
        ))  
        self.inout_valid = signal
        return ret  
    
    def elsewhen(self, signal: Union[hgl_core.Reader, int, str]) -> hgl_core.Reader:
        assert len(self.branches) > 0, 'no `when` statement before `elsewhen`'
        signal = Bool(self._to_signal(signal))
        ret = hgl_core.UInt[1](0, name='case_elsewhen')
        self.branches.append((
            (self.read(signal),), 
            self.write(ret),
        )) 
        self.inout_valid = None
        return ret 
        
    def otherwise(self) -> hgl_core.Reader:
        assert len(self.branches) > 0, 'no `when` statement before `otherwise`'
        ret = hgl_core.UInt[1](0, name='case_otherwise')
        self.branches.append((
            None, 
            self.write(ret),
        )) 
        self.inout_valid = None
        return ret 

    def case(self, items:  Tuple[Union[int, hgl_core.Reader, hgl_core.BitPat]]) -> hgl_core.Reader:
        """ 
        accept multiple case items,
        BitPat and special immd is allowed
        """
        assert not self.complete, "case item after default"

        if items[0] is ...:
            assert len(items) == 1, 'invalid syntax'
            assert self.branches, "no case items before default"
            self.complete = True
            ret = hgl_core.UInt[1](0, name='case_default')
            self.branches.append((
                None, 
                self.write(ret),
            )) 
            return ret
        else: 
            # single signal, immd, or BitPat
            items_valid = []
            for i in items:
                if isinstance(i, hgl_core.Reader):  
                    # does not support variable width 
                    assert i._type._storage == 'packed', 'case item must be fixed width'
                    items_valid.append(self.read(i)) 
                else: 
                    val = self.sel_signal._type._eval(i)
                    if isinstance(val, hgl_core.Logic):
                        assert val.x == 0, 'unknown item is not allowed'
                    else:
                        assert isinstance(val, hgl_core.BitPat), f'value error: {val}'
                    items_valid.append(val)

            ret = hgl_core.UInt[1](0, name='case_active') 
            self.branches.append((
                tuple(items_valid), 
                self.write(ret),
            )) 
            return ret


    def sim_vx(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py  
        if self.active_signal is None:
            active_data = hgl_core.Logic(1,0)
        else:
            active_data = self.active_signal._data
        sel_data = self.sel_signal._data
        while 1:
            yield  
            active = hgl_core.Logic(active_data.v, active_data.x)
            sel = hgl_core.Logic(sel_data.v, sel_data.x)
            # disable all output
            if active.v == 0 and active.x == 0:
                for _, s in self.branches: 
                    simulator.update_v(delay, s._data, gmpy2.mpz(0))
                    simulator.update_x(delay, s._data, gmpy2.mpz(0))
                continue   
            # active == x or 1
            branch_active: List[hgl_core.Logic] = [] 
            already_matched = hgl_core.Logic(0,0)       # the default branch requires
            for item, _ in self.branches: 
                if item is None:
                    curr_matched = ~ already_matched
                else: 
                    data = (i._data._getval_py() if isinstance(i, hgl_core.Reader) else i for i in item)
                    data_bool = [i._eq(sel) for i in data]
                    curr_matched = hgl_core.Logic(0,0)
                    for i in data_bool:
                        curr_matched = curr_matched | i  

                if already_matched == hgl_core.Logic(0,0):   # no prev match
                    already_matched = curr_matched  
                elif already_matched == hgl_core.Logic(1,0): # has prev match
                    curr_matched = hgl_core.Logic(0,0)       # no curr match
                else:                                        # prev `x` match
                    if curr_matched == hgl_core.Logic(1,0):  # curr does match
                        already_matched = hgl_core.Logic(1,0)
                branch_active.append(curr_matched & active)  # maybe meta match
            
            for data, (_, signal) in zip(branch_active, self.branches):
                simulator.update_v(delay, signal._data, data.v)
                simulator.update_x(delay, signal._data, data.x)
    
    def sim_v(self):
        """ seek for first active branch, set others to zero
        """
        delay = self.timing['delay']
        simulator = self._sess.sim_py  
        if self.active_signal is None:
            active = hgl_core.Logic(1,0)
        else:
            active = self.active_signal._data

        sel = self.sel_signal._data                     # case selector
        outputs: List[hgl_core.SignalData] = []         # output signals 
        items: List[ 
            Optional[List[Union[
                hgl_core.Logic,
                hgl_core.BitPat,
                hgl_core.SignalData
            ]]]
        ] = []
        for item, s in self.branches:
            outputs.append(s._data)
            if item is None:
                items.append(item)
            else:
                items.append([i._data if isinstance(i, hgl_core.Reader) else i for i in item]) 
        items.append(None)              # dummy element

        while 1:
            yield 
            if active.v == 0:           # no branch should active
                for data in outputs:
                    simulator.update_v(delay, data, gmpy2.mpz(0))
                continue

            for i in range(len(items)):
                item = items[i] 
                if item is None:
                    break 
                for value in item:
                    if isinstance(value, hgl_core.BitPat): 
                        if (sel.v & (~value.x)) == value.v:
                            break
                    elif sel.v == value.v:
                        break
                else:
                    continue 
                break
            
            for j in range(len(outputs)):
                if j == i:
                    simulator.update_v(delay, outputs[j], gmpy2.mpz(1))
                else:
                    simulator.update_v(delay, outputs[j], gmpy2.mpz(0))

    

    def dump_sv(self, builder: sv.ModuleSV): 
        """
        always_comb begin   
            if(active) begin 
                case(sel)  
                    xxx,xxx: xxx=1; 
                    default: xxx=1;
                endcase
            end 
            else begin 
                xxx = 0; 
                xxx = 0;
            end 
        end
        """  
        return ''
        active = '1' if self.active_signal is None else builder.get_name(self.active_signal)
        sel = builder.get_name(self.sel_signal)
        items_str = []
        for item, target in self.items:
            if item is None:
                items_str.append(('default', builder.get_name(target)))
            elif isinstance(item, tuple):
                items_str.append((','.join(builder.get_name(i) for i in item), builder.get_name(target)))
            else:
                items_str.append(( builder.get_name(item), builder.get_name(target)))
        case_body = '\n'.join(f'{item}:{target}=1;' for item, target in items_str) 
        else_branch = '\n'.join(f'{target}=0;' for _, target in items_str) 

        case_body = '      '.join(case_body.splitlines(keepends=True))
        else_branch = '    '.join(else_branch.splitlines(keepends=True))
        builder.Block(self, 
f"""
always_comb begin   
  if({active}) begin 
    case({sel})  
      {case_body}
    endcase
  end 
  else begin 
    {else_branch}
  end 
end""" )




class CondStack:
    """ stack of when/switch frames; stored in local scope of functions
    """
    def __init__(self):
        self.stacks: List[CaseGate] = []   # grow for nested 'when', 'switch'
        self.prev_stack: CaseGate = None     # exit 'elsewhen' and 'otherwise' stmt
        self.active_signals: List[Optional[hgl_core.Reader]] = [None] 

    def new(self, sel: Union[hgl_core.Reader, Tuple]):
        """ new 'when' stmt or 'switch' stmt 
        """
        self.stacks.append(CaseGate(sel, self.active_signals[-1])) 

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
    def tail(self) -> CaseGate:
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
        self.condframe.new(Signal(1))               # new `switch(1) ...` 
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




def MuxSwitch(sel: Any, *args):
    """
    MuxSwitch(1,
        (cond1, [a,b]),
        (cond2, [b,c]),
        (cond3, [c,d]),
        (...,   [1,2])
    ) 
    """
    assert args 
    case_item, default = args[-1]
    assert case_item is ..., f'no default value {args}'
    default = ToArray(default) 
    ret = WireNext(default)
    with __hgl_switch__(sel):
        for item, source in args[:-1]:
            if not isinstance(item, tuple):
                item = (item,)
            with __hgl_once__(item):
                __hgl_partial_assign__(ret, source)
    return ret

def MuxWhen(*args):
    return MuxSwitch(1, *args)


