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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import re 
import inspect
import builtins  

from ._hgl import HGL



class Dispatcher:
    """Dynamic dispatch unary and binary/multi functions

    1. support boardcast
    2. easy access
    3. *args will match binary functions 
    4. cache 
    """
    
    def __init__(self):
        # 'Add': [(f, (UInt, SInt), (UInt, SInt, str, int))]
        self._table: Dict[str,List[
            Tuple[Callable, Tuple[type, ...], Tuple[type, ...]]
        ]] = {}
        self._cache = {}

    @property
    def copy(self) -> Dispatcher:
        ret = self.__class__()
        ret._table.update(self._table)  
        return ret 
    
    def update(self, other: Dispatcher):
        """ other prior to self
        """
        assert isinstance(other, Dispatcher)
        for f_name, fs in other._table:
            if f_name not in self._table:
                self._table[f_name] = []
            self._table[f_name] = fs + self._table[f_name]
        
    def dispatch(self, 
                 f_name:str, 
                 f: Callable, 
                 arg1: Union[List[type], Tuple[type], type], 
                 arg2: Union[List[type], Tuple[type], type, None] = None):
        """dispatch unary/binary functions

        Ex.: dispatch('Add', f, [UInt, SInt], [UInt, SInt, None])
             dispatch('Add', f, [UInt, SInt], Any )

        None means unary function
        Any means match all (except None)

        list means Union
        """
        assert callable(f)
        if f_name not in self._table:
            self._table[f_name] = []

        self._cache.clear()
        
        if not isinstance(arg1, (tuple, list)):
            arg1 = (arg1,)
        if not isinstance(arg2, (tuple, list)):
            arg2 = (arg2,) 

        for i in arg1: 
            assert i is Any or isinstance(i, type), f'{i} is not type'
        for i in arg2: 
            assert i is None or i is Any or isinstance(i, type), f'{i} is not type'
                
        # (f, (UInt, SInt), (UInt, SInt))
        self._table[f_name].insert(0, (f, tuple(arg1), tuple(arg2)))

    def type(self, obj): 
        if isinstance(obj, HGL):
            return obj.__hgl_type__
        else:
            return type(obj)

    def call(self, f_name:str, *args, **kwargs):
        """ ex. call('Add', 1, 1.2, type='fulladder')
        """
        f = None
        if len(args) == 1:
            f = self.find(f_name, self.type(args[0]), None)
        elif len(args) > 1:
            f = self.find(f_name, self.type(args[0]), self.type(args[1]))

        if f is not None:
            #---------------------------
            # if self._sess.verbose_dispatch:
            #     types = ', '.join([arg.__class__.__name__ for arg in args])
            #     self._sess.print(f"@dispatcher_call {f_name}({types}) --> {self._func_name(f)}")
            #---------------------------
            return f(*args, **kwargs)
        else:
            types = ', '.join([self.type(arg).__name__ for arg in args])
            raise TypeError(f"{f_name}({types}) can not found")


    def find(self, f_name: str, t1: type, t2: type) -> Optional[Callable]:
        """
        find a registered function by type of arguments
        return None if not find

        t1: type 
        t2: type or None

        ex. find('Add', int, str)
        """
        ret = self._cache.get((f_name, t1, t2))
        if ret is not None:
            return ret 
        
        else:
            if f_name not in self._table:
                return None
            for stored_f, stored_t1, stored_t2 in self._table[f_name]:
                
                if (Any in stored_t1 or t1 in stored_t1) and (
                    (t2 is not None and Any in stored_t2) or (t2 in stored_t2)): 
                    self._cache[(f_name, t1, t2)] = stored_f 
                    return stored_f

            return None
        

    def _show_types(self, t1: tuple, t2: tuple) -> str:
        t1 = ['Any' if i is Any else 'None' if i is None else i.__name__ for i in t1] 
        t2 = ['Any' if i is Any else 'None' if i is None else i.__name__ for i in t2] 
        return f"({'|'.join(t1)}, {'|'.join(t2)})"
        
        
    def _func_name(self, f: Callable) -> str:
        if not hasattr(f, '__qualname__'):
            f = type(f)
        module = f.__module__ 
        name = f.__qualname__ 
        if hasattr(f, '__hgl_wrapped__'):
            wrapped = getattr(f, '__hgl_wrapped__')
            return f"{f.__name__}({self._func_name(wrapped)})"
        else:
            ret =f"{module}.{name}"
            return re.sub(r'.<locals>','', ret)        
        
    def __str__(self):
        ret = ""
        for name, l in self._table.items():
            ret += f"{name}:\n"
            for f in l:
                ret += f"  {self._func_name(f[0])}{self._show_types(f[1],f[2])}\n"
        return ret



#------------------------ 
# decorators for dispatch
#------------------------ 

# default dispatcher
default_dispatcher = Dispatcher()

def dispatch(f_name: str, type1 = Any, type2 = None, *, dispatcher = None):
    """
        Ex.: @dispatch('Add', [UInt, SInt], [Any, None])
             @dispatch('Add', Any, None )
    """
    if dispatcher is None:
        dispatcher = default_dispatcher
    def decorator(f):
        dispatcher.dispatch(f_name, f, type1, type2)
        return f 
    return decorator 
    

    

def singleton(_class: type) -> object:
    """ return the instance of the input class
    """
    assert inspect.isclass(_class)
    name = _class.__name__ 
    obj = _class() 
    
    if name[:2] == '__' and name[-2:] == '__':
        setattr(builtins, name, obj)
    return obj