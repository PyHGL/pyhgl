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
from typing import Any, Generator, List, Tuple, Set

import inspect
import bisect

from pyhgl.array import *
from pyhgl.array.functions import HGLPattern
import pyhgl.logic.hgl_basic as hgl_basic
import pyhgl.logic._session as _session 


assert_dispatcher = Dispatcher()

class StrData(hgl_basic.SignalData):
    
    __slots__ = 'v'
    
    def __init__(self, v: str = ''):
        super().__init__()
        self.v = v 
    
    def copy(self) -> StrData:
        return StrData(self.v) 
    
    def __len__(self):
        return 0 
    
    def __str__(self):
        return self.v 
    
    def _setval_py(self, v, key = None) -> bool:
        if self.v == v:
            return False 
        else:
            self.v = v 
            return True  
        
@singleton  
class StrType(hgl_basic.SignalType):
    
    __slots__ = ()
    
    def __len__(self):
        return 0 
    
    def __str__(self):
        return 'StrSignal'
    
    def _getval_py(self, data: StrData, key = None) -> str:
        return data.v 
    
    def _eval(self, v: str) -> str:
        return str(v) 
    
    def __call__(self, v: Union[int, str] = '', *, name: str='temp_str') -> hgl_basic.Reader:
        return hgl_basic.Reader(data=StrData(self._eval(v)), type=self, name=name)

    def show(self: hgl_basic.Reader) -> str:
        return '\n'.join(f't={t:>10}  {v}' for t, v in zip(self._timestamps, self._values))


class Pattern(HGLPattern):
    
    sess: _session.Session
    
    def f(self, t: int, cache: Dict[str, list]):
        """
        Args:
            t: verification time 
            cache: cached values, ex. {'a':[0,0,1]}
        yield:
            (0, None, n): wait until simulation time n 
            (0, signal, n): wait for signal changed n times 
            (1, next_t, cache): accept and goto next verification time 
            None: not accept, assertion failed
        """
        yield (1, t, cache) 
        
        
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        self.sess = HGL._sess 
        return self
        
    def __pow__(self, n: Union[int, tuple, list, set]):
        if isinstance(n, int):
            return _Repeat(self, n,n)
        elif isinstance(n, (tuple, list)):
            return _Repeat(self, *n) 
        elif isinstance(n, set):
            return _Alter(self, *(_Repeat(self, i, i) for i in n))
        else:
            raise ValueError(n) 



def _to_pattern(x) -> Pattern:
    """
    convert x to Pattern
    """
    if isinstance(x, Pattern):
        return x 
    elif isinstance(x, hgl_basic.Reader):
        return _Atom(x) 
    elif isinstance(x, (int, tuple, list, set)):
        clk, edge = HGL._sess.module.clock
        if edge == 0:
            clock = negedge(clk)
        else:
            clock = posedge(clk)
        return clock ** x 
    elif isinstance(x, dict):
        return _Cache(x)
    elif inspect.isfunction(x):
        return _Atom(x)
    else:
        raise TypeError(x)


@singleton 
class __hgl_rshift__(HGL):
    """ PyHGL operator >>>
    """
    def __call__(self, *args):
        if len(args) == 2:
            return _Sequence(_to_pattern(args[0]), _to_pattern(args[1])) 
        else:
            return _to_pattern(args[0])

        
        
@singleton 
class __hgl_imply__(HGL):
    """ PyHGL operator |->
    """
    def __call__(self, a, b):
        return  _Imply(_to_pattern(a), _to_pattern(b))

        
"""
    posedge(clk1)(0):
         _____       _____
        |     |_____|
        ^here       
          ____
        _|
         ^here
        
    posedge(clk1)(0):
         _____       _____
        |     |_____|
                    ^here       
          ____
        _|
         ^here

    
x >>> posedge(clk1)**(2,3) >>> negedge(clk2)**(1,3) >>> y |-> (posedge(clk3) >>> z)**(2,4)

    
Assert(
    trigger = (clk, 1),
    disable = (rst, None),
    pattern = x >>> posedge(clk)**[1,2] >>> negedge(clk) >>> y |-> delay(-1) >>> dualedge(clk) >>> a & b & c
)


pattern = cache(x,y)**3 |-> 2 >>> cache(z) >>> lambda x: x[2] == x[0] + x[1] 

def f(t, cache):
    cache[x].append(...)
    
{'x': io.x, 'y': io.y} >>> 1 >>> cache(x=io.x) |-> 1 >>> lambda _:_.x[1] == _.x[0] + _.y[0]


"""


class _Atom(Pattern):
    """ _Atom(Signal|Callable)
    """
    def __init__(self, s: Union[hgl_basic.Reader, Callable[[dict],bool]]):
        if isinstance(s, hgl_basic.Reader):
            s._track() 
        else:
            if not inspect.isfunction(s):
                raise TypeError('invalid target')
        self.s = s 
    
    def f(self, t:int, cache: Dict[str, list]):
        if isinstance(self.s, hgl_basic.Reader):
            if self.s._history(t) > 0:
                yield (1, t, cache) 
        else:
            if self.s(cache):
                yield (1, t, cache)
        
        
        
class _Cache(Pattern):
    
    def __init__(self, x: Dict[str, Any]):
        self._items: Dict[str, Union[hgl_basic.Reader, Callable]] = {}
        for k, v in x.items():
            if not isinstance(k, str):
                raise TypeError('name is not str')
            self._items[k] = v 
    
    def f(self, t:int, cache: Dict[str, list]):
        for k, v in self._items.items():
            if k not in cache:
                cache[k] = []
            if inspect.isfunction(v):
                cache[k].append(v())
            else:
                cache[k].append(hgl_basic.getv(v))
        yield (1, t, cache)
        
        
        
@dispatch('Assert_Or', Any, Any, dispatcher=assert_dispatcher)
class _Alter(Pattern):
    
    def __init__(self, *patts: Pattern):
        self.patts = patts 
        
    def f(self, t:int, cache: Dict[str, list]):
        for patt in self.patts:
            # copy cache and yield 
            for next_t in patt.f(t, {k:v[:] for k,v in cache.items()}):
                yield next_t
        

class _Sequence(Pattern):
    
    def __init__(self, patt1: Pattern, patt2: Pattern):
        self.patt1 = patt1 
        self.patt2 = patt2 
        
    def f(self, t: int, cache: Dict[str, list]):
        # positions that patt1 match
        for next_t in self.patt1.f(t, cache):
            if next_t[0] == 0:
                yield next_t 
            else:
                # positions that patt2 match
                for next_next_t in self.patt2.f(next_t[1], next_t[2]):
                    yield next_next_t 
                                 

        
class _Repeat(Pattern):
    
    def __init__(self, patt: Pattern, n1: int, n2: int):
        """
        pattern repeat n1 | n1+1 | n1+2 | ... | n2 times
        """
        if n1 < 0 or n2 < n1:
            raise ValueError(f'cannot repeat negative times')
        
        self.n1 = n1 
        self.n2 = n2
        self.patt = patt
        
    def repeat_n(self, t: int, cache: Dict[str, list], n: int):
        """ dfs
        """
        if n == 0:
            yield (1, t, cache)
        # repect 1
        for next_t in self.patt.f(t, cache):
            # wait simulator
            if next_t[0] == 0:
                yield next_t 
            # full match
            else:
                for next_next_t in self.repeat_n(next_t[1], next_t[2], n-1):
                    yield next_next_t

    def f(self, t:int, cache: Dict[str, list]):
        """ bfs 
        """
        cache_t = []
        # go to n1
        for next_t in self.repeat_n(t, cache, self.n1):
            if next_t[0] == 0:
                yield next_t
            else:
                cache_t.append(next_t)
                yield next_t 
        for _ in range(self.n1, self.n2):
            new_t = []
            for next_t in cache_t: 
                for next_next_t in self.patt.f(next_t[1], next_t[2]):
                    if next_next_t[0] == 0:
                        yield next_next_t 
                    else:
                        new_t.append(next_next_t)
                        yield next_next_t  
            cache_t = new_t



        
@dispatch('Assert_Not', Any, dispatcher=assert_dispatcher)
class _Assert_Not(Pattern):
    def __init__(self, patt: Pattern):
        self.patt = patt 
        
    def f(self, t:int, cache: Dict[str, list]):
        # if matches, return None
        for next_t in self.patt.f(t, {k:v[:] for k,v in cache.items()}):
            if next_t[0] == 0:
                yield next_t 
            else:
                return
        yield (1, t, cache) 
        

class _Imply(Pattern):
    """
    a |-> b  same as  !a || a >>> b
    """
    
    def __init__(self, patt1: Pattern, patt2: Pattern):
        self.patt1 = patt1 
        self.patt2 = patt2 
        
    def f(self, t:int, cache: Dict[str, list]):
        patt1_matched = False
        for next_t in self.patt1.f(t, {k:v[:] for k,v in cache.items()}):
            if next_t[0] == 0:
                yield next_t 
            else:
                patt1_matched = True
                for next_next_t in self.patt2.f(next_t[1], next_t[2]):
                    yield next_next_t 
        if not patt1_matched:
            yield (1, t, cache) 


class delay(Pattern):
    
    def __init__(self, step: int):
        self.step = step 
        
    def f(self, t:int, cache: Dict[str, list]):
        next_t = t + self.step
        # wait simulator
        if (x:=(next_t - self.sess.sim_py.t)) > 0:
            yield (0, None, x)
        yield (1, next_t, cache)
        
        
class until(Pattern):
    """ go to the time when signal is desired value
    """
    def __init__(self, s: hgl_basic.Reader, v: Any):
        assert isinstance(s, hgl_basic.Reader)
        s._track()
        self.s = s 
        self.v = v 
    
    def f(self, t: int, cache: Dict[str, list]):
        s = self.s
        idx = bisect.bisect(s._timestamps, t) - 1 
        if idx < 0:
            idx = 0 
        if s._values[idx] == self.v:
            yield (1, t, cache) 
            return 
        for next_idx in range(idx+1, len(s._values)):
            if s._values[next_idx] == self.v:
                yield (1, s._timestamps[next_idx], cache)
                return  
        while 1: 
            yield (0, s, 1) 
            if s._values[-1] == self.v: 
                yield (1, s._timestamps[-1], cache) 
                return 

    def __iter__(self):
        if self.s._getval_py() == self.v: 
            return 
        while 1: 
            yield self.s 
            if self.s._getval_py() == self.v:
                return 


class posedge(Pattern):
    
    def __init__(self, s: hgl_basic.Reader):
        
        if not isinstance(s, hgl_basic.Reader) or len(s) != 1:
            raise ValueError(f'input should be 1 bit signal')
        
        s._track()
        self.signal = s 
        
    def f(self, t: int, cache: Dict[str, list]):
        s = self.signal
        next_idx = self._get_next_idx(s, t)
        if (x:=(next_idx + 1 - len(s._timestamps))) > 0:
            yield (0, s, x)
        yield (1, s._timestamps[next_idx], cache)
            
            
    def _get_next_idx(self, s: hgl_basic.Reader, t: int) -> int:
        idx = bisect.bisect(s._timestamps, t) - 1 
        if idx < 0:
            idx = 0 
        if s._values[idx] == 0:
            return idx + 1 
        else:
            return idx + 2   
        
    def __iter__(self):
        if self.signal._getval_py() == 0:
            yield self.signal 
        else:
            yield self.signal
            yield self.signal
    
    
class negedge(posedge):
    
    def _get_next_idx(self, s: hgl_basic.Reader, t: int) -> int:
        idx = bisect.bisect(s._timestamps, t) - 1 
        if idx < 0:
            idx = 0 
        if s._values[idx] == 0:
            return idx + 2
        else:
            return idx + 1  
        
    def __iter__(self):
        if self.signal._getval_py() == 0:
            yield self.signal 
            yield self.signal
        else:
            yield self.signal
        
        
class dualedge(posedge):
    
    def _get_next_idx(self, s: hgl_basic.Reader, t: int) -> int:
        idx = bisect.bisect(s._timestamps, t) - 1 
        if idx < 0:
            idx = 0 
        return idx + 1

    def __iter__(self):
        yield self.signal




class Assert(HGL):
    
    _sess: _session.Session

    def __init__(self, pattern: Pattern, *, N: int = None):
        """
        N: trigger n times
        """ 
        self.sess = self._sess

        clk, edge = self.sess.module.clock
        if edge:
            self.trigger = posedge(clk)
        else:
            self.trigger = negedge(clk)
            
        self.disable: Tuple[hgl_basic.Reader, int] = self.sess.module.reset
            
        self.pattern = _to_pattern(pattern)
                
        self.result: Dict[int, Any] = {}
        
        frame,filename,line_number,function_name,lines,index = inspect.stack()[1]
        self.msg = f'{filename}:{line_number}'
        
        self.sess.sim_py.insert_coroutine_event(0, self.assert_task(N)) 


    def assert_task(self, N: int = None):
        n = 0 
        while (N is None) or n < N:
            # trigger
            yield from self.trigger 
            # disable
            if self.disable is None or self.disable[0]._getval_py() != self.disable[1]:
                self.sess.sim_py.insert_coroutine_event(0, self.assert_event(n))
            n += 1

    def assert_event(self, n: int):
        """ n: n-th trigger
        """
        start_t = self.sess.sim_py.t
        end_t = None 
        self.result[start_t] = 'running'
        
        for next_t in self.pattern.f(start_t, {}):
            if next_t[0] == 0:
                if next_t[1] is None:
                    yield next_t[2]
                else:
                    for _ in range(next_t[2]):
                        yield next_t[1]
            else:
                end_t = next_t[1]
                break 

        if end_t is None: 
            self.result[start_t] = 'fail'
        else:
            self.result[start_t] = f'pass in {end_t-start_t}'


    def __str__(self):
        return self.msg + ''.join(f'\n{t}:{s}' for t, s in self.result.items())




class AssertCtrl(HGL):
    
    _sess: _session.Session
    
    def __init__(
        self, 
        trigger: Tuple[hgl_basic.Reader, int] = ..., 
        disable: Tuple[hgl_basic.Reader, int] = ...
    ): 
        """
        trigger: default is clock 
        disable: default is reset
        """
        if trigger is ...:
            trigger = self._sess.module.clock 
        if disable is ...:
            disable = self._sess.module.reset 
            
        clk, edge = trigger 
        assert isinstance(clk, hgl_basic.Reader) and isinstance(edge, (int, bool))
        self.trigger = trigger
        assert disable is None or isinstance(disable[0], hgl_basic.Reader)
        self.disable = disable
        
        self._clk_restore = []
        self._rst_restore = [] 
        self._dispatcher_restore = []
        
    def __enter__(self):
        self._clk_restore.append(self._sess.module.clock)
        self._rst_restore.append(self._sess.module.reset) 
        self._dispatcher_restore.append(self._sess.module.dispatcher)
        self._sess.module.clock = self.trigger 
        self._sess.module.reset = self.disable
        self._sess.module.dispatcher = assert_dispatcher

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._sess.module.reset = self._clk_restore.pop()
        self._sess.module.clock = self._rst_restore.pop()
        self._sess.module.dispatcher = self._dispatcher_restore.pop()

