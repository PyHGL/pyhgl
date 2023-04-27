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
from typing import Any, Generator, List, Tuple, Set, Literal

import inspect
import bisect

from pyhgl.array import *
from pyhgl.array.functions import HGLPattern
import pyhgl.logic.hgl_core as hgl_core
import pyhgl.logic._session as _session  
import pyhgl.logic._config as config
import pyhgl.tester.runner as tester_runner


assert_dispatcher = Dispatcher()



class Pattern(HGLPattern):
    
    sess: _session.Session
    
    def f(self, t: int):
        """ coroutine based SVA
        Args:
            t: verification time (may behind the real simulation time)
        yield:
            (0, n):      wait simulator, until n step passed 
            (0, signal): wait simulator, until signal changed 
            (1, next_t): accept and goto next verification time 
            StopIter:    stop generation without success
        """
        yield (1, t)
        
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        self.sess = HGL._sess 
        return self
        
    def __pow__(self, n: Union[int, tuple, list, set]):
        """
        patt ** n: repeat n times 
        patt ** [m,n]: repeat m to n times 
        patt ** {1,3,5}: repeat 1 or 3 or 5 times
        """
        if isinstance(n, int):
            return _Repeat(self, n,n)
        elif isinstance(n, (tuple, list)): 
            assert len(n) == 2, 'patt ** [m,n]'
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
    elif isinstance(x, hgl_core.Reader):    # assert x
        return _Atom(x) 
    elif isinstance(x, (int, tuple, list, set)): 
        # n     : wait n clock edges
        # [m,n] : wait m to n clock edges 
        # {2,4} : wait 2 or 4 clock edges
        clk, edge = config.conf.clock 
        if edge:
            patt = Rise(clk)
        else:
            patt=  Fall(clk)
        return patt ** x 
    elif inspect.isfunction(x):  # assert f
        # lambda : True
        return _Atom(x)
    else:
        raise TypeError(x, 'invalid Assertion pattern')



@dispatch('Sequence', Any, [Any,None], dispatcher=assert_dispatcher)
def _sequence(*args):
    if len(args) == 2:
        return _Sequence(_to_pattern(args[0]), _to_pattern(args[1])) 
    else:
        return _to_pattern(args[0])


@dispatch('Imply', Any, Any, dispatcher=assert_dispatcher)
def _imply(a, b):
    return _Imply(_to_pattern(a), _to_pattern(b))
        
@dispatch('Signal', Any, dispatcher=assert_dispatcher)
def _signal(obj):
    return obj
        
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

    
x >>> Rise(clk1)**[2,3] >>> Fall(clk2)**[1,3] >>> y |-> (Rise(clk3) >>> z)**[2,4]

"""


class _Atom(Pattern):
    """ _Atom(Signal|Callable)
    """
    def __init__(self, s: Union[hgl_core.Reader, Callable]):
        if isinstance(s, hgl_core.Reader): 
            if s._data.tracker is None:
                self.sess.track(s)
        else:
            assert inspect.isfunction(s), f'invalid type {s}'
        self.s = s 
    
    def f(self, t:int):
        if isinstance(self.s, hgl_core.Reader):
            logic_value = self.s._data.tracker.history(t)
            if logic_value.v > 0 and logic_value.x == 0:
                yield (1, t)
        else:
            if self.s():
                yield (1, t)
        
        
        
# class _Cache(Pattern):
    
#     def __init__(self, x: Dict[str, Any]):
#         self._items: Dict[str, Union[hgl_core.Reader, Callable]] = {}
#         for k, v in x.items():
#             if not isinstance(k, str):
#                 raise TypeError('name is not str')
#             self._items[k] = v 
    
#     def f(self, t:int, cache: Dict[str, list]):
#         for k, v in self._items.items():
#             if k not in cache:
#                 cache[k] = []
#             if inspect.isfunction(v):
#                 cache[k].append(v())
#             else:
#                 cache[k].append(hgl_core.getv(v))
#         yield (1, t, cache)
        
        
        
@dispatch('LogicOr', Any, Any, dispatcher=assert_dispatcher)
class _Alter(Pattern):
    """ patt1 || patt2 || patt3
    """
    def __init__(self, *patts: Pattern):
        self.patts = patts 
        
    def f(self, t:int):
        for patt in self.patts:
            for next_t in patt.f(t):  # if one of patts reutrn True, then True
                yield next_t 


@dispatch('LogicAnd', Any, Any, dispatcher=assert_dispatcher)
def _logicand(a, b):
    return _Sequence(a, b)
        

class _Sequence(Pattern):
    """  patt1 >>> patt2
    """
    def __init__(self, patt1: Pattern, patt2: Pattern):
        """ success only if both success
        """
        self.patt1 = patt1 
        self.patt2 = patt2 
        
    def f(self, t: int):
        for next_t in self.patt1.f(t):
            if next_t[0] == 0:      # wait simulator
                yield next_t 
            else:       # patt1 return success
                for next_next_t in self.patt2.f(next_t[1]):  # start patt2
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
        
    def repeat_n(self, t: int, n: int):
        """ use dfs
        """
        # match nothing, default success
        if n == 0:   
            yield (1, t)
        # repect 1
        for next_t in self.patt.f(t):
            # wait simulator
            if next_t[0] == 0:
                yield next_t 
            # full match
            else:
                # repeat n-1
                for next_next_t in self.repeat_n(next_t[1], n-1):
                    yield next_next_t

    def f(self, t:int):
        """ use bfs 
        """
        cache_t = []
        # go to n1, there may be multiple success
        for next_t in self.repeat_n(t, self.n1):
            if next_t[0] == 0:
                yield next_t
            else:
                cache_t.append(next_t)
                yield next_t 
        for _ in range(self.n1, self.n2):
            new_t = []
            for next_t in cache_t: 
                for next_next_t in self.patt.f(next_t[1]):
                    # wait simulator
                    if next_next_t[0] == 0:
                        yield next_next_t 
                    # success n1 + 1
                    else:
                        new_t.append(next_next_t)
                        yield next_next_t 
            # checkpoint for n1 + i 
            cache_t = new_t
        # if one of n1 ~ n2 success, than success


@dispatch('LogicNot', Any, dispatcher=assert_dispatcher)
class _Assert_Not(Pattern):
    def __init__(self, patt: Pattern):
        self.patt = patt 
        
    def f(self, t:int):
        # if matches, return None
        for next_t in self.patt.f(t):
            if next_t[0] == 0:
                yield next_t 
            else:
                return
        yield (1, t) 
        

class _Imply(Pattern):
    """
    a |-> b  same as  !a || a >>> b
    """
    
    def __init__(self, patt1: Pattern, patt2: Pattern):
        self.patt1 = patt1 
        self.patt2 = patt2 
        
    def f(self, t:int):
        patt1_matched = False
        # return all successes in patt1 >>> patt2
        for next_t in self.patt1.f(t):
            if next_t[0] == 0:
                yield next_t 
            else:
                patt1_matched = True
                for next_next_t in self.patt2.f(next_t[1]):
                    yield next_next_t 
        # !patt1 success, also success
        if not patt1_matched:
            yield (1, t) 


class Delay(Pattern):
    
    def __init__(self, step: int):
        self.step = step 
        
    def f(self, t:int):
        # next verification time
        next_t = t + self.step
        # XXX wait extra 1 time step, since signal history may not avaliable
        diff = next_t - self.sess.sim_py.t + 1
        if diff:
            yield (0, diff)
        yield (1, next_t)
        

class Until(Pattern):
    """ go to the time when signal is desired value
    """
    def __init__(self, s: hgl_core.Reader, v: hgl_core.Logic):
        assert isinstance(s, hgl_core.Reader) 
        if s._data.tracker is None:
            self.sess.track(s)
        self.s = s 
        self.v = hgl_core.Logic(v)
    
    def f(self, t: int):
        s = self.s 
        tracker = s._data.tracker
        idx = tracker.history_idx(t) 
        # no wait
        if tracker.values[idx] == self.v:
            yield (1, t) 
            return 
        # find in history
        for next_idx in range(idx+1, len(tracker.values)):
            if tracker.values[next_idx] == self.v:
                yield (1, tracker.timestamps[next_idx])
                return  
        # wait simulator for newest value
        while 1: 
            yield (0, s) 
            yield (0, 1)
            if tracker.values[-1] == self.v: 
                yield (1, tracker.timestamps[-1]) 
                return 

def Rise(s: hgl_core.Reader): 
    """ wait until signal from 0 to 1, unknown-state ignored
    """
    assert isinstance(s, hgl_core.Reader) and len(s) == 1, f'{s} not 1-bit signal'
    return _Sequence(
        Until(s, 0),
        Until(s, 1)
    )

def Fall(s: hgl_core.Reader): 
    """ wait until signal from 1 to 0, unknown-state ignored
    """
    assert isinstance(s, hgl_core.Reader) and len(s) == 1, f'{s} not 1-bit signal'
    return _Sequence(
        Until(s, 1),
        Until(s, 0)
    )

class Change(Pattern):
    """ wait until signal changes, including unknown-state
    """
    def __init__(self, s: hgl_core.Reader):
        assert isinstance(s, hgl_core.Reader) 
        if s._data.tracker is None:
            self.sess.track(s)
        self.s = s 
    
    def f(self, t: int):
        s = self.s 
        tracker = s._data.tracker
        idx = tracker.history_idx(t) 
        curr_v = tracker.values[idx]
        # find in history
        for next_idx in range(idx+1, len(tracker.values)):
            if tracker.values[next_idx] != curr_v:
                yield (1, tracker.timestamps[next_idx])
                return  
        # wait simulator for newest value
        while 1: 
            yield (0, s) 
            yield (0, 1)
            if tracker.values[-1] != curr_v: 
                yield (1, tracker.timestamps[-1]) 
                return 

def Print(format: str, *args: Any) -> _Atom:
    def _print():
        print(format.format(*args))
        return True
    return _Atom(_print)


# -------------
# functions 
# -------------
@dispatch('Eq', Any, Any, dispatcher=assert_dispatcher)
class _Eq(Pattern):
    def __init__(self, a: Any, b: Any):
        self.a = a 
        self.b = b
        
    def f(self, t:int):
        # if matches, return None
        if isinstance(self.a, hgl_core.Reader):
            a = self.a._data._getval_py()
        else:
            a = self.a 
        if isinstance(self.b, hgl_core.Reader):
            b = self.b._data._getval_py()
        else:
            b = self.b 
        if a == b:
            yield (1,t)

@dispatch('Ne', Any, Any, dispatcher=assert_dispatcher)
class _Ne(Pattern):
    def __init__(self, a: Any, b: Any):
        self.a = a 
        self.b = b
        
    def f(self, t:int):
        # if matches, return None
        if isinstance(self.a, hgl_core.Reader):
            a = self.a._data._getval_py()
        else:
            a = self.a 
        if isinstance(self.b, hgl_core.Reader):
            b = self.b._data._getval_py()
        else:
            b = self.b 
        if a != b:
            yield (1,t)

# class posedge(Pattern):
    
#     def __init__(self, s: hgl_core.Reader):
        
#         if not isinstance(s, hgl_core.Reader) or len(s) != 1:
#             raise ValueError(f'input should be 1 bit signal')
        
#         s._track()
#         self.signal = s 
        
#     def f(self, t: int, cache: Dict[str, list]):
#         s = self.signal
#         next_idx = self._get_next_idx(s, t)
#         if (x:=(next_idx + 1 - len(s._timestamps))) > 0:
#             yield (0, s, x)
#         yield (1, s._timestamps[next_idx], cache)
            
            
#     def _get_next_idx(self, s: hgl_core.Reader, t: int) -> int:
#         idx = bisect.bisect(s._timestamps, t) - 1 
#         if idx < 0:
#             idx = 0 
#         if s._values[idx] == 0:
#             return idx + 1 
#         else:
#             return idx + 2   
        
#     def __iter__(self):
#         if self.signal._getval_py() == 0:
#             yield self.signal 
#         else:
#             yield self.signal
#             yield self.signal
    
        
# class dualedge(posedge):
    
#     def _get_next_idx(self, s: hgl_core.Reader, t: int) -> int:
#         idx = bisect.bisect(s._timestamps, t) - 1 
#         if idx < 0:
#             idx = 0 
#         return idx + 1

#     def __iter__(self):
#         yield self.signal

@dispatch('Assert', Any, [Any, None], dispatcher=assert_dispatcher)
class _Assert(HGL):
    
    _sess: _session.Session

    def __init__(self, pattern: Pattern, *, N: int = None):
        """
        pattern is triggered at clock edge N time 
        """ 
        self.sess: _session.Session = self._sess

        self.trigger: Tuple[hgl_core.Reader, int] = config.conf.clock 
            
        self.disable: Optional[Tuple[hgl_core.Reader, int]] = config.conf.reset 
            
        self.pattern = _to_pattern(pattern)
                
        self.result: Dict[int, Any] = {}

        frame,filename,line_number,function_name,lines,index = inspect.stack()[1]
        test_key = [self.sess.filename]
        test_key.extend(i._unique_name for i in self.sess.module._position)
        test_key.append(f'line:{line_number}') 
        self.test_node = tester_runner._root.get_or_register(
            [test_key[0], '.'.join(test_key[1:])]
        )
        
        self.sess.sim_py.insert_coroutine_event(0, self.top_task(N)) 


    def top_task(self, N: int = None):
        n = 0 
        while (N is None) or n < N:
            # wait trigger
            signal, value = self.trigger 
            value = 1 if value > 0 else 0
            value_n = 1 - value 
            while signal._data._getval_py() != value_n:
                yield signal 
            while signal._data._getval_py() != value:
                yield signal 
            # check disable
            if self.disable is None or self.disable[0]._data._getval_py() != self.disable[1]:
                self.sess.sim_py.insert_coroutine_event(0, self.sub_task(n))
            n += 1

    def sub_task(self, n: int):
        """ n: n-th trigger
        """
        start_t = self.sess.sim_py.t
        end_t = None 
        self.result[start_t] = 'running'
        
        for next_t in self.pattern.f(start_t):
            # wait simulator
            if next_t[0] == 0:
                yield next_t[1]
            # first seccess
            else:
                end_t = next_t[1]
                break 

        if end_t is None: 
            self.result[start_t] = 'fail'
            self.test_node.count_failed += 1
        else:
            self.result[start_t] = f'pass in {end_t-start_t}'
            self.test_node.count_passed += 1


    def __str__(self):
        return ''.join(f'\n{t}:{s}' for t, s in self.result.items())




class AssertCtrl(HGL):
    
    _sess: _session.Session
    """
    with AssertCtrl() as :
        # def f():
        #     print(f'state: {state} t:{conf.t}')
        #     return True
        Assert(Rise(io.init) |-> 1 >>> state == 1)
    """
    def __init__(
        self, 
        trigger: Tuple[hgl_core.Reader, int] = ..., 
        disable: Tuple[hgl_core.Reader, int] = ...
    ): 
        """
        trigger: default is clock 
        disable: default is reset
        """
        if trigger is ...:
            trigger = config.conf.clock 
        if disable is ...:
            disable = config.conf.reset 
            
        clk, edge = trigger 
        assert isinstance(clk, hgl_core.Reader) and isinstance(edge, (int, bool))
        assert disable is None or isinstance(disable[0], hgl_core.Reader)
        self.trigger = trigger
        self.disable = disable
        
        self._clk_restore = []
        self._rst_restore = [] 
        self._dispatcher_restore = [] 

    def edge(self, signal: hgl_core.Reader, value: Literal[0,1]):
        if value == 0:
            return Fall(signal)
        else:
            return Rise(signal)
    
    def until(self, signal: hgl_core.Reader, value: Any):
        return Until(signal, value)
        
    def __enter__(self):
        self._clk_restore.append(config.conf.clock)
        self._rst_restore.append(config.conf.reset) 
        self._dispatcher_restore.append(config.conf.dispatcher)
        self._sess.module._conf.clock = self.trigger 
        self._sess.module._conf.reset = self.disable
        self._sess.module._conf.dispatcher = assert_dispatcher 
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        config.conf.reset = self._clk_restore.pop()
        config.conf.clock = self._rst_restore.pop()
        config.conf.dispatcher = self._dispatcher_restore.pop()

