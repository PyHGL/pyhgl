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
from typing import Generator, List, Tuple, Set

import io 
import gmpy2 
import traceback

from sortedcontainers import SortedDict
from vcd import VCDWriter
from itertools import chain
from collections import defaultdict 

from pyhgl.array import *
import pyhgl.logic.hardware as hardware
import pyhgl.logic._session as _session 
import pyhgl.logic.hglmodule as hglmodule


class Simulator(HGL):
    
    __slots__ = 'sess', 'time_wheel', 'priority_queue', 't' 
    
    def __init__(self, sess: _session.Session, length: int = 8):
        """
        events are stored in a time_wheel, overflowed events are stored in a priority queue
        
        - signal event:  (SignalType, SignalData, key, value)
            - signal events are executed first 
            - one signal cannot update twice in one step
            - delay >= 1
            
        these events can read current signal values and insert other kinds of events
        - gate event: 
            - call when gate changes, or any input of gate changes
            - one gate connot update twice in one step
        - coroutine event: generator
            - yield delay: int|str|signal
            - signal time = simulator.t + 1
        - edge event: generator
            - call when triggered by yield signal
            - signal time = simulator.t + 1
        """
        self.sess = sess
        # length >= 2, signal_events, gate_events, coroutine_events
        self.time_wheel: List[
            Tuple[
                List[tuple],                # signal_events
                Dict[hardware.Gate, None],  # gate_events, remove duplicate triggered gates
                List[Generator]             # coroutine_events
            ]
        ] = [self.new_events() for _ in range(1 << length)]
        # Dict[int, Event]
        self.priority_queue = SortedDict()
        # current time, increase 1 after step()
        self.t: int = 0
        
    def new_events(self) -> Tuple[List[tuple], Dict[hardware.Gate, None], List[Generator]]:
        # signal_events, gate_events, coroutine_events
        return ([],{},[])
        
    def insert_signal_event(
        self, 
        dt: int, 
        event: Tuple[hardware.SignalType, hardware.SignalData, Optional[tuple], Any]
    ): 
        """
        - dt: > 0 if insert by gate, >= 0 if insert by coroutine event
        - event: (type, data, key, value)
          - key: None | tuple of (start, length)
          - value: immediate, str or gmpy2.mpz
        """ 
        try:
            self.time_wheel[dt][0].append(event) 
        except:
            t = self.t + dt
            if t not in self.priority_queue:
                self.priority_queue[t] = self.new_events()
            self.priority_queue[t][0].append(event)
        
    def insert_gate_event(self, dt: int, g: hardware.Gate):
        try:
            self.time_wheel[dt][1][g] = None
        except:
            t = self.t + dt
            if t not in self.priority_queue:
                self.priority_queue[t] = self.new_events()   
            self.priority_queue[t][1][g] = None 
            
    def insert_coroutine_event(self, dt: int, event: Generator):
        """ generator
        """
        try:
            self.time_wheel[dt][2].append(event)
        except:
            t = self.t + dt
            if t not in self.priority_queue:
                self.priority_queue[t] = self.new_events()   
            self.priority_queue[t][2].append(event)      
             
             
    def step(self, n: int = 1):
        """ step 1 timescale
        
        each signal should not updated more than once in 1 step
        each gate should not called more than once in 1 step
        """
        time_wheel = self.time_wheel 
        queue = self.priority_queue
        
        for next_t in range(self.t + 1, self.t + n + 1):                    # 100ms
            
            signal_events, gate_events, coroutine_events = time_wheel[0]    # 200ms 
            
            if signal_events or gate_events or coroutine_events:            # 200ms  
                # insert triggered coroutine events to next step
                coroutine_events_next = time_wheel[1][2]                    # 80ms
                #------------------------------------------
                # if self.sess.verbose_sim:
                #     _signal = ','.join(self._show_signal_event(i) for i in signal_events)
                #     self.sess.print(f't={self.t}, signals: {_signal}')
                #------------------------------------------
                
                # 1. exec user tasks
                for g in coroutine_events:                                  # 400ms clock
                    try:
                        ret = next(g)
                        if isinstance(ret, int):
                            self.insert_coroutine_event(ret, g)
                        elif isinstance(ret, str):
                            self.insert_coroutine_event(self.sess.timing._get_timestep(ret), g)
                        elif isinstance(ret, hardware.Reader): 
                            ret._events.append(g)
                        else:
                            self.sess.log.warning(f'{ret} is not valid trigger, stop task {g}')
                    except StopIteration:
                        pass   
                    except:
                        self.sess.log.warning(traceback.format_exc())
                coroutine_events.clear()  
                
                # 2. update signal values
                for e in signal_events:
                    
                    T:      hardware.SignalType
                    data:   hardware.SignalData 
                    key:    Optional[tuple] 
                    value:  Any 
                    
                    T, data, key, value = e
                    # cannot update signal twice 
                    if data._t == next_t:
                        self.sess.log.warning(f'update {data._name} twice at t={next_t}')
                    data._t = next_t  
                    # value changed 
                    if T._setimmd(data, key, value): 
                        for reader in data.reader: 
                            # changed gates 
                            gate_events.update(reader._driven)
                            # track values
                            if reader._timestamps:
                                reader._update()
                            # triggered events
                            if _e:=reader._events:
                                coroutine_events_next.extend(_e)
                                _e.clear()
                signal_events.clear() 
                
                #------------------------------------------
                # if self.sess.verbose_sim:
                #     _gate = ','.join(f'{i}' for i in gate_events)
                #     self.sess.print(f't={self.t}, gates: {_gate}')
                #------------------------------------------
                
                # 3. execute gate
                for g in gate_events: 
                    g.forward() 
                self.sess.log.n_exec_gates += len(gate_events)
                gate_events.clear() 
                
                
                if signal_events:
                    raise Exception('insert new events of zero-delay')
                    
            # TODO slow
            time_wheel.append(time_wheel.pop(0))        # 700ms
            # time_wheel.append(time_wheel.pop())         # 300 ms
            # time_wheel.insert(0, time_wheel.pop(0))   # 1700ms
            
            
            if queue:                                   # 80ms
                t0 = next_t + len(time_wheel) - 1
                if queue.keys()[0] == t0:
                    self.time_wheel[-1] = queue.pop(t0) 
            # update time
            self.t = next_t                             # 120ms
            
            
            
            
    def _show_signal_event(
        self, 
        e: Tuple[hardware.SignalType, hardware.SignalData, Optional[tuple], Any]
    ) -> str:
        _, data, key, value = e 
        name = data._name 
        if key is not None:
            name = f'{name}[{key[0]}+:{key[1]}]' 
        if isinstance(value, (int, gmpy2.mpz, gmpy2.xmpz)):
            _value = bin(value)
        else:
            _value = str(value)
        return f'{name} <- {_value}'
            
    def __str__(self):
        return 'simulator'

    
    

    
# ------------
# vcd writer 
# ------------ 


class VCD(HGL):
        
    def __init__(self, sess: _session.Session):
        
        self.sess = sess
        
        real_timescale = sess.timing.timescale or 1e-9
        self._timescale = f'{round(real_timescale/1e-9)} ns'
        self._signals: Dict[hardware.Reader, Tuple] = {}
    
    def _track(self, *args: hardware.Reader):
        new_signals = []
        for i in args:
            i = ToArray(i)
            if isinstance(i, Array):
                new_signals.extend(i._flat) 
            elif isinstance(i, hglmodule.Module):
                for _, v in i.__dict__.items():
                    if isinstance(v, hardware.Reader):
                        new_signals.append(v)
            elif isinstance(i, hardware.Reader):
                new_signals.append(i) 
        
        for v in new_signals:
            if isinstance(v, hardware.Reader) and v not in self._signals:
                v._track()
                # signal name
                name = v._name 
                # signal scope
                if v._data.writer is not None:
                    _modules = self.sess.verilog.gates[v._data.writer._driver]._position
                    scope = '.'.join(m._unique_name for m in _modules)  
                else:
                    scope = self.sess.module._position[0]._unique_name
                self._signals[v] = (scope, name) 
                
                
    def _emit(self, filename: str):
        f = open(filename, 'w', encoding='utf8') 
        writer = VCDWriter(f, self._timescale, date='today')
        vars: Dict[hardware.Reader, Any] = {} 
        for signal, (scope, name) in self._signals.items():
            vars[signal] = self._register_var(writer, scope, name, signal)
            
        values = []
        for signal, var in vars.items():
            for t, v in zip(signal._timestamps, signal._values):
                if not isinstance(v, str):
                    v = int(v)
                values.append((t, v, var))
        values.sort(key=lambda _:_[0])
        for t, v, var in values:
            writer.change(var, t, v)
        writer.close()
        f.close()
        
    def _register_var(self, writer: VCDWriter, scope: str, name: str, signal: hardware.Reader):
        init = signal._values[0]
        if isinstance(init, str):
            var_type = 'string'
            size = None 
        else:
            var_type = 'integer'
            size = len(signal._type)
            init = int(init)
        try:
            return writer.register_var(scope, name, var_type, size=size, init=init) 
        except KeyError:
            name = f'{name}@{id(signal)}'
            return writer.register_var(scope, name, var_type, size=size, init=init)
