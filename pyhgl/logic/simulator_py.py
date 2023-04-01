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
import bisect 
import vcd

from sortedcontainers import SortedDict
from itertools import chain

from pyhgl.array import *
import pyhgl.logic.hgl_core as hgl_core
import pyhgl.logic._session as _session 
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic.utils as utils


class Simulator(HGL):
    
    __slots__ = 'sess', 'time_wheel', 'priority_queue', 't', 'changed_gates', 'triggered_events','__dict__'
    
    def __init__(self, sess: _session.Session, length: int = 8):
        """
        events are stored in a time_wheel, overflowed events are stored in a priority queue
        
        - signal event:  (SignalData, value)
            - signal events are executed first 
            - one signal cannot update twice in one step
            - delay >= 1
            
        other kinds of events can read current signal value and insert events at delay=0
        - gate event: 
            - call when gate changes, or any input of gate changes
            - one gate connot update twice in one step
        - coroutine event: generator
            - trigger: delay
            - when exec, signal time is (simulator time + 1)
        - edge event: generator
            - call when triggered by signal
            - when exec, signal time is (simulator time + 1)
        """
        self.sess = sess
        # length >= 2, signal_events, coroutine_events
        self.time_wheel: List[
            Tuple[
                List[tuple],                 # signal_events
                List[Generator]              # coroutine_events
            ]
        ] = [self.new_events() for _ in range(1 << length)]
        # Dict[int, Event]
        self.priority_queue = SortedDict() 
        # remove duplicated gates
        self.changed_gates: Dict[hgl_core.Gate, None] = {} 
        # user defined event triggered by signal
        self.triggered_events: Dict[hgl_core.SignalData, List[Generator]] = {}
        # current time, increase 1 after step()
        self.t: int = 0

        # TODO cpp changed data, get triggered coroutine_events 
        self.sensitive_data: Dict[int, hgl_core.SignalData] = {}

    def new_events(self):
        return ([],[])
        
    def insert_signal_event(
        self, 
        dt: int, 
        target: hgl_core.SignalData,
        value: hgl_core.Logic,
        trace: Union[str, hgl_core.Gate, None],
    ): 
        """
        - dt: > 0 if insert by gate, >= 0 if insert by coroutine event
        - event: (SignalData, value, trace)
        """ 
        if self.sess.verbose_sim: 
            if isinstance(trace, hgl_core.Gate):
                trace = trace.trace
            assert isinstance(target, hgl_core.SignalData), f'wrong target: {target}\n{trace}'
            assert isinstance(value, hgl_core.Logic), f'wrong immd: {type(value)} - {value}\n{trace}'
        try:
            self.time_wheel[dt][0].append((target, value, trace)) 
        except:
            t = self.t + dt
            if t not in self.priority_queue:
                self.priority_queue[t] = self.new_events()
            self.priority_queue[t][0].append((target, value, trace))
            
    def insert_coroutine_event(self, dt: int, event: Generator):
        """ generator
        """
        try:
            self.time_wheel[dt][1].append(event)
        except:
            t = self.t + dt
            if t not in self.priority_queue:
                self.priority_queue[t] = self.new_events()   
            self.priority_queue[t][1].append(event)   

    def add_sensitive(self, data: hgl_core.SignalData):
        self.triggered_events[data] = []

    def init(self):
        for g in self.sess.verilog.gates:
            self.changed_gates[g] = None

    def init_cpp(self):
        # get a sensitive map from TData to SignalData: cpp index: data
        for signal in self.sess.waveform._signals:
            data = signal._data
            for i in chain(data._v, data._x):
                self.sensitive_data[i._cpp_data[1]] = data

               
    def step(self, n: int = 1):
        """ step 1 timescale
        
        each signal should not updated more than once in 1 step
        each gate should not called more than once in 1 step
        """
        time_wheel = self.time_wheel 
        queue = self.priority_queue
        
        for next_t in range(self.t + 1, self.t + n + 1):                    # 100ms
            
            signal_events, coroutine_events = time_wheel[0]    # 200ms 
            gate_events = self.changed_gates
            
            if signal_events or gate_events or coroutine_events:            # 200ms  
                # insert triggered coroutine events to next step
                coroutine_events_next = time_wheel[1][1]                    # 80ms
                #------------------------------------------
                if self.sess.verbose_sim:
                    _signal = ','.join(self._show_signal_event(i) for i in signal_events)
                    self.sess.print(f't={self.t}, signals: {_signal}')
                #------------------------------------------
                
                # 1. exec user tasks
                for g in coroutine_events:                                  # 400ms clock
                    try:
                        ret = next(g)
                        if isinstance(ret, int):
                            self.insert_coroutine_event(ret, g)
                        elif isinstance(ret, str):
                            self.insert_coroutine_event(self.sess.timing._get_timestep(ret), g)
                        elif isinstance(ret, hgl_core.Reader): 
                            self.triggered_events[ret._data].append(g)
                        else:
                            self.sess.log.warning(f'{ret} is not valid trigger, stop task {g}')
                    except StopIteration:
                        pass   
                    except:
                        self.sess.log.warning(traceback.format_exc())
                coroutine_events.clear()  
                
                # 2. update signal values
                for e in signal_events:
                    
                    target: hgl_core.SignalData 
                    value:  hgl_core.LogicData 
                    
                    target, value, _ = e
                    # cannot update signal twice, multiple driver
                    # if target._t == next_t:
                    #     self.sess.log.warning(f'update {target._name} twice at t={next_t}')
                    # target._t = next_t  
                    if target._update_py(value):    # value changed 
                        for reader in target.reader: 
                            # changed gates 
                            gate_events.update(reader._driven)
                            # triggered events
                            if _e:=self.triggered_events.get(target):
                                coroutine_events_next.extend(_e)
                                _e.clear()
                signal_events.clear() 
                
                #------------------------------------------
                if self.sess.verbose_sim:
                    _gate = ','.join(f'{i}' for i in gate_events)
                    self.sess.print(f't={self.t}, gates: {_gate}')
                #------------------------------------------
                
                # 3. execute gate
                for g in gate_events: 
                    g.forward() 
                self.sess.log.n_exec_gates += len(gate_events)
                gate_events.clear() 
                
                if signal_events:
                    raise Exception('insert new events of zero-delay')
                    
            time_wheel.append(time_wheel.pop(0))        # 700ms
            
            if queue:                                   # 80ms
                t0 = next_t + len(time_wheel) - 1
                if queue.keys()[0] == t0:
                    self.time_wheel[-1] = queue.pop(t0) 
            # update time
            self.t = next_t                             # 120ms
            
    def step_cpp(self, n: int = 1):
        """
        1. user task 
        2. cpp.step
        3. if some data is triggered, insert triggered user task 
        4. update t, continue
        """
        dll = self.sess.sim_cpp.dll 
        _t = self.t + n
        while self.t < _t:
            ...
            
    def _show_signal_event( self, e: Tuple[hgl_core.SignalData, hgl_core.LogicData, Any ]) -> str:
        target, value, _ = e
        name = target._name 
        return f"{name}:{target} <- {value}"
            

    
    

    
# ------------
# vcd writer 
# ------------ 
 
# TODO add to Reader
class Tracker(HGL): 

    _sess: _session.Session

    def __init__(self, signal: hgl_core.Reader):
        self.input = signal 
        # store history values for verification and waveform
        self.timestamps: list[int] = []
        self.values: list[hgl_core.Logic] = [] 

        # signal name
        name = signal._data._name 
        # signal scope
        if (m:=signal._data._module) is not None:
            scope = '.'.join(i._unique_name for i in m._position)  
        else:
            scope = self._sess.module._position[0]._unique_name 
        self.info = (scope, name)
    
    def __iter__(self):
        # called by python simulator before value updated at t
        while 1:
            self.timestamps.append(self._sess.sim_py.t)  
            self.values.append(self.input._data._getval_py())  
            yield self.input

    def history(self, t: int) -> hgl_core.Logic:
        """ get value at time t
        """
        idx = bisect.bisect(self.timestamps, t) - 1
        if idx < 0: 
            idx = 0 
        return self.values[idx]


class VCD(HGL):
        
    def __init__(self, sess: _session.Session):
        
        self.sess = sess
        real_timescale: float = sess.timing.timescale or 1e-9
        self._timescale: str = f'{round(real_timescale/1e-9)} ns' 

        self._trackers: Dict[Tracker, Tuple[str]] = {}  # {Tracker:('Module_0.Adder_1', 'io_x_1')}
        self._tracked_data: Dict[hgl_core.SignalData, None] = {}
    
    def _track(self, *args: hgl_core.Reader):
        new_signals = []
        for i in args:
            i = ToArray(i)
            if isinstance(i, Array):
                new_signals.extend(i._flat) 
            elif isinstance(i, module_hgl.Module):
                for _, v in i.__dict__.items():
                    if isinstance(v, hgl_core.Reader):
                        new_signals.append(v)
            elif isinstance(i, hgl_core.Reader):
                new_signals.append(i) 
        
        for s in new_signals:
            if isinstance(s, hgl_core.Reader) and s._data not in self._tracked_data: 
                self._tracked_data[s._data] = None 
                tracker = Tracker(s)
                self._trackers[tracker] = None 
                self.sess.sim_py.insert_coroutine_event(0, iter(tracker)) 
                self.sess.sim_py.add_sensitive(s._data)
                
                
    def _dump(self, filename: str):
        f = open(filename, 'w', encoding='utf8') 
        writer = vcd.VCDWriter(f, self._timescale, date='today')

        vars: Dict[Tracker, Any] = {} 
        for tracker in self._trackers: 
            vars[tracker] = self._register_var(writer, tracker)
            
        values = []
        for tracker, var in vars.items():
            width = len(tracker.input)
            for t, v in zip(tracker.timestamps, tracker.values):
                if not isinstance(v, str):  # Logic 
                    v = utils.logic2str(v.v, v.x, width, prefix=False)
                values.append((t, v, var))
        values.sort(key=lambda _:_[0]) 
        for t, v, var in values:
            writer.change(var, t, v)
        writer.close()
        f.close()
        
    def _register_var(self, writer: vcd.VCDWriter, tracker: Tracker):
        init = tracker.values[0] 
        scope, name = tracker.info
        if isinstance(init, str):
            var_type = 'string'
            size = None 
        else:
            var_type = 'wire'
            size = len(tracker.input)
            init = utils.logic2str(init.v, init.x, size, prefix=False)
        try:
            return writer.register_var(scope, name, var_type, size=size, init=init) 
        except KeyError:
            name = f'{name}@{id(tracker)}'
            return writer.register_var(scope, name, var_type, size=size, init=init)
