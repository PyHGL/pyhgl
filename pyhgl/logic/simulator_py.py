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
from typing import Generator, List, Tuple, Set, Literal

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
import pyhgl.logic._config as _config
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic.utils as utils 
import pyhgl.tester.runner as tester_runner


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
            - yield trigger: t_delay | signal | signaldata
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
        if data.events is None:
            data.events = []

    def init(self):
        for g in self.sess.verilog.gates:
            self.changed_gates[g] = None

    def init_cpp(self):
        # get a sensitive map from TData to SignalData: cpp index: data
        for signal in self.sess.waveform._signals:
            data = signal._data
            for i in chain(data._v, data._x):
                self.sensitive_data[i._cpp_data[1]] = data

               
    def step(self):
        """ step 1 timescale
        
        each signal should not updated more than once in 1 step
        each gate should not called more than once in 1 step
        """
        time_wheel = self.time_wheel 
        queue = self.priority_queue
        next_t = self.t + 1
        
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
                    # stop task
                    if ret is None:
                        pass
                    # delay time step
                    elif isinstance(ret, int):
                        self.insert_coroutine_event(ret, g)
                    # wait signal change
                    elif isinstance(ret, hgl_core.Reader):  
                        self.add_sensitive(ret._data) 
                        ret._data.events.append(g)
                    # wait signal change
                    elif isinstance(ret, hgl_core.SignalData):
                        self.add_sensitive(ret)
                        ret.events.append(g)
                    else:
                        self.sess.log.warning(f'{ret} is not valid trigger, stop task {g}')
                except:
                    pass   
            coroutine_events.clear()  
            
            # 2. update signal values
            for e in signal_events:
                
                target: hgl_core.SignalData 
                value:  hgl_core.LogicData 
                
                target, value, trace = e
                # cannot update signal twice
                if target._update_py(value):        # value changed 
                    if target.tracker is not None:  # record history
                        target.tracker.record()
                    for reader in target.reader: 
                        # triggered gates 
                        gate_events.update(reader._driven)
                        # triggered events 
                        if (ces:=target.events) is not None:
                            coroutine_events_next.extend(ces)
                            ces.clear()
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
            
                
        time_wheel.append(time_wheel.pop(0))        # 700ms
        
        if queue:                                   # 80ms
            t0 = next_t + len(time_wheel) - 1
            if queue.keys()[0] == t0:
                time_wheel[-1] = queue.pop(t0) 
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
            

    
# --------------------
# simulation task tree
# --------------------

def task(f: Callable):
    return _Task(f)

class _Task(HGL):
    """ simulation task function, based on corouting 

    @task test_adder(self, dut):
        for _ in range(100):    
            x = setr(dut.io.x)
            y = setr(dut.io.y)
            yield 100 
            self.Assert(getv(dut.io.out) == x + y)
        yield self.join(task1(dut), task2(dut), task3(dut)) 
        yield self.join_any(task1(dut), task2(dut), task3(dut))
        yield task1(dut), task2(dut)
        yield self.edge(clk, 1)
        yield dut.io.ready 
        yield self.until(reset, 0)
        yield self.clock()  # default clock 
        yield generator     # inspect.isgenerator() 
        self.EQ += 1,2
    """
    def __init__(self, f: Callable) -> None:
        # test function
        assert inspect.isgeneratorfunction(f)
        self.f = f 
        self.sess: _session.Session = None 
        # arguments of test function
        self.args = None
        self.kwargs = None
        # callback function when task finished
        self.callback: Callable = None 
        self.father: _Task = None
        # record assertions 
        self.test_key = []
        self.test_node: tester_runner._TestTreeNode = None 
        # generator status 
        self.it: Generator = None
    
    def _init(self, father: Union[_Task, _session.Session], callback: Callable = None):
        """ record sess, test_node, callback
        """
        if isinstance(father, _Task):
            self.sess = father.sess 
            self.test_key = father.test_key + [self.f.__name__]
            self.test_node = tester_runner._root.get_or_register(
                [self.test_key[0], '.'.join(self.test_key[1:])]
            ) 
            self.father = father
        elif isinstance(father, _session.Session):
            self.sess = father 
            self.test_key = [father.filename, self.f.__name__]
            self.test_node = tester_runner._root.get_or_register(self.test_key)
            self.father = None
        else:
            raise Exception()
        self.callback = callback 
        self.EQ = tester_runner.EQ(self.test_node)

    def __call__(self, *args: Any, **kwargs: Any) -> _Task: 
        ret = _Task(self.f)
        ret.args = args 
        ret.kwargs = kwargs
        return ret

    def __iter__(self):
        self.it = self._iter()
        return self.it

    def _iter(self):
        try:
            assert self.args is not None, 'uninitialized task'   
            for ret in self.f(self, *self.args, **self.kwargs): # new iterator   
                if ret is None:
                    break
                elif isinstance(ret, (int, hgl_core.Reader)):  # timestep or sensitive
                    yield ret 
                elif isinstance(ret, str):          # ex. '50ns'
                    yield self.sess.timing._get_timestep(ret)  
                elif inspect.isgenerator(ret):
                    yield from ret 
                elif isinstance(ret, _Task):        # ex. task1()
                    yield from self.join(ret)
                elif isinstance(ret, (tuple,list,Array)): # ex. [task1(), task2()]
                    for i in ret:
                        yield from self.join(i)
                else:                               
                    raise TypeError(f'{ret}')
        except Exception:                       # ignore GeneratorExit
            self.test_node.exception = traceback.format_exc()
        finally:
            if self.callback is not None:
                self.callback() 
            self.test_node.finish(f'step={self.t}')

    @property 
    def t(self):
        return self.sess.sim_py.t 
        
    def join(self, *args: _Task):
        """ blocked until all tasks finished
        """
        for i in args:
            assert isinstance(i, _Task) 
        n_finished = 0
        n_total = len(args)
        def join_callback():
            nonlocal n_finished
            n_finished += 1  
            if n_finished == n_total:
                self.sess.sim_py.insert_coroutine_event(0, self.it)
        for task in args:
            task._init(father=self, callback=join_callback)
            self.sess.sim_py.insert_coroutine_event(0, iter(task)) 
        yield None
    
    def join_none(self, *args: _Task):
        """ no block
        """
        for i in args:
            assert isinstance(i, _Task)
            i._init(father=self)
            self.sess.sim_py.insert_coroutine_event(0, iter(i))
        yield from []

    def join_any(self, *args: _Task):
        """ blocked until one task finished
        """
        for i in args:
            assert isinstance(i, _Task)
        n_finished = 0
        n_total = len(args)
        def join_callback():
            nonlocal n_finished
            n_finished += 1  
            if n_finished == 1:
                self.sess.sim_py.insert_coroutine_event(0, self.it)
        for task in args:
            task._init(father=self, callback=join_callback)
            self.sess.sim_py.insert_coroutine_event(0, iter(task)) 
        yield None

    def edge(self, signal: hgl_core.Reader, value: Literal[0,1]):
        value = 1 if value > 0 else 0
        value_n = 1 - value 
        while signal._data._getval_py() != value_n:
            yield signal 
        while signal._data._getval_py() != value:
            yield signal 
        
    def clock(self, n = 1):
        default_clk = self.sess.module._conf.clock  
        for _ in range(n):
            yield from self.edge(*default_clk)
    
    def clock_n(self, n = 1):
        """ there is a small delay of register output after clock edge, wait half clock cycle
        """
        clk, edge = self.sess.module._conf.clock 
        for _ in range(n):
            yield from self.edge(clk, 0 if edge else 1)
    
    def reset(self, n = 1):
        default_rst = self.sess.module._conf.reset 
        if not default_rst:
            return 
        else:
            rst, edge = default_rst
            yield from self.clock_n()
            hgl_core.setv(rst, edge) 
            yield from self.clock_n(n)
            hgl_core.setv(rst, not edge)

    def until(self, signal: hgl_core.Reader, value: Any):
        while signal._data._getval_py() != value:
            yield signal
        
    def Assert(self, v: bool):
        tester_runner._AssertTrue(v, self.test_node)
    
    def Eq(self, a, b):
        tester_runner._AssertEq((a,b), self.test_node)
    
    def AssertEq(self, a, b):
        tester_runner._AssertEq((a,b), self.test_node, msg=f'  step={self.t}')

# ------------
# vcd writer 
# ------------ 
 
# TODO add to Reader
class Tracker(HGL): 

    _sess: _session.Session

    def __init__(self, data: hgl_core.SignalData):
        self.sess = self._sess
        self.input_data = data 
        # store history values for verification and waveform
        self.timestamps: list[int] = []
        self.values: list[hgl_core.Logic] = [] 
        # record at once
        self.timestamps.append(self.sess.sim_py.t)  
        self.values.append(self.input_data._getval_py()) 

        # dumpVCD
        # signal name
        name = data._name 
        # signal scope
        if (m:=data._module) is not None:
            scope = '.'.join(i._unique_name for i in m._position)  
        else:
            scope = self._sess.module._position[0]._unique_name 
        self.info = (scope, name)
    
    def record(self):
        self.timestamps.append(self.sess.sim_py.t + 1)  
        self.values.append(self.input_data._getval_py()) 

    def history(self, t: int) -> hgl_core.Logic:
        """ get value at time t
        """
        idx = bisect.bisect(self.timestamps, t) - 1
        if idx < 0: 
            idx = 0 
        return self.values[idx] 
    
    def history_idx(self, t: int) -> int:
        idx = bisect.bisect(self.timestamps, t) - 1
        if idx < 0: 
            idx = 0 
        return idx


class VCD(HGL):
        
    def __init__(self, sess: _session.Session):
        
        self.sess = sess
        real_timescale: float = sess.timing.timescale or 1e-9
        self._timescale: str = f'{round(real_timescale/1e-9)} ns' 

        self._tracked_data: Dict[hgl_core.SignalData, None] = {}  
    
    def _track(self, *args: hgl_core.Reader): 
        array_expanded1 = []
        for i in args:
            obj = ToArray(i) 
            if isinstance(obj, Array):
                array_expanded1.extend(obj._flat)
            else:
                array_expanded1.append(obj)
        module_expanded = []
        for i in array_expanded1:
            if isinstance(i, module_hgl.Module):
                for _, v in i.__dict__.items():
                    module_expanded.append(v)
                if isinstance(i._conf.clock, tuple):
                    module_expanded.append(i._conf.clock[0])
                if isinstance(i._conf.reset, tuple):
                    module_expanded.append(i._conf.reset[0])  
            else:
                module_expanded.append(i)
        array_expanded2 = []
        for i in module_expanded:
            if isinstance(i, Array):
                array_expanded2.extend(i._flat)
            elif isinstance(i, hgl_core.Reader):
                array_expanded2.append(i)
        
        for s in array_expanded2: 
            if isinstance(s, hgl_core.Reader) and s._data not in self._tracked_data: 
                tracker = Tracker(s._data)
                s._data.tracker = tracker
                self._tracked_data[s._data] = None 
                
                
    def _dump(self, filename: str):
        f = open(filename, 'w', encoding='utf8') 
        writer = vcd.VCDWriter(f, self._timescale, date='today')

        vars: Dict[Tracker, Any] = {}  
        for data in self._tracked_data:
            vars[data.tracker] = self._register_var(writer, data.tracker)
            
        values = []
        for tracker, var in vars.items():
            width = len(tracker.input_data)
            for t, v in zip(tracker.timestamps, tracker.values):
                if not isinstance(v, str):  # Logic 
                    v = utils.logic2str(v.v, v.x, width=width, prefix=False, radix='b')
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
            size = len(tracker.input_data)
        try:
            return writer.register_var(scope, name, var_type, size=size) 
        except KeyError:
            name = f'{name}@{id(tracker)}'
            return writer.register_var(scope, name, var_type, size=size)


