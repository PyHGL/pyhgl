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
from typing import  Optional, Dict, Any, List, Tuple, Generator, Callable

import io
import os
import inspect
import traceback
import time

import pyhgl.logic.utils as utils 
import pyhgl.tester.utils as tester_utils

from pyhgl.array import * 
import pyhgl.logic.config as config
import pyhgl.logic.hardware as hardware
import pyhgl.logic.verilogmodule as verilogmodule  
import pyhgl.logic.simulator as simulator
import pyhgl.logic.hglmodule as hglmodule
import pyhgl.logic.assertion as assertion


class Session(HGL):
    
    __slots__ = '__dict__', 'module', 'simulator', 'verilog', 'timing'
    
    def __init__(
            self, 
            *,
            conf: Tuple[config.HGLConf, list, dict] = None, 
            timing: config.TimingConf = None,
            verbose_conf = False,
            verbose_hardware = False,
            verbose_verilog = False,
            verbose_sim = False
        ) -> None:
        
        self.verbose_conf       = verbose_conf 
        self.verbose_hardware   = verbose_hardware 
        self.verbose_verilog    = verbose_verilog
        self.verbose_sim        = verbose_sim 
        self.enable_assert = False 
        self.enable_warning = False

        self._intend = 0
        self._prev_sess = []
        # count how many instance of modules in this session
        self.count_modules: int = 0
        
        # current module
        self.module: hglmodule.Module = None 
        # current simulator, unique
        self.simulator = simulator.Simulator(self) 
        # verilog emitter, unique 
        self.verilog = verilogmodule.Verilog(self) 
        # timing info
        self.timing = timing or config.TimingConf()   
        # waveform
        self.waveform = simulator.VCD(self)  
        # build/sumulate warnings
        self.log = _Logging()
        
        #-------------------------------------------------------
        if self.verbose_conf:
            print(tester_utils._fill_terminal('┏', '━', 'left')) 
            self.print(f"✭ New Session with config: {conf[0] if conf else None}")
            self.print(f'timing: {self.timing}')
        #-------------------------------------------------------        
        with self:
            # a dummy module before the real toplevel module 
            m = object.__new__(hglmodule.Module) 
            m.__head__()  
            m.dispatcher = default_dispatcher.copy
            self.module = m 
            
            if conf is not None: 
                conf_ret = conf[0].exec(conf[1], conf[2])
            else:
                conf_ret = ({},{})
            new_paras: Dict[str, Any] = conf_ret[0]
            subconfigs: Dict[config.HGLConf, None] = conf_ret[1]
            # clock: (clk, 0|1)  reset: (rst, 0|1) | None
            paras = {
                'clock': (hardware.Clock(), 1), 
                'reset': (hardware.Wire(hardware.UInt(0, name='reset')), 1),
            }  
            paras.update(new_paras)
            self.module._subconfigs.update(subconfigs)        
            self.module._conf = type("Conf_Global", (config.ModuleConf,), paras)

        
    def _new_module_name(self, name: str) -> str:
        """ return a valid unique name of module declaration 
        
        name:
            prefered name 
        return:
            unique name
        """
        ret = f'{name}_{self.count_modules}' 
        self.count_modules += 1 
        return ret 
    
    def _new_module_instance(self, m: hglmodule.Module) -> List[hglmodule.Module]:
        """ register to father module and return position in module tree
        """
        self.verilog.modules.append(m) 
        if self.module is None:
            return [m]
        else:
            self.module._submodules[m] = None           # module tree
            return self.module._position + [m]          # position
        
    def _new_gate(self, gate: hardware.Gate) -> None:
        """ synthesizable gate, return number of all gates
        """
        self._add_gate(gate) 
        self.simulator.insert_gate_event(0, gate)
        
    def _add_gate(self, gate: hardware.Gate, module: hglmodule.Module = None):
        """ verilog, mapping from (part of) gates to module
        """
        assert isinstance(gate, hardware.Gate)
        if module is None:
            module = self.module
        self.verilog.gates[gate] = module  
        
    def _remove_gate(self, gate: hardware.Gate):    
        if gate in self.verilog.gates:
            self.verilog.gates.pop(gate) 
        gate.forward = lambda : None
            
    def _get_timing(self, id: str) -> Optional[dict]: 
        return self.timing.get(id) 
        
    def print(self, obj: object, intend: int = 0):
        """ print for debug
        """
        if intend > 0:
            self._intend += intend 
            print(tester_utils._fill_terminal(self._intend * '┃  ' + '┏', '━', 'left'))
        elif intend < 0:
            print(tester_utils._fill_terminal(self._intend * '┃  ' + '┗', '━', 'left'))
            self._intend += intend 
        if obj is None: return 
        s = str(obj)
        i =  self._intend * '┃  ' + '┃'  
        print(i + f'\n{i}'.join(s.splitlines()))
                
    def enter(self):
        self._prev_sess.append(self._sess)
        self._sess = self 
        
    def exit(self):
        self._sess = self._prev_sess.pop()
        
    def __enter__(self):
        self.enter()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()
    
    def emitVCD(self, filename='test.vcd'):
        if filename[-4:] != '.vcd':
            raise ValueError('invalid filename: use xxx.vcd')  
        
        filename = tester_utils.relative_path(filename, 2)
        self.waveform._emit(filename)
        print(tester_utils._yellow('Waveform: ') + filename)
        return filename
    
    
    def emitVerilog(self, filename='test.sv', **kwargs):
        if not (filename[-3:] == '.sv' or filename[-2:] == '.v'):
            raise ValueError('invalid filename')
        filename = tester_utils.relative_path(filename, 2)
        self.verilog.emit(filename, **kwargs) 
        print(tester_utils._yellow('Verilog: ') + filename)
        return _IcarusVerilog(filename)
    
    
    def emitGraph(self, filename='test.gv'):
        if not (filename[-3:] == '.gv' or filename[-4:] == '.dot'):
            raise ValueError('invalid filename')
        filename = tester_utils.relative_path(filename, 2)
        ret = self.verilog.emitGraph(filename) 
        print(tester_utils._yellow('Graphviz: ') + filename)
        return ret 
            
            
    def emitSummary(self):
        print(self)

    def __str__(self):
        ret = [tester_utils._yellow("Summary: ")]
        ret.append(f'  n_modules: {self.count_modules}')
        ret.append(f'  n_gates: {len(self.verilog.gates)}')
        ret.append(f'  t: {self.simulator.t}')
        ret.append(str(self.log))
        return '\n'.join(ret)
                
        
    # waveform 
    def track(self, *args, **kwargs):
        self.waveform._track(*args, **kwargs)
        
        
    def run(self, dt: Union[int, str]) -> None:
        self.enter()
        if isinstance(dt, str):
            dt = self.timing._get_timestep(dt)
        self.simulator.step(dt)
        self.exit() 
        
    def task(self, *args):
        with self:
            for g in args:
                assert inspect.isgenerator(g)
                self.simulator.insert_coroutine_event(0, g)





class _Logging:
    def __init__(self) -> None:
        self.warnings = []
        self.n_exec_gates = 0
        
    def warning(self, msg: str, start: int = 3, end: int = 100):
        x = ['Traceback:\n', utils.format_hgl_stack(start, end)]
        msg = '  ' + '  '.join(msg.splitlines(keepends=True))
        x.insert(0, f"{tester_utils._red('Warning:')}\n{msg}\n")
        self.warnings.append(''.join(x)) 

    def error(self, msg):
        ... 
        
    def info(self, msg):
        ...
        
    def __str__(self) -> str:
        ret = [f'  n_exec_gates: {self.n_exec_gates}']
        for i in self.warnings:
            ret.append('─────────────────────────────────────────────────')
            ret.append(str(i))
        return '\n'.join(ret)


class _IcarusVerilog:
    def __init__(self, filename) -> None:
        self.filename = filename 

    def sim(self):
        filename = self.filename
        assert os.path.isfile(filename)
        if filename[-3:] == '.sv':
            target = filename[:-3]  
        elif filename[-2:] == '.v':
            target = filename[:-2] 
        else:
            raise Exception('invalid filename')
        
        t = time.time()
        os.system(f'iverilog -g2012 -o {target}.vvp {filename}')
        print(f'{tester_utils._yellow("iverilog_compile:")} {filename} -> {target}.vvp, cost {time.time()-t} s')
        t = time.time()
        os.system(f'vvp {target}.vvp')
        print(f'{tester_utils._yellow("iverilog_sim:")} {time.time()-t} s')