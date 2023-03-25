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
from typing import Any, Dict, List, Set, Union, Tuple, Optional, Literal

import sys
import ctypes
import platform
import subprocess
import os
import shutil
import gmpy2
from itertools import chain
from pyhgl.array import HGL
import pyhgl.logic.hgl_basic as hgl_basic
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic._session as _session
import pyhgl.logic.utils as utils 
import pyhgl.tester.utils as tester_utils



class AST(HGL):

    _sess: _session.Session 

    def __str__(self):
        name = self.__class__.__name__ 
        fields = self.__dict__ 
        body = ''.join([f'{k}: {v}\n' for k, v in fields.items()]) 
        body = '  ' + '  '.join(body.splitlines(keepends=True))
        return f'{name}{{\n{body}}}' 


class Expr(AST):
    """ multiple inputs/outputs

    x = a & b & c 
    x = cat([a,b,c][idx0], [a,b,c][idx1])   # part select

    x3 = a << 3 | b >> 3
    """
    def __init__(self) -> None:
        self.inputs: List[TData] = []       # data may read
        self.outputs: List[TData] = []      # data may write
        self.masks: List[int] = []          # masks
        self.delay: int = 1 
    
    def dump(self, builder: Node) -> str:
        raise NotImplementedError() 
    
    def copyx(self):
        ...
    

class TData(AST):

    def __init__(
        self, 
        v: int = 0, 
        width: int = 64, 
        storage: Literal['local', 'global', 'const', 'global_always', 'local_static'] = 'global',
        name: str = 'temp',
    ) -> None:
        self._cpp_data: Tuple = None   # (global_data, idx, setv)

        self.v: int = int(v & gmpy2.bit_mask(width))  # initial value, 64 bits
        self.width = width 
        self.storage = storage
        self.name = name  

        self.source: Expr = None    # ex. x <== a & b & c
        self.writer: Node = None
        self.reader: Dict[Node, None] = {}

    def __len__(self):
        return self.width 

    def __str__(self):
        return f'{self.storage} uint{self.width}_t {self.name}{{{self.v}}}'


def getval(l: List[TData]) -> gmpy2.xmpz:
    ret = gmpy2.xmpz(0)
    for i, data in enumerate(l): 
        pdata, idx_data, _ = data._cpp_data
        ret[i*64, i*64+63] = pdata[idx_data << 1]
    return ret

def setval(target: TData, mask: int, value: int, delay: int) -> None:
    """ event: (target, mask, value)
    """ 
    _, idx, cpp_setv = target._cpp_data 
    cpp_setv(idx, mask, value, delay)


def GlobalArray(v: gmpy2.mpz, bit_length: int, name: str) -> List[TData]:
    """ SignalData to cpp
    """
    ret = []
    start = 0 
    while start + 64 < bit_length:
        ret.append(TData(
            v[start, start+64], 
            width=64, 
            storage='global', 
            name=f'{name}{start//64}'
        )) 
        start += 64 
    ret.append(TData(
        v[start:bit_length], 
        width=bit_length-start, 
        storage='global', 
        name=f'{name}{start//64}'
    ))
    return ret 

def GlobalMem(v: gmpy2.mpz, shape: List[int], name: str, part: Literal['v', 'x']) -> List[List]:
    ret = []
    if len(shape) > 1:
        shape_next = shape[1:] 
        w = 1 
        for i in shape_next:
            w *= i
        for i in range(shape[0]):
            ret.append(GlobalMem(
                v=v[i*w, i*w+w],
                shape=shape_next,
                name=name,
                part=part
            )) 
    else:
        temp = GlobalArray(v=v, bit_length=shape[-1], name=name) 
        cpp: Cpp = HGL._sess.sim_cpp  
        global_data = cpp.golbal_v if part == 'v' else cpp.global_x
        for data in temp:
            data.storage = 'global_always'
            global_data[data] = None     # add to global, in order 
            ret.append(data)
    return ret

def ToData(v):
    if isinstance(v, TData):
        return v 
    else:
        return TData(v, storage='const') 

def Constant(v:int = 0, w = 64):
    return TData(v, width=w, storage='const')

def LocalVar(v = 0, w = 64):
    return TData(v, width=w, storage='local')

def GlobalVar(v = 0, w = 64):
    return TData(v, width=w, storage='global') 


class OpNot(Expr):
    def __init__(self, target: TData, input: TData, w: int = 64, delay: int = 1, mask:int = None):
        super().__init__()
        self.inputs.append(input)
        self.outputs.append(target)
        target.source = self 
        self.w = w 
        self.delay = delay 
        self.masks.append(mask)

    def dump(self, builder: Node) -> str:
        if self.w == 64:
            return f'~{builder.get_name(self.inputs[0])}'
        else:
            mask = f'{gmpy2.bit_mask(self.w)}ULL'
            return f'(~{builder.get_name(self.inputs[0])}) & {mask}'


class OpXor(Expr):
    def __init__(self, target: TData, *args: TData, delay: int = 1, mask: int = None) -> None:
        super().__init__()
        self.inputs = args
        self.outputs.append(target)
        target.source = self
        self.delay = delay 
        self.masks.append(mask)

    def dump(self, builder: Node) -> str:
        body = ' ^ '.join(builder.get_name(i) for i in self.inputs)
        builder.update()
        pass # TODO 


class OpAnd(OpXor):

    def dump(self, builder: Node):
        return ' & '.join(builder.get_name(a) for a in self.args)


class OpOr(OpXor):

    def dump(self, builder: Node):
        return ' | '.join(builder.get_name(a) for a in self.args)



class OpReg(Expr):
    def __init__(self) -> None:
        self.clock = None 
        self.reset = None 
        self.target = None # always global 
        self.input = None 
        # read: input, clock, reset, target 
        # trigger: clock, reset 
        # merge: not allowed, because trigger is different

class Update(AST):
    def __init__(
        self, 
        left: TData, 
        right: Union[TData, Expr], 
        delay: int = 0, 
        mask: Optional[int] = None
    ) -> None:

        self.left = left 
        self.right = right 
        self.delay = delay
        self.mask = mask

    def dump(self, builder: Node) -> str:
        left = builder.get_name(self.left)
        if isinstance(self.right, TData):
            right = builder.get_name(self.right)
        else:
            right = self.right.dump(builder)

        if self.left.storage in ['global', 'global_always']:
            if self.mask is None:
                mask = f'{gmpy2.bit_mask(64)}ULL'
            else:
                mask = f'{self.mask}ULL'
            return f'time_wheel.insert_signal_event(&{left}, {mask}, {right}, {self.delay});'
        else:
            if self.mask is None:
                return f'{left} = {right};'
            else:
                mask = f'{self.mask}ULL' 
                return f'{left} = {left} | {mask} & ({right});'


class Node(AST):
    """ represents a gate or a function in cpp

    optimization:
        - data in iports maybe const 
        - data in oports only read by self -> local var 
    """

    _sess: _session.Session  

    def __init__(self, name: str = '') -> None:
        self._sess.sim_cpp.nodes[self] = None  # add a new node
        self.body: List[AST] = []              # function body
        self.name: str = name or 'exec'
        self._localvars: Dict[TData, None] = {} 

        self.iports_v: Dict[TData, None] = {}
        self.iports_x: Dict[TData, None] = {}  # includes local var
        self.oports_v: Dict[TData, None] = {} 
        self.oports_x: Dict[TData, None] = {}  # includes local var
        self.oports_x_full: Dict[TData, None] = {} # local vars from v but read by x, reversed order
        self.unknown: bool = False
        self.iports = self.iports_v     
        self.oports = self.oports_v   

        self.trigger: Dict[TData, None] = None # default by all source; for reg, only clk & rst  
        # dump
        self._names_v: Dict[TData, str] = {}
        self._names_x: Dict[TData, str] = {}
        self._builder: Cpp = None

    def dump_graph(self, g): 
        iports = []
        oports = []
        body = []
        for i in chain(self.iports_v, self.iports_x):
            iports.append(f"<i{id(i)}> {i.name}") 
            if i.writer is not None:
                g.edge(f'{id(i.writer)}:o{id(i)}', f'{id(self)}:i{id(i)}')
        for o in chain(self.oports_v, self.oports_x):
            oports.append(f"<o{id(o)}> {o.name}")
            if o.source is not None:
                body.append(o.source.__class__.__name__) 

        label = "{{%s} | %s | {%s}}" % ( '|'.join(iports), ' '.join(body), '|'.join(oports))
        curr_gate = str(id(self))  
        g.node(name = curr_gate, label = label, shape='record', color='blue') 
        
    def dump_unknown(self):
        self.unknown = True
        self.iports = self.iports_x 
        self.oports = self.oports_x 

    def dump_value(self):
        self.unknown = False
        self.iports = self.iports_v 
        self.oports = self.oports_v

    def read(self, data: TData): 
        self.iports[data] = 1 
        data.reader[self] = None

    def write(self, data: TData):
        assert data.storage != 'const'
        self.oports[data] = None 
        data.writer = self

    def Xor(self, *args: TData, target: TData = None, delay: int = 1, mask: int = None) -> TData:
        for i in args:
            self.read(i) 
        if target is None:
            target = LocalVar() 
        self.write(target)
        self.oports[target] = OpXor(target, *args, delay=delay, mask=mask) 

    def merge(self, other: Node):
        for i in other.oports_v:
            i.writer = self  
            self.oports_v[i] = None 
        for i in other.oports_x:
            i.writer = self 
            self.oports_x[i] = None 
        for i in other.iports_v:
            if other in i.reader:
                i.reader.pop(other)
            i.reader[self] = None 
            self.iports_v[i] = None  
        for i in other.iports_x:
            if other in i.reader:
                i.reader.pop(other) 
            i.reader[self] = None 
            self.iports_x[i] = None 

    def optimize(self):
        for i in chain(self.oports_v, self.oports_x):
            if i.storage == 'global':
                if len(i.reader) == 1 and next(iter(i.reader)) is self:  # only read by self
                    i.storage = 'local'
    
    def split(self):
        for o in self.oports_x:
            self._split(o)

    def _split(self, data: TData):
        """ data: x output """
        self.oports_x_full[data] = None
        if data.source is not None:
            for i in data.source.inputs:
                if i.storage in ['local', 'local_static']:
                    self._split(i)

    def get_name(self, obj: TData) -> str:
        """ if is local variable, record declaration
        if global variable, return 'data.xxx'

        TODO x and v  use _{name}_{n}_local
        """
        if obj.storage in ['local', 'local_static']:
            names = self._names_x if self.unknown else self._names_v 
            if obj in names:
                return names[obj]
            else:
                ret = f'_{obj.name}_{len(names)}_local' 
                names[obj] = ret 
                return ret 
        elif obj.storage == 'const':
            return f'{obj.v}ULL'  
        else:
            return f'data.{self._builder.get_name(obj)}'  # global data 
        
    # TODO solve sensitive list

    def dump(self, builder: Cpp) -> str: 
        self._temp_builder = builder
        body = []
        for ast in self.body:
            body.append(ast.dump(self)) 
        local_var_dec = []
        for i in self._localvars:
            local_var_dec.append(f'uint64_t {builder.get_name(i)} = {i.v}ULL;')
        func_body = '\n'.join(chain(local_var_dec, body))
        ret = f'void {builder.get_name(self)}(){{\n{func_body}\n}}'
        self._temp_builder = None
        return ret
    
    def Not(self, a: TData, target: TData = None, width: int = 64, delay: int = 1, mask: int = None) -> TData:
        self.read(a)
        if target is None:
            target = LocalVar(v=0,w=width) 
        self.write(target)
        OpNot(target=target, input=a, w=width, delay=delay, mask=mask)
        return target

    def And(self, *args: TData, target: TData = None, delay: int = 1, mask: int = None) -> TData:
        for i in args:
            self.read(i)
        if target is None:
            target = LocalVar(0)
        self.write(target) 
        OpAnd(target, *args, delay=delay, mask=mask)
        return target

    def Or(self, *args: TData, target: TData = None, delay: int = 1, mask: int = None) -> TData:
        for i in args:
            self.read(i)
        if target is None:
            target = LocalVar(0)
        self.write(target) 
        OpOr(target, *args, delay=delay, mask=mask)
        return target 
    
    def AndR(self, *args: TData, target: TData = None, delay: int = 1) -> TData:
        for i in args:
            self.read(i)
        if target is None:
            target = LocalVar(0)
        self.write(target)
        # TODO OpAndR
        return target
    
    def Or2(self, a: List[TData], b: List[TData], target: List[TData] = None, delay: int = 1) -> List[TData]:
        return []

    def Bool(self, *args: TData, target: TData = None, delay: int = 1) -> TData:
        # TODO 
        return target

    def RShift(self, a: TData, b: TData) -> TData:
        return TData(0, local=True)

    def LShift(self, a: TData, b: TData) -> TData:
        return TData(0, local=True)







class Cpp(HGL):

    _sess: _session.Session

    def __init__(self, _sess: _session.Session) -> None:
        self.dll: ctypes.CDLL = None 
        # two Nodes per Gate
        self.nodes: Dict[Node, None] = {}
        # data : fanouts   v,x,v,x,v,x
        self.global_var: Dict[TData, List[Node]] = {}
        # used names
        self.names: Dict[Union[TData, Node], str] = {} 
        # node functions ex. void and_exec(){...}
        self.functions: List[str] = [] 

        # dump
        self.golbal_v: Dict[TData, None] = {}
        self.global_x: Dict[TData, None] = {}

    def dump_graph(self, filename:str='ast.gv'):
        if not (filename[-3:] == '.gv' or filename[-4:] == '.dot'):
            raise ValueError('invalid filename') 
        filename = os.path.join(self._sess.build_dir, filename)
        import graphviz
        g = graphviz.Digraph(
            'G', 
            format='svg', 
            filename=filename,
            engine='dot',
            strict=True, 
            graph_attr={'overlap':'false'}
        ) 
        for n in self.nodes:
            n.dump_graph(g)
        g.save()
        return g

    def get_name(self, obj: Union[TData, Node]) -> str:
        if isinstance(obj, TData):
            if obj.storage == 'const':
                return f'{obj.v}ULL'
        if obj in self.names:
            return self.names[obj]
        else:
            ret = f'_{obj.name}_{len(self.names)}'
            self.names[obj] = ret 
            return ret

    def _merge_gates(self):
        pass 


    def dump(self) -> None:
        for gate in self._sess.verilog.gates:
            gate.dump_cpp() 

        self._merge_gates()

        for n in self.nodes:
            n.optimize()
        for n in self.nodes:
            n.split() 

        assert len(self.global_x) == len(self.golbal_v) 

        # dump function. ex. void xor(){}
        # dump node. ex. Node xor_node {0, &xor}
        for n in self.nodes:
            n.dump(self)
        
        # nodes_def =[]
        # for n in self.nodes:
        #     fun_name = self.get_name(n)
        #     nodes_def.append(f'Node {fun_name}_node {{{0}, &{fun_name}}};') 

        # dump fanouts. ex. Node * xor_fanout [] = {nullptr};
        fanouts_def = []
        fanouts_init = []
        for data in chain(self.global_x, self.golbal_v):
            data_name = self.get_name(data)
            drivens = data.reader  
            _list = []
            for _node, _x in drivens: 
                node_name = self.get_name(_node) + '_nodex' if _x else '_nodev'  # xor_nodex 
                _list.append(f'&{node_name}')
            _list.append('nullptr') 
            _list = ','.join(_list)
            fanouts_def.append(f'Node *{data_name}_fanout[] = {{{_list}}};')
            fanouts_init.append(f'data.{data_name}_fa = {data_name}_fanout;')
        fanouts_def = '\n'.join(fanouts_def)
        fanouts_init = '\n'.join(fanouts_init)
        fanouts_init = f"void init_data(){{{fanouts_init}}}"

        # dump struct
        struct_body = [] 
        for data in chain(self.global_x, self.golbal_v):
            data_name = self.get_name(data)
            struct_body.append(f'uint64_t {data_name} {{{data.v}ULL}}; Node **{data_name}_fa {{}};')
        struct_body = '\n'.join(struct_body)
        struct = f'#pragma pack(8)\nstatic struct{{\nuint64_t start;\n{struct_body}\nuint64_t end;\n}} data;\nconstexpr uint64_t data_size = {len(self.global_x)*2};'
        
        ret = [_template1, struct, _template2, fanouts_def, fanouts_init, _template3]
        ret = '\n'.join(ret)

        home = os.path.join(self._sess.build_dir, 'sim')
        if os.path.exists(home):
            shutil.rmtree(home)
        os.makedirs(home)
        with open(os.path.join(home, 'CMakeLists.txt'), 'w', encoding='utf8') as f:
            f.write("""
                cmake_minimum_required(VERSION 3.15)

                set(CMAKE_CXX_STANDARD 20)
                set(CMAKE_CXX_STANDARD_REQUIRED ON)
                project(pyhgl_sim)

                add_library(pyhgl_ext SHARED main.cpp)
            """)
        with open(os.path.join(home, 'main.cpp'), 'w', encoding='utf8') as f:
            f.write(ret) 


    def build(self):
        home = os.path.join(self._sess.build_dir, 'sim')
        cmake_build_dir = os.path.join(home, 'cpp')
        subprocess.run(['cmake', '-S', home, '-B', cmake_build_dir])
        subprocess.run(['cmake', '--build', cmake_build_dir, '--config', 'Release']) 
        dll_path = os.path.join(cmake_build_dir, 'Release/pyhgl_ext.dll')
        self.dll_load(dll_path) 
        self.dll.init(0)
        _pdata = self.dll.get_global_data()
        _psetv = self.dll.setv
        for idx, data in enumerate(self.global_var):
            data._cpp_data = (_pdata, idx, _psetv)

    def dll_load(self, path: str):
        dll = ctypes.CDLL(path)
        dll.gett.restype = ctypes.c_uint64
        dll.getv.argtypes = [ctypes.c_uint64]
        dll.getv.restype = ctypes.c_uint64
        dll.get_global_data.restype = ctypes.POINTER(ctypes.c_uint64)
        dll.get_changed_signal.argtypes = [ctypes.c_uint64]
        dll.get_changed_signal.restype = ctypes.c_uint64
        dll.setv.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        dll.step.argtypes = [ctypes.c_uint64] 
        dll.step.restype = ctypes.c_uint64
        dll.init.argtypes = [ctypes.c_uint64]
        self.dll = dll

    def dll_close(self):
        utils.get_dll_close()(self.dll._handle)
        self.dll = None



_template1 = """
#include <iostream>
#include <format>
#include <vector>
#include <array>
#include <chrono>
#include <queue>

using std::uint64_t;

constexpr unsigned N_EVENT = 8; // length of timewheel

struct Node
{
    void (*exec)(Node *);
    bool ready = false; // already in the list, don't update twice
    bool valid = true;  // whether triggered by x update
};

struct Event
{
    /* target[mask] <== value at time */
    uint64_t *target;
    uint64_t mask;
    uint64_t value;
    uint64_t time;

    bool operator<(const Event &e) const
    {
        return time > e.time;
    }
};

/*********************************************************************/


"""
        
_template2 = """


/*********************************************************************/

uint64_t *data_start = &data.start + 1;
uint64_t *data_end = &data.end;
// uint64_t data_size = data_end - data_start; // len(data) * 2

struct
{
    std::vector<Event> event_list[1 << N_EVENT]{};
    std::priority_queue<Event> event_queue{};

    uint64_t t{0};                     // current time
    std::vector<Node *> changed_nodes {}; // nodes whose input changed
    std::vector<int64_t> chagned_data {}; // index of changed signal data 

    inline void insert_signal_event(uint64_t *target, uint64_t mask, uint64_t value, uint64_t delay)
    {
        auto time = t + delay;
        if (delay < (1 << N_EVENT))
            event_list[time % (1 << N_EVENT)].push_back(Event{target, mask, value, time});
        else
            event_queue.push(Event{target, mask, value, time});
    }

    inline void step()
    {
        auto &signal_events = event_list[t % (1 << N_EVENT)];
        if (!signal_events.empty())
        {
            for (auto &event : signal_events)
            {
                uint64_t odd_value = event.target[0];
                uint64_t mask = event.mask;
                uint64_t new_value = event.value; 
                bool is_x = event.target - data_start >= data_size;

                if ((odd_value & mask) != new_value) // value changed
                {
                    *event.target = odd_value & (~mask) | new_value; // update value
                    Node **fanouts = (Node **)event.target[1];
                    for (; Node *pnode = *fanouts; fanouts++) // until pnode == nullptr
                    {
                        if (!pnode->ready && (pnode->valid || is_x))
                        {
                            // always triggered
                            // only triggered by x
                            pnode->ready = true;
                            changed_nodes.push_back(pnode);
                        }
                    }
                }
            }
            signal_events.clear();
        }
        // execute gate
        for (auto pnode : changed_nodes)
        {
            pnode->exec(pnode);
            pnode->ready = false;
        }
        changed_nodes.clear();
        // deal with event_queue
        if (!event_queue.empty())
        {
            while (event_queue.top().time == t + (1 << N_EVENT))
            {
                signal_events.push_back(event_queue.top());
                event_queue.pop();
            }
        }
        // next t
        t = t + 1;
    }

} events;

/*********************************************************************/



"""


_template3 = """

extern "C" _declspec(dllexport) uint64_t gett()
{
    return events.t;
}

extern "C" _declspec(dllexport) uint64_t getv(uint64_t idx)
{
    idx = idx << 1;
    if (idx < data_size)
        return data_start[idx];
    else
        return 0;
}

extern "C" _declspec(dllexport) void setv(uint64_t idx, uint64_t mask, uint64_t v, uint64_t delay)
{
    idx = idx << 1;
    if (idx < data_size)
    {
        events.insert_signal_event(data_start + idx, mask, v, delay);
    }
}

extern "C" _declspec(dllexport) uint64_t * get_global_data()
{
    return data_start;
}

extern "C" _declspec(dllexport) uint64_t get_changed_signal(uint64_t idx)
{
    return events.chagned_data[idx];
}

extern "C" _declspec(dllexport) uint64_t step(uint64_t n) // reutrn number of triggered signals
{
    auto t{std::chrono::system_clock::now()};
    for (uint64_t i{}; i < n; ++i)
    {
        events.chagned_data.clear();
        events.step();
        if (!events.chagned_data.empty())
            break;
    }
    std::chrono::duration<double> dt{std::chrono::system_clock::now() - t};
    double time_cost = dt.count(); // second
    return events.chagned_data.size();
}

extern "C" _declspec(dllexport) void init(uint64_t init_type) // push all nodes.
{
    init_data();
    for (uint64_t i = 1; i < data_size; i += 2)
    {
        Node **fanouts = (Node **)data_start[i];
        for (; Node *pnode = *fanouts; fanouts++)
        {
            if (!pnode->ready)
            {
                pnode->ready = true;
                events.changed_nodes.push_back(pnode);
            }
        }
    }
}


"""

###############################

"""
data[3] 0 -- 64 -- 128 -- 192
slice [120:130]

read: idx 1, 2 
temp1: data[1] & mask(8) << 56
temp2: data[2] & mask(2) 

out of range should be x: const 



out = self.output._split()
builder = Node()
x = builder.add(xxx)
y = builder.not(x)

static slicing:
    1. get n 
    2. 

dynamic slicing: 
    int temp[] = a, b  # builder.array(a,b)
    out_v, out_x = slice(temp, idx, width)

partial assign: (120:130)
    a, b = (in1, in2) << 56
    mask1, mask2 = xxx 

    target1 = target1 & (!mask1) | a 
    target2 = target2 & (!mask2) | b 

dynamic partial assign:
    n = full target 

XXX typing casting does not  allow down case , safer
    out of range default is 0, not x

"""
