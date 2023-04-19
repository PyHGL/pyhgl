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
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union, Type

import inspect

from pyhgl.array import *
import pyhgl.logic._session as _session
import pyhgl.logic.module_sv as module_sv
import pyhgl.logic._config as hglconfig
import pyhgl.logic.hgl_core as hgl_core


 
def module(f):
    """ PyHGL module decorator
    
    return a subclass of `Module` 

    usage:
        @module name(*args, **kwargs):
            ... 
    or:
        @module name(self, *args, **kwargs):
            ...  

    TODO 
    user defined module:
    with m:
        ...
    run __tail__ when exit context
    """
    args = list(inspect.signature(f).parameters.keys())
    if args and args[0] == 'self':
        return type(f.__name__, (Module,),{'__body__': f})
    else:
        return type(f.__name__, (Module,),{'__body__': staticmethod(f)})  


class MetaModule(type):
    """ syntax setting config id of module
    
    ex. Adder['adder3','xxx'](w=8)
    """
    def __getitem__(self, keys: str):
        for i in keys:
            if not isinstance(i, str):
                raise KeyError(f'invalid module id: {i}')
        m: Module = object.__new__(self)
        m.__head__() 
        if isinstance(keys, str):
            keys = [keys]
        m._ids = list(keys)
        return m 



class Module(Container, metaclass=MetaModule):
    

    __slots__ = '_sess', '__dict__'
    
    def __head__(self):  
        self._sess: _session.Session = HGL._sess 
        if not isinstance(self._sess, _session.Session):
            raise Exception('instantiating module under no Session')
        # prefered name (instance name, ex. adder)
        self._name = self.__class__.__name__.lower() 
        # unique name (declaration name, ex. Adder_1)
        self._unique_name = f'{self.__class__.__name__}_{len(self._sess.verilog.modules)}'
        # position in module tree (include self)
        self._position: List[Module] = self._sess._new_module(self) 
        # submodules 
        self._submodules: Dict[Module, None] = {}
        # identifiers to be matched for configs
        self._ids: List[str] = [self._name]
        # store module level parameters, subclass of `hglconfig.ModuleConf`
        self._conf: Type[hglconfig.ModuleConf] = None
        # lazy-callables for child modules, ordered
        self._subconfigs: Dict[hglconfig.HGLConf, None] = {}
        # module_sv necessary for generating verilog 
        self._module = module_sv.ModuleSV(self) 

        # temp stack
        self._prev: List[Module] = []

    def __conf__(self): 
        up = self._sess.module
        up_conf = up._conf 
            
        # record matched callables
        matched_configs: List[hglconfig.HGLConf] = []
        inherit: bool = True
        for config in up._subconfigs: 
            for _id in self._ids:
                if config.filter.match(_id):
                    matched_configs.append(config) 
                    break
            # config always valid
            if config.always:
                self._subconfigs[config] = None
            if not config.inherit:
                inherit = False

        bases = (up_conf,) if inherit else (hglconfig.ModuleConf,) 
        self._conf = type(f"Conf_{self._unique_name}", bases, {}) 
                    
        # ---------------------------------
        if self._sess.verbose_conf:
            self._sess.print(f"Module: {self._unique_name}", 1)
            self._sess.print(f"id: {', '.join(self._ids)}")
            self._sess.print(f"matched configs: {', '.join(str(i) for i in matched_configs)}")
        # ---------------------------------             
        
        with self:
            paras = {}
            
            for config in matched_configs:
                new_paras, subconfigs = config.exec()
                paras.update(new_paras) 
                self._subconfigs.update(subconfigs)
                
            # ---------------------------------
            if self._sess.verbose_conf:
                self._sess.print(f"subconfigs: {', '.join(str(i) for i in self._subconfigs)}")
                _paras = {k:v for k,v in paras.items() if k != 'dispatcher'}
                self._sess.print(f"new_paras: {_paras}")
            # ---------------------------------
            for k,v in paras.items():
                setattr(self._conf, k, v)
        

    def __body__(*args, **kwargs): 
        '''your hdl module defination'''


    def __tail__(self, f_locals:dict):  
        # delete input driver
        for i in self._module.inputs:
            i._data._module = None

        # record 
        for k, v in f_locals.items():
            if k[0] != '_':  
                self.__dict__[k] = v 
                if isinstance(v, hgl_core.Reader) and (v._data._module is self or v._data._module is None):
                    # update prefered name
                    v._data._name = k
                elif isinstance(v, Module) and v in self._submodules:
                    # update prefered name
                    v._name = k 
                elif isinstance(v, Array):
                    _rename_array(v, k, _module = self)

        # ---------------------------------
        if self._sess.verbose_conf:
            self._sess.print(None, -1)
        # ---------------------------------

    def __new__(cls, *args, **kwargs) -> Module:
        m = object.__new__(cls)
        m.__head__()
        m.__conf__()
        with m:
            f_locals = m.__body__(*args, **kwargs)
            m.__tail__(f_locals)             
        return m 
    
    def __call__(self, *args, **kwargs) -> Module:
        if self._conf is not None:
            raise Exception('non-empty module is not callable')
        self.__conf__() 
        with self:
            f_locals = self.__body__(*args, **kwargs)
            self.__tail__(f_locals)
        return self 
    
    def __enter__(self):
        if HGL._sess is not self._sess:
            raise Exception('invalid session')
        self._prev.append(self._sess.module)
        self._sess.module = self 
        
    def __exit__(self, exc_type, exc_value, traceback): 
        self._sess.module = self._prev.pop()
    
    def _format_slice(self, key: Union[int, str, list]) -> Union[str, Iterable]:
        """ support array-like slicing
        return:
            str:
                single item 
            list|dict :
                multi items
        """
        if isinstance(key, str):
            return self.__dict__, key
        elif isinstance(key, list): 
            return self.__dict__, key
        elif isinstance(key, dict):
            return self.__dict__, key 
        elif key is None:
            return [self], [0] 
        elif key is ...:
            return self.__dict__, {k:k for k in list(self.__dict__.keys())} 
        else:
            raise KeyError(key) 

    def __setitem__(self, keys, value):
        raise AttributeError('__setitem__ invalid for module')
     
    def __getattr__(self, k: str) -> Any:
        try:
            return getattr(self._conf, k)
        except:
            raise AttributeError(f'{k} is neither Module member nor Config parameter for id:{self._ids}')


    def __str__(self) -> str:
        """ return full name, ex. Global.Top.Adder_1
        """
        return '.'.join(i._unique_name for i in self._position)
 


def _rename_array(obj: Union[Array, hgl_core.Reader], name: str, *, _module):
    if isinstance(obj, hgl_core.Reader) and (obj._data._module is _module or obj._data._module is None):
        obj._data._name = name
    elif isinstance(obj, Array): 
        if len(obj._keys()):        # named array
            for k, v in obj._items():
                new_name =  f"{name}_{k}"
                _rename_array(v, new_name, _module=_module)
        else:       # nd-array
            for idx, v in enumerate(obj):
                new_name = f'{name}_{idx}'
                _rename_array(v, new_name, _module=_module)
    else:
        return 
    

class _Ports(HGLFunction):
    
    def __init__(self, direction: str) -> None:
        self.direction = direction 
    
    def __call__(self, s: Union[Array, hgl_core.Reader]) -> Any: 
        s = Signal(s)
        module: Module = self._sess.module 
        assert len(module._position) > 1, 'IO should be called inside a module'

        def make_io(x):
            assert isinstance(x, hgl_core.Reader)  
            assert x._direction == 'inner'
            x._direction = self.direction
            if x._direction == 'input':
                assert x._data.writer is None  
                x._data._module = module._position[-2] 
                module._module.inputs[x] = None
            elif x._direction == 'output':
                x._direction = 'output'  
                module._module.outputs[x] = None 
            elif x._direction == 'inout':  
                # convert to tri state wire
                if x._data._module is None:
                    Wtri(x)
                module._module.inouts[x] = None 
        Map(make_io, s) 
        return s
    
    def __matmul__(self, x):
        return self.__call__(x)
    
    def __rmatmul__(self, x):
        return self.__call__(x)
    
Input = _Ports('input')
Output = _Ports('output')
InOut = _Ports('inout')
Inner = _Ports('inner')
        

# TODO copy value
@singleton 
class CopyIO(HGLFunction):
    """
    copy signals with direction. does not copy default value
    """
    _sess: _session.Session
    
    def __call__(self, x: Union[hgl_core.Reader, Array]) -> Any:   
        x = Signal(x)
        def f(s: hgl_core.Reader):
            ret = s._type()
            if s._direction == 'input': 
                ret = Input(ret)
            elif s._direction == 'output':
                ret = Output(ret)
            elif s._direction == 'inout':
                ret = InOut(ret)
            else:
                raise Exception(f'{x} no direction')
            return ret
        return Map(f, x)
    
    
# TODO Flip each time
@singleton 
class FlipIO(HGLFunction):
    """
    copy signals with flipped direction.  does not copy default value
    """
    
    _sess: _session.Session
    
    def __call__(self, x: Union[hgl_core.Reader, Array]) -> Any:  
        x = Signal(x)
        def f(s: hgl_core.Reader):
            ret = s._type()
            if s._direction == 'input': 
                ret = Output(ret)
            elif s._direction == 'output':
                ret = Input(ret)
            elif s._direction == 'inout':
                ret = InOut(ret)
            else:
                raise Exception(f'{x} no direction')
            return ret
        return Map(f, x)
    

@dispatch('Matmul', Any, [HGLFunction, type(lambda _:_), type])
def _call(x, f):
    """ Signal|Array @ Callable
    """
    return f(x) 



# TODO ConnectIO(*args)  auto io connect