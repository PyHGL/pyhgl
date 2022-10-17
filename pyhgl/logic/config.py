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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Type

import sys
import inspect
import re

from pyhgl.array import * 
import pyhgl.logic._session as _session
import pyhgl.logic.utils as utils


"""
construct a tree-like config as global config. 

@conf top:
    conf.p.dispatcher = ...
    clock = ... 
    reset = ... 
    
    width = 10 
    @conf.always('adder.*') add:
        width = conf.up.width * 2 
    @conf.clean('adder.*') add2:
        height = 3
"""



@singleton 
class _conf_always:
    def __call__(self, obj: Union[Callable, str]):
        if isinstance(obj, str):
            def wrapper(f):
                assert callable(f)
                return HGLConf(f, obj, always=True)
            return wrapper 
        elif callable(obj):
            return HGLConf(obj, obj.__name__+'.*', always=True)
        else:
            raise ValueError('incorrect useage of decorator')
        
@singleton 
class _conf_clean:
    def __call__(self, obj: Union[Callable, str]):
        if isinstance(obj, str):
            def wrapper(f):
                assert callable(f)
                return HGLConf(f, obj, inherit=False)
            return wrapper 
        elif callable(obj):
            return HGLConf(obj, obj.__name__+'.*', inherit=False)
        else:
            raise ValueError('incorrect useage of decorator') 

@singleton 
class _conf_sim:
    def __call__(self, f) -> Type:
        return type(f.__name__, (TimingConf,),{'__body__':staticmethod(f)})
        
@singleton
class conf(HGL):
    """ decorator function, return HGLConf
    """
    _sess: _session.Session
    
    always = _conf_always 
    clean = _conf_clean 
    timing = _conf_sim
    
    def __call__(self, obj: Union[Callable, str]):
        if isinstance(obj, str):
            def wrapper(f):
                assert callable(f)
                return HGLConf(f, obj)
            return wrapper 
        
        elif callable(obj):
            return HGLConf(obj, obj.__name__ + '.*')
        
        else:
            raise ValueError('incorrect use of decorator') 
        
    @property
    def up(self) -> Type[ModuleConf]: 
        """ return father module's conf
        """
        position = self._sess.module._position 
        if len(position) > 1:
            return position[-2]._conf 
        else:
            return None
        
    @property 
    def p(self):
        """ return current module's conf
        """
        return self._sess.module
        
        
    
    
class HGLConf(HGL):
    
    def __init__(
        self, 
        f: Callable = lambda : {}, 
        filter: str = '_', 
        *, 
        always = False, 
        inherit = True
    ) -> None:

        self.f = f 
        self.filter = re.compile(filter, flags=re.IGNORECASE) 
        self.always = always 
        self.inherit = inherit 
        # self._arg_names = set(inspect.signature(f).parameters.keys())
    
    
    def __call__(self, *args, **kwargs):
        """ 
        TODO return a new class
        """
        return (self, args, kwargs)
    
    
    def exec(self, args = [], kwargs = {}) -> Tuple[dict, Dict[HGLConf, None]]:
        """
        TODO ignore 'io', 'up', 'self'
        """
        # execute config function
        local_var: Dict[str, Any] = self.f(*args, **kwargs)
            
        subconfigs = {}
        parameters = {}
        
        for k, v in local_var.items(): 
            if isinstance(v, HGLConf):
                subconfigs[v] = None
            elif k[0] != '_':
                parameters[k] = v 
                
        return parameters, subconfigs
    
    def __str__(self):
        func = f"{self.f.__module__}.{self.f.__qualname__}"
        func = re.sub(r'.<locals>','', func) 
        filter = f"'{self.filter.pattern}'"
        
        ret = [func, filter]

        if self.always:
            ret.append('always')
        if self.inherit:
            ret.append('inherit') 
            
        return f"HGLConf({','.join(ret)})"
    
        

class ModuleConf: 
    """ parameters store in class attributes
    """
    clock: Any 
    reset: Any


# ----------- 
# timing 
# -----------

    
class TimingConf(HGL):
    """
    @timing TimingConfig:
        timescale = '10ns' # timeunit: 'ns' timeprecision '10ns'

        Logic = dict( delay = 1 ) 
        Gate = dict( delay = 2 ) 
        
    {
        'timescale': '1ns',
        'Wire': {
            'delay': 1,
            'power': 0
        },
        'Gate': {
            'delay': 1,
            'power':0
        },
        'Clock': {
            'period': 100
        }
    }
    """
    def __init__(self, *args, **kwargs):
        
        f_locals: dict = self.__body__(*args, **kwargs)
        if timescale:=f_locals.get('timescale'):
            self.timescale, temp = utils.quantity(timescale) 
            self.time_info = (round(temp[0]), temp[1])
        else:
            self.timescale: float = 1e-9
            self.time_info: Tuple[int, str] = (1, 'ns')
        self._values = self._get_quantity(f_locals)

    def get(self, key: str):
        return self._values.get(key)
        
    def __body__(*args, **kwargs): 
        return {} 
    
    def _get_timestep(self, v: str) -> int:
        _v, _ = utils.quantity(v)
        return round(_v / self.timescale)
    
    def _get_quantity(self, v: Dict[str, Union[int, float, str]]):
        """ 
        convert quantities with unit to desired value 
        
        time -> int(time/timescale)
        """
        ret = {}
        if isinstance(v, Array):
            items = v._items()
        else:
            items = v.items() 
            
        for key, value in items:
            if isinstance(value, (dict, Array)):
                ret[key] = self._get_quantity(value)
            elif isinstance(value, (int, float)):
                ret[key] = value
            elif isinstance(value, str):
                _v, _ = utils.quantity(value)
                ret[key] = round(_v/self.timescale)
        return ret
                
    def __bool__(self):
        return True

    def __str__(self):
        return str(self._values)
    
    
    

