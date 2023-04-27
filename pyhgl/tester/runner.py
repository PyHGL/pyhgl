from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import re
import sys
import time
import types
import inspect
import traceback 

from .utils import _red, _green, _blue, _yellow, _fill_terminal, caller_filename


class _TestTreeNode:
    def __init__(self, name: str = 'top', level: int = 0) -> None:
        self.nodes: Dict[str, _TestTreeNode] = {}    # subtrees
        self.name: str = name 
        self.level = level  # deepth 
        
        self.exception: str = None  # exception msg
        self.msg: str = ''          # other msg
         
        self.assertions_passed: List[_Assertion] = [] 
        self.assertions_failed: List[_Assertion] = [] 
        # not all assertions are stored
        self.count_passed = 0
        self.count_failed = 0
        # time cost 
        self.t = time.time() 
        self.t_cost = 0 

        # assertion 
        self.EQ = EQ(self)

    def get_or_register(self, keys: Tuple[str]) -> _TestTreeNode:
        """ if key not in subtrees, generate new node
        """
        key = keys[0]
        if key in self.nodes:
            next_node = self.nodes[key]
        else:
            next_node = _TestTreeNode(name=key, level=self.level+1) 
            self.nodes[key] = next_node

        if len(keys) == 1:
            return next_node
        else:
            return next_node.get_or_register(keys[1:]) 
        
    def count_results(self):
        count_passed = self.count_passed 
        count_failed = self.count_failed 
        for node in self.nodes.values():
            _passed, _failed = node.count_results()
            count_passed += _passed 
            count_failed += _failed 
        return count_passed, count_failed
        
    def finish(self, msg: str = ''):
        self.t_cost = time.time() - self.t 
        self.msg = msg
        
    def __str__(self): 
        if self.level == 0:
            ret = '\n'.join(str(i) for i in self.nodes.values())  
            n_passed, n_failed = self.count_results() 
            total_t = time.time() - self.t 
            return f'{ret}\n\n{_green(n_passed)} passed, {_red(n_failed)} failed, time: {total_t:.4f}s\n{_fill_terminal("━", "━")}\n'  
        elif self.level == 1:
            ret = '\n'.join(str(i) for i in self.nodes.values()) 
            return f'{_fill_terminal("━", "━")}\n{_yellow(self.name)}\n{ret}'
        else:
            n_passed = _green(f'{self.count_passed:>8}') 
            n_failed = _red(f'{self.count_failed:>8}') 
            prefix = (self.level-1) * '  '
            ret = [f'{prefix}{_blue(self.name):<70}{n_passed} passed{n_failed} failed{self.t_cost:>10.4f}s  {self.msg}'] 
            for i in self.assertions_passed:
                ret.append(f'{prefix}│ {i}')
            for i in self.assertions_failed:
                ret.append(f'{prefix}│ {i}') 
            if self.exception: 
                ret.append(prefix + prefix.join(self.exception.splitlines(keepends=True))) 
            ret.extend(str(i) for i in self.nodes.values())
            return '\n'.join(ret)
        
        
    def Assert(self, v: bool, *args):
        _AssertTrue(v, self) 

    def AssertEq(self, a: Any, b:Any, *args):
        _AssertEq((a,b), self)


class EQ: 
    def __init__(self, node) -> None:
        self.node = node

    def __iadd__(self, v: Tuple[Any, Any]):
        # skip if not inside test function
        _AssertEq(v, self.node) 
        return self 

_root = _TestTreeNode()

class _Assertion:
    pass 

class _AssertEq(_Assertion):
    
    def __init__(self, v: Tuple[Any, Any], node: _TestTreeNode, level: int = 2, msg: str = ''):
        a, b = v
        # should support ==, otherwise will stop testcase
        passed = a == b  
        if passed:
            node.count_passed += 1 
            return 
        # if failed, store information
        node.count_failed += 1
        node.assertions_failed.append(self)
        # assertion msg
        n = node.count_failed + node.count_passed - 1
        r = f"assertion {n:<6} failed   because {repr(a)} != {repr(b)}"
        frame,filename,line_number,function_name,lines,index = inspect.stack()[level]
        self.msg = f"{filename}:{line_number:<15}{r}{msg}"

    def __str__(self) -> str:
        return self.msg        

class _AssertTrue(_Assertion):

    def __init__(self, v: bool, node: _TestTreeNode, level: int = 2):
        self.msg = ''
        passed = bool(v)  
        if passed:
            node.count_passed += 1 
            return 
        # if failed, store information
        node.count_failed += 1
        node.assertions_failed.append(self)

        n = node.count_failed + node.count_passed - 1
        r = f"assertion {n:<6} failed"
        frame,filename,line_number,function_name,lines,index = inspect.stack()[level]
        self.msg = f"{filename}:{line_number:<15}{r}"

    def __str__(self) -> str:
        return self.msg    


   



class _Tester:
    """ decorator
    """

    _global_enable: bool = True
    _global_filter: re.Pattern = None 
    _stack: List[str] = []

    def __init__(
        self, 
        enable: bool = True, 
        debug: bool = False,
        cond: Any = True
    ) -> None:
        """
        filter: regexp filter of function name 
        enable: if False, skip this test case   
        debug: if True, raise Exception
        """
        self._enable = enable 
        self._debug = debug
        self._cond = cond

    def __call__(self, f: types.FunctionType) -> None:
        # disable
        if (
            not self._global_enable or  
            not self._enable or 
            not self._cond or
            not inspect.isfunction(f)
        ):
            return 
        # new test node
        if self._stack: 
            self._stack.append(f.__name__) 
        else:
            self._stack.append(caller_filename())
            self._stack.append(f.__name__) 
        # apply filter
        if self._global_filter is not None and not self._global_filter.search('/'.join(self._stack)):
            pass 
        else:
            # exec
            node = _root.get_or_register(self._stack)  
            # debug mode
            if self._debug:
                f(node) 
            else:
                try:
                    f(node)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()  
                    e = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)[2:]) 
                    node.exception = _red('│ ') + _red('\n│ ').join(e.splitlines())
            node.finish()
        # pop stack 
        if len(self._stack) == 2:
            self._stack.clear()
        else:
            self._stack.pop()
        return 

    def copy(self) -> _Tester:
        ret = object.__new__(_Tester)
        ret.__dict__.update(self.__dict__)
        return ret

    def when(self, cond) -> _Tester:
        """
        @tester.when(cond) ...
        """
        ret = self.copy()
        ret._cond = cond 
        return ret

    @property 
    def debug(self) -> _Tester:
        """ debug mode, exit when exception
        @tester.debug ...
        """
        ret = self.copy()
        ret._debug = True 
        return ret
    
    @property 
    def disable(self) -> _Tester:
        """ skip mode
        """
        ret = self.copy()
        ret._enable = False 
        return ret

    def filter(self, s) -> None:
        """ set a global filter that matches case name
        tester.filter('adder.*')
        """
        _Tester._global_filter = re.compile(s)

    def close(self):
        """ disable all test cases
        tester.close()
        """
        _Tester._global_enable = False
        
    def summary(self):
        print(_root)
        
tester: _Tester =  _Tester()

    
