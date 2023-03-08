from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import re
import sys
import time
import types
import inspect
import traceback 

from .utils import _red, _green, _blue, _yellow, _fill_terminal


class _Property:
    
    def __init__(self, v):              self.__v = v 
    def __set__(self, obj, v):          self.__v = v
    def __get__(self, obj, cls=None):   return self.__v       


class _Test:
    # {filename:[testcases]}
    _results: Dict[str, List[_TestCase]] = _Property({})
    # testcase1.testcase2...
    _stack: List[_TestCase] = _Property([])    

class _Assertion(_Test):
    pass 

class _TestCase(_Test):
    """ test cases are functions in test file 
    """
    def __init__(self, f: types.FunctionType, file: str, name: str) -> None:
        # test function
        self.f = f
        self.file = file
        self.name = name
        # record exception trace info
        self.exception: str = None 
        # a testcase contains 0 or 1 or more assertions
        self.assertions_passed: List[_Assertion] = [] 
        self.assertions_failed: List[_Assertion] = [] 
        # not all assertions are stored
        self.count_passed = 0
        self.count_failed = 0
        # time cost 
        self.t = 0 
                 
    def exec(self):
        # enter
        self._stack.append(self)
        curr_time = time.time()
        try:
            # run test function
            self.f()
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()  
            e = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)[2:]) 
            self.exception = _red('\n    │').join([_red('    │interrupt:')] + e.splitlines())
        self.t = time.time() - curr_time
        # exit
        self._stack.pop()
        
    def __str__(self, prefix = '') -> str:
        n_passed = _green(f'{self.count_passed:<6}') 
        n_failed = _red(f'{self.count_failed:<6}')
        ret = [f'{prefix}{_blue(self.name):<60}{n_passed}passed       {n_failed}failed       in {self.t:>8.4f}s\n'] 
                
        for i in self.assertions_passed:
            ret.append(f'{prefix}{i}')
        for i in self.assertions_failed:
            ret.append(f'{prefix}{i}') 

        if self.exception:
            ret.append(self.exception) 
            ret.append('\n')   
        
        return ''.join(ret)
         
         
def singleton(cls):
    return cls()


class _Runtest(_Test):
    """ decorator
    """

    _global_enable: bool = True
    _global_filter: re.Pattern = None

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

    def __call__(self, f: types.FunctionType) -> types.FunctionType:
        
        if (
            not self._global_enable or 
            not self._enable or 
            not self._cond or
            not inspect.isfunction(f)
        ):
            return f
        
        if self._stack:
            # nested test function
            file = self._stack[-1].file
            name = f'{self._stack[-1].name}.{f.__name__}'
        else: 
            # top level test function, 
            file = re.sub(r'\\', '/', f.__code__.co_filename) 
            name = f.__name__

        # apply filter
        fullname = f'{file}/{name}'
        if self._global_filter is not None and not self._global_filter.search(fullname):
            return f
        
        # debug mode
        if self._debug:
            f()
            return f
        
        # normal mode
        if file not in self._results:
            self._results[file] = [] 
        result = _TestCase(f, file = file, name=name)
        self._results[file].append(result)
        result.exec()
        return f   

    @singleton 
    class EQ(_Test):
        def __iadd__(self, v: Tuple[Any, Any]):
            # skip if not inside test function
            if self._stack:
                _AssertEq(v) 
            return self 

    @singleton 
    class NE(_Test):
        def __iadd__(self, v: Tuple[Any, Any]):
            # skip if not inside test function
            if self._stack:
                # TODO not equal
                _AssertEq(v) 
            return self 

    def copy(self) -> _Runtest:
        ret = object.__new__(_Runtest)
        ret.__dict__.update(self.__dict__)
        return ret

    def when(self, cond) -> _Runtest:
        """
        @tester.when(1) ...
        """
        ret = self.copy()
        ret._cond = cond 
        return ret

    @property 
    def debug(self) -> _Runtest:
        """ debug mode, exit when exception
        """
        ret = self.copy()
        ret._debug = True 
        return ret
    
    @property 
    def disable(self) -> _Runtest:
        """ skip mode
        """
        ret = self.copy()
        ret._enable = False 
        return ret

    def clear(self) -> None:
        """ clear all results
        """
        self._results.clear() 


    def filter(self, s) -> None:
        """ set a global filter that matches case name
        """
        _Runtest._global_filter = re.compile(s)

    def close(self):
        """ disable all test cases
        """
        _Runtest._global_enable = False
        
    def summary(self):
        """ print summary
        """
        print('')
        total_passed = 0 
        total_failed = 0 
        total_t = 0
        for filename, testcases in self._results.items():
            print(_fill_terminal('━', '━'))
            print(' ',_yellow(filename))
            for testcase in testcases:
                total_passed += testcase.count_passed
                total_failed += testcase.count_failed
                total_t += testcase.t
                print(testcase.__str__('  '), end='') 
        print(f'\n{_yellow("tester:")} {_green(total_passed)} passed, {_red(total_failed)} failed, total_time: {total_t:.4f}s\n')  
        
"""
usage:

@tester 
def test_case():
    tester.Assert == 1,0

@tester.debug
def test_case():
    tester.Assert == 1,1

tester.summary()
"""
tester: _Runtest =  _Runtest()

    


class _AssertEq(_Assertion):
    
    def __init__(self, v: Tuple[Any, Any]):
        curr = self._stack[-1]
        a, b = v
        self.msg = ''
        # should support ==, otherwise will stop testcase
        passed = a == b  
        if passed:
            curr.count_passed += 1 
            return 
        
        # if failed, store information
        curr.count_failed += 1
        curr.assertions_failed.append(self)

        r = f"{_red('assertion fail')} because {repr(a)} != {repr(b)}"
        n = curr.count_failed + curr.count_passed - 1
        frame,filename,line_number,function_name,lines,index = inspect.stack()[2]
        self.msg = f"  {filename}:{line_number:<5} {n:<6} {r}\n"

    def __str__(self) -> str:
        return self.msg        
        


