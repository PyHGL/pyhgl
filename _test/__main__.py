import pyhgl
from pyhgl.tester import tester

import sys, traceback

_code = ';'.join(sys.argv[1:])
_globals = {}
_locals = {}
try:
    exec(_code, _globals, _locals)
except:
    traceback.print_exc()
    print('error when exec: ', _code)

if f:=_locals.get('filter'):
    assert isinstance(f, str)
    tester.filter(f)

from _test import test0_grammar
from _test import test1_config
from _test import test2_basic
from _test import test3_simple 
from _test import test5_SRV32
from _test import test6_AES
from _test import test7_chisel_examples
from _test import test8_basic_simd
# from . import test2_assign
# from . import test3_async
# from . import test4_riscv 
# from . import test5_picorv32
# from . import t2_conf
# from . import t4_comb
# from . import t5_cond
# from . import t6_comb_reg
# from . import t7_gate 


tester.summary()

