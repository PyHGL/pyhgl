import sys
import pyhgl
from pyhgl.tester import tester

import sys, traceback


filter = '|'.join(sys.argv[1:])
if not filter: filter = '.*'
tester.filter(filter)

from _test import test0_grammar
from _test import test1_config
from _test import test2_basic
from _test import test3_simple 
from _test import test5_SRV32
from _test import test6_AES
from _test import test7_chisel_examples
from _test import test8_basic_simd



tester.summary()

