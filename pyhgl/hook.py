""" 
python import hook: https://www.python.org/dev/peps/pep-0302/
reference: https://github.com/pfalcon/python-imphook
"""

import sys 
import importlib
import traceback
import re
import pyhgl.parser



_drop = re.compile(r'\s*File "<frozen importlib._bootstrap')

class FileLoader_pyh(importlib._bootstrap_external.FileLoader):
 
    def create_module(self, spec):
        """load PyHGL source files(.pyh) and compile
        """
        # print(spec.name, spec.origin)
        # print(self.name, self.path)
        # ast_tree = pyhgl.parser.hgl_parse_file(spec.origin, verbose=True)
        try:
            ast_tree = pyhgl.parser.hgl_parse_file(spec.origin)
        except:
            _stack = [i for i in traceback.format_stack() if not _drop.match(i)]
            _stack[-1] = 'PyHGL Traceback:\n'
            
            etype, value, tb = sys.exc_info()
            estack = traceback.format_exception(etype, value, tb)[1:]
            # clear error info
            if etype is SyntaxError:
                estack = estack[-4:]
                
            s = ['Python Traceback:\n'] + _stack + estack
            print(''.join(s))
            sys.exit(1)
            
        self.code_obj = compile(ast_tree, spec.origin, 'exec')
        
    def exec_module(self, module):
        c, e = self.code_obj, module.__dict__
        try:
            exec(c,e) 
        except:
            _stack = [i for i in traceback.format_stack() if not _drop.match(i)]
            _stack[-1] = 'PyHGL Traceback:\n'
            etype, value, tb = sys.exc_info()
            estack = traceback.format_exception(etype, value, tb)[2:]        
            s = ['Python Traceback:\n'] + _stack + estack
            print(''.join(s))
            sys.exit(1)


def insert_import_hook():
    """insert FileLoader_pyh to the first path_hook found in sys.path_hooks
    """
    for i, path_hook in enumerate(sys.path_hooks):
        if not isinstance(path_hook, type):
            # Assume it's a type wrapped in a closure,
            # as is the case for FileFinder.
            path_hook = type(path_hook("."))
        if path_hook is importlib._bootstrap_external.FileFinder:
            sys.path_hooks.pop(i)
            insert_pos = i
            break
    else:
        insert_pos = 0

    # Mirrors what's done by importlib._bootstrap_external._install(importlib._bootstrap)
    # loaders for different file extensions
    loaders = [(FileLoader_pyh, ['.pyh'])] + importlib._bootstrap_external._get_supported_file_loaders()
    # path_hook closure captures supported_loaders in itself, return FileFinder
    sys.path_hooks.insert(insert_pos, importlib._bootstrap_external.FileFinder.path_hook(*loaders))
    sys.path_importer_cache.clear()

insert_import_hook() 





"""
IPython and jupyter notebook cell_magic support 

usage:
    %%pyh 
    ...
"""
try:
    from IPython import get_ipython
    from IPython.core.magic import  register_cell_magic, needs_local_scope
    
    @register_cell_magic
    @needs_local_scope
    def pyh(line='', cell=None, local_ns=None):
        shell = get_ipython()
        try:
            ast_tree = pyhgl.parser.hgl_parse_string(cell, 'exec')
        except:
            shell.showsyntaxerror()  
            return 
        try:
            code = compile(ast_tree, '<pyhgl_cell>', 'exec')
            exec(code, shell.user_ns, local_ns)
        except:
            shell.showtraceback()
            return 
except:
    pass 

