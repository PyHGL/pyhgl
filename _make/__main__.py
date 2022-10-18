 
import pyhgl.tester.utils as utils
 
import os 
import sys
import subprocess 
import shutil



def make_parser():
    try:
        import pegen as _
    except:
        raise Exception('module pegen not found') 
    
    gram_file = utils.relative_path('../pyhgl/parser/hgl_parser.gram', check_exist=True)
    target = utils.relative_path('../pyhgl/parser/hgl_parser.py', check_exist=True)     
    
    print(f"{utils._yellow('making parser:')} {gram_file} -> {target}") 
    utils.run_python(f"-m pegen {gram_file} -q -o {target}" ) 
    
def make_doc():
    try:
        import sphinx as _ 
        import myst_parser as _ 
        import linkify_it as _ 
        import sphinx_multiversion as _
    except:
        raise Exception('python -m pip install -U sphinx, sphinx_multiversion, myst_parser, linkify-it-py')
    
    source = utils.relative_path('../documents', check_exist=True)
    target = utils.relative_path('../docs', check_exist=True)
    shutil.rmtree(target)
    os.mkdir(target)
    
    print(f"{utils._yellow('making docs:')} {source} -> {target}")
    utils.run_python(f"-m sphinx -q -b html {source} {target}" ) 
    
    
make_parser() 
make_doc()