 
import os 
import shutil 

from .utils import *


def make_requires():
    """ install setuptools wheel twine pegen sphinx sphinx_rtd_theme sphinx_multiversion myst_parser linkify-it-py"""
    python("-m pip install -U setuptools wheel twine pegen")
    python("-m pip install -U sphinx sphinx_rtd_theme sphinx_multiversion myst_parser linkify-it-py")

def make_parser():
    """ generate pyhgl parser """
    gram_file = relative_path('../pyhgl/parser/hgl_parser.gram', check_exist=True)
    target = relative_path('../pyhgl/parser/hgl_parser.py', check_exist=True)     
    print(f"{gram_file} -> {target}") 
    python(f"-m pegen {gram_file} -q -o {target}" ) 


def make_install():
    """ install pyhgl package as editable """
    python('-m build --wheel')
    python('-m pip install -e .')


def make_release():
    """ upload to pypi.org """
    python('-m twine upload dist/*')


def make_doc():
    """ generate pyhgl doc """
    try:
        import sphinx, myst_parser, linkify_it, sphinx_multiversion
    except:
        raise Exception('make_requires')
    
    source = relative_path('../documents', check_exist=True)
    target = relative_path('../docs', check_exist=True)
    shutil.rmtree(target)
    os.mkdir(target)
    
    print(f"{source} -> {target}")
    python(f"-m sphinx -q -b html {source} {target}" ) 


finish()

