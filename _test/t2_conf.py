
from pyhgl.logic import *



@conf 
def toplevel():
    para1 = 'para1'
    para2 = 'para2'
    double_width = 3
    
    @conf 
    def Root():
        width = 8 

        @conf 
        def Left():
            width = 4
            width1 = 7 
            
            return locals()

        return locals() 
    
    @conf.always('.*')
    def All():
        double_width = conf.up.double_width * 2
        return locals()
    
    @conf.always
    def LL1():
        para3 = 'para3'
        return locals()
    @conf.always
    def LL2():
        para4 = 'para4'
        return locals()


    return locals()



@module 
def Left(n = 1):
    if n > 1:
        l = Left(n-1)
    return locals()

@module 
def Root(width=4):
    l = Left(3)
    
    ll = Left['LL1','LL2']()

    return locals() 

with Session(conf=toplevel(), verbose_conf=True):
    root = Root()

