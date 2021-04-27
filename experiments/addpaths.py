import os
import sys

def add2path(path):
    """Add path to system path"""
    abspath = os.path.abspath(path)
    if abspath not in sys.path: sys.path.append(abspath)


def addpaths():
    """Add all custom .py file paths"""
    add2path(os.path.join('..', '..', 'solver'))
    add2path(os.path.join('..', '..', 'generator'))
    add2path(os.path.join('..', '..', 'utils'))
    add2path(os.path.join('..', '..', 'models'))

    for path in os.listdir(os.path.join('..', '..', 'models')):
        fullpath = os.path.join('..', '..', 'models', path)
        if os.path.isdir(fullpath) and path.startswith('cython'):
            add2path(fullpath)
          
    print('Custom paths added to system path.')
          
addpaths()

    
    