"""
Package devoted to plotting code

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/plotting/__init__.py,v 1.3 2012/01/27 15:07:14 burnett Exp $
Authors:  T. Burnett, M. Kerr, E. Wallace, M. Roth, J. Lande
"""
from importlib import import_module

sed = import_module(__name__ + '.sed')

try:
	tsmap = import_module(__name__ + '.tsmap')
except Exception:
	tsmap = None

try:
	counts = import_module(__name__ + '.counts')
except Exception:
	counts = None
