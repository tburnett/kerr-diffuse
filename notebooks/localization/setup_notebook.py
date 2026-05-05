import os
os.chdir('/home/burnett/work/pixel-table')
from pathlib import Path
import sys

repo_root = Path.cwd().resolve()
if str(repo_root) not in sys.path:
    # print(f'Adding {repo_root} to sys.path')
    sys.path.insert(0, str(repo_root))


from importlib import reload