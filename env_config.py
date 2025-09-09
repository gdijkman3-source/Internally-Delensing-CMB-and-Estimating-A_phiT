import os

# Centralized environment variable definitions
# Modify these paths at one place as needed
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Default paths (override via existing environment variables if set)
PLENS = os.environ.get('PLENS', '/mnt/d/THESIS/PLENS')
INPUT = os.environ.get('INPUT', '/mnt/d/THESIS/SIMS')
PARAMS = os.environ.get('PARAMS', os.path.join(PROJECT_ROOT, 'input'))
KFIELD = os.environ.get('KFIELD', '/mnt/d/THESIS/LENSING MAPS')

# Set the environment variables for the project
os.environ['PLENS'] = PLENS
os.environ['INPUT'] = INPUT
os.environ['PARAMS'] = PARAMS
os.environ['KFIELD'] = KFIELD
