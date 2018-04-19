#!/usr/bin/env python3

import sys
import os
from distutils.core import setup
from subprocess import Popen, PIPE

def get_version():
    proc = Popen(['git', 'describe', '--tags', '--long', '--always', '--dirty'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('utf8').strip()
    stderr = stderr.decode('utf8').strip()
    if proc.returncode != 0:
        sys.exit('Error while getting the git version.\n%s' % stderr)
    return stdout

def set_version(directory, version):
    filename = os.path.join(directory, 'version.py')
    with open(filename, 'w') as f:
        f.write('__version__ = "%s"' % version)

if __name__ == '__main__':
    set_version('pytree', get_version())
    setup(name='pytree',
          version='0.0.1',
          description='Simple implementation of model trees.',
          author='Tom Cornebize',
          author_email='tom.cornebize@gmail.com',
          packages=['pytree']
         )
