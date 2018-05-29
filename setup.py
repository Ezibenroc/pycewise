#!/usr/bin/env python3

import sys
from distutils.core import setup
import subprocess

VERSION = '0.0.0'


def run(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        sys.exit('Error with the command %s.\n' % ' '.join(args))
    return stdout.decode('ascii').strip()


def git_version():
    return run(['git', 'rev-parse', 'HEAD'])


def git_tag():
    return run(['git', 'describe', '--always', '--dirty'])


def write_version(filename, version_dict):
    with open(filename, 'w') as f:
        for version_name in version_dict:
            f.write('%s = "%s"\n' % (version_name, version_dict[version_name]))


if __name__ == '__main__':
    write_version('pytree/version.py', {
            '__version__': VERSION,
            '__git_version__': git_version(),
        })
    setup(name='pytree',
          version=VERSION,
          description='Simple implementation of model trees.',
          author='Tom Cornebize',
          author_email='tom.cornebize@gmail.com',
          packages=['pytree'],
          )
