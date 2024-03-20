"""Caterva2 services for tests.

This ensures that Caterva2 broker, publisher and subscriber services are
running before proceeding to tests.  It has three modes of operation:

- Standalone script: when run as a script, it starts the services as children
  and makes sure that they are available to other local programs.  If given an
  argument, it uses it as the directory to store state in; otherwise it uses
  the value in `DEFAULT_STATE_DIR`.  If the directory does not exist, it is
  created and populated with example datasets.  If a second argument is given,
  it is taken as the source of example datasets, instead of those from the
  source distribution.

  Terminating the program stops the services.

  Usage example::

      $ cd Caterva2
      $ python -m caterva2.tests.services &  # state in ``_caterva2``
      [3] 12345
      $ pytest
      $ kill %3

- pytest fixture with external services: when using `services()` as a fixture,
  it checks that the services are available to other local programs.  It does
  not tamper with the state directory nor stop the services when tests finish.

  Usage example: same as above (but on the pytest side).

- pytest fixture with managed services: if the environment variable
  ``CATERVA2_USE_EXTERNAL`` is set to 1, the `services()` fixture uses
  external services; otherwise, it takes care of starting the services as
  children and making sure that they are available to other local programs.
  It also uses the value in `TEST_STATE_DIR` as the directory to store state
  in.  If the directory exists, it is removed first. Then the directory is
  created and populated with the example files from the source distribution.
  When tests finish, the services are stopped.

  Usage example::

      $ cd Caterva2
      $ env CATERVA2_USE_EXTERNAL=1 pytest  # state in ``_caterva2_tests``
"""

import collections
import itertools
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import httpx
import pytest

import caterva2 as cat2

from pathlib import Path


DEFAULT_STATE_DIR = '_caterva2'
TEST_STATE_DIR = DEFAULT_STATE_DIR + '_tests'
TEST_DEFAULT_ROOT = 'foo'
TEST_CATERVA2_ROOT = TEST_DEFAULT_ROOT


local_port_iter = itertools.count(8100)


def service_ep_getter(first):
    def get_service_ep():
        nonlocal first
        if first is not None:
            ep, first = first, None
            return ep
        return 'localhost:%d' % next(local_port_iter)
    return get_service_ep


get_bro_ep = service_ep_getter(cat2.bro_host_default)
get_pub_ep = service_ep_getter(cat2.pub_host_default)
get_sub_ep = service_ep_getter(cat2.sub_host_default)


def make_get_http(host, path='/'):
    def check():
        url = f'http://{host}{path}'
        try:
            r = httpx.get(url, timeout=0.5)
            return r.status_code == 200
        except httpx.ConnectError:
            return False
    check.__name__ = f'get_http_{host}'  # more descriptive
    check.host = host  # to get final value
    return check


def http_service_check(conf, conf_sect, def_host, path):
    return make_get_http(conf.get(f'{conf_sect}.http', def_host),
                         path)


def bro_check(conf):
    return http_service_check(conf, 'broker',
                              get_bro_ep(), '/api/roots')


def pub_check(id, conf):
    return http_service_check(conf, f'publisher.{id}',
                              get_pub_ep(), '/api/list')


def sub_check(conf):
    return http_service_check(conf, 'subscriber',
                              get_sub_ep(), '/api/roots')


TestRoot = collections.namedtuple('TestRoot', ['name', 'source'])


class Services:
    def __init__(self):  # mostly to appease QA
        pass


class ManagedServices(Services):
    def __init__(self, state_dir, reuse_state=True,
                 roots=None, configuration=None):
        super().__init__()

        self.state_dir = Path(state_dir).resolve()
        self.reuse_state = reuse_state
        self.roots = list(roots)
        self.configuration = configuration

        self._procs = {}
        self._endpoints = {}
        self._setup_done = False

    def _start_proc(self, name, *args, check=None):
        if check is not None and check():
            raise RuntimeError(
                f"check for service \"{name}\" succeeded before start"
                f" (external service running?): {check.__name__}")

        self._procs[name] = subprocess.Popen(
            [sys.executable,
             '-m' + f'caterva2.services.{name[:3]}',
             '--statedir=%s' % (self.state_dir / name),
             *args])

        if check is None:
            return
        self._endpoints[name] = check.host

        start_timeout_secs = 10
        start_sleep_secs = 1
        for _ in range(int(start_timeout_secs / start_sleep_secs)):
            time.sleep(start_sleep_secs)
            if check():
                break
        else:
            raise RuntimeError(
                f"service \"{name}\" failed to become available"
                f" after {start_timeout_secs:d} seconds")

    def _get_data_path(self, root):
        return self.state_dir / f'data.{root.name}'

    def _setup(self):
        if self._setup_done:
            return

        if not self.reuse_state and self.state_dir.is_dir():
            shutil.rmtree(self.state_dir)
        self.state_dir.mkdir(exist_ok=True)

        for root in self.roots:
            data_path = self._get_data_path(root)
            if root.source.is_dir():
                if not data_path.exists():
                    shutil.copytree(root.source, data_path, symlinks=True)
                data_path.mkdir(exist_ok=True)
            elif not data_path.exists():
                shutil.copy(root.source, data_path)

        self._setup_done = True

    def start_all(self):
        self._setup()

        self._start_proc('broker', check=bro_check(self.configuration))
        for root in self.roots:
            self._start_proc(f'publisher.{root.name}',
                             root.name, self._get_data_path(root),
                             check=pub_check(root.name, self.configuration))
        self._start_proc('subscriber', check=sub_check(self.configuration))

    def stop_all(self):
        for proc in self._procs.values():
            try:
                os.kill(proc.pid, signal.SIGTERM)
                time.sleep(1)
                os.kill(proc.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass

    def wait_for_all(self):
        for proc in self._procs.values():
            proc.wait()

    def get_endpoint(self, service):
        return self._endpoints.get(service)


class ExternalServices(Services):
    def __init__(self, roots=None, configuration=None):
        super().__init__()
        self.roots = list(roots)
        self.configuration = conf = configuration

        self._checks = checks = {}
        checks['broker'] = bro_check(conf)
        for root in roots:
            checks[f'publisher.{root.name}'] = pub_check(root.name, conf)
        checks['subscriber'] = sub_check(conf)

    def start_all(self):
        failed = [check.__name__ for check in self._checks.values()
                  if not check()]
        if failed:
            raise RuntimeError("failed checks for external services: "
                               + ' '.join(failed))

    def stop_all(self):
        pass

    def wait_for_all(self):
        pass

    def get_endpoint(self, service):
        if service not in self._checks:
            return None
        return self._checks[service].host


@pytest.fixture(scope='session')
def services(examples_dir, configuration):
    # TODO: Consider using a temporary directory to avoid
    # polluting the current directory with test files
    # and tests being influenced by the presence of a configuration file.
    root = TestRoot(TEST_CATERVA2_ROOT, examples_dir)
    srvs = (ExternalServices(roots=[root],
                             configuration=configuration)
            if os.environ.get('CATERVA2_USE_EXTERNAL', '0') == '1'
            else ManagedServices(TEST_STATE_DIR, reuse_state=False,
                                 roots=[root],
                                 configuration=configuration))
    try:
        srvs.start_all()
        yield srvs
    finally:
        srvs.stop_all()
    srvs.wait_for_all()


def main():
    from . import files, conf

    root = TestRoot(TEST_DEFAULT_ROOT, files.get_examples_dir())
    if '--help' in sys.argv:
        print(f"Usage: {sys.argv[0]} [STATE_DIRECTORY=\"{DEFAULT_STATE_DIR}\" "
              f"[ROOT=\"{root.name}={root.source}\"]]")
        return

    state_dir = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_STATE_DIR
    if len(sys.argv) >= 3:
        rname, rsource = re.match(r'(?:(.+?)=)?(.+)',  # ``[name=]source``
                                  sys.argv[2]).groups()
        root = TestRoot(rname if rname else TEST_DEFAULT_ROOT,
                        Path(rsource))

    # TODO: Consider allowing path to configuration file, pass here.
    configuration = conf.get_configuration()
    srvs = ManagedServices(state_dir, reuse_state=True,
                           roots=[root],
                           configuration=configuration)
    try:
        srvs.start_all()
        srvs.wait_for_all()
    finally:
        srvs.stop_all()


if __name__ == '__main__':
    main()
