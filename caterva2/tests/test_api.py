###############################################################################
# Caterva2 - On demand access to remote Blosc2 data repositories
#
# Copyright (c) 2023 The Blosc Developers <blosc@blosc.org>
# https://www.blosc.org
# License: GNU Affero General Public License v3.0
# See LICENSE.txt for details about copyright and rights to use.
###############################################################################
import pathlib

import httpx

import blosc2
import pytest

import caterva2 as cat2
import numpy as np

from .services import TEST_PUBLISHED_ROOT as published_root
from .. import api_utils


def my_path(dspath, slice_):
    slice_ = api_utils.slice_to_string(slice_)
    if slice_:
        suffix = dspath.suffix
        dspath = dspath.with_suffix('')
        dspath = pathlib.Path(f'{dspath}[{slice_}]{suffix}')
    return dspath


def my_urlpath(ds, slice_):
    path = pathlib.Path(ds.path)
    suffix = path.suffix
    slice2 = api_utils.slice_to_string(slice_)
    if slice2 or suffix not in {'.b2frame', '.b2nd'}:
        path = 'downloads' / path.with_suffix('')
        slice3 = f"[{slice2}]" if slice2 else ""
        path = pathlib.Path(f'{path}{slice3}{suffix}')
    path = f"http://{cat2.sub_host_default}/files/{path}"
    return path


def test_roots(services):
    roots = cat2.get_roots()
    assert roots[published_root]['name'] == published_root
    assert roots[published_root]['http'] == cat2.pub_host_default

def test_root(services):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    assert myroot.name == published_root
    assert myroot.host == cat2.sub_host_default

def test_list(services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    example = examples_dir
    nodes = set(str(f.relative_to(str(example))) for f in example.rglob("*") if f.is_file())
    assert set(myroot.node_list) == nodes

def test_file(services):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    file = myroot['README.md']
    assert file.name == 'README.md'
    assert file.host == cat2.sub_host_default


def test_dataset_frame(services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['ds-hello.b2frame']
    assert ds.name == 'ds-hello.b2frame'
    assert ds.host == cat2.sub_host_default

    example = examples_dir / ds.name
    a = blosc2.open(example)[:]
    assert ord(ds[1]) == a[1]  # TODO: why do we need ord() here?
    assert ds[:1] == a[:1]
    assert ds[0:10] == a[0:10]
    assert ds[10:20] == a[10:20]
    assert ds[:] == a
    assert ds[10:20:1] == a[10:20:1]
    # We don't support step != 1
    with pytest.raises(Exception) as e_info:
        np.testing.assert_array_equal(ds[::2], a[::2])
        assert ds[::2] == a[::2]
        assert str(e_info.value) == 'Only step=1 is supported'

def test_dataset_1d(services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['ds-1d.b2nd']
    assert ds.name == 'ds-1d.b2nd'
    assert ds.host == cat2.sub_host_default

    example = examples_dir / ds.name
    a = blosc2.open(example)[:]
    np.testing.assert_array_equal(ds[1], a[1])
    np.testing.assert_array_equal(ds[:1], a[:1])
    np.testing.assert_array_equal(ds[0:10], a[0:10])
    np.testing.assert_array_equal(ds[10:20], a[10:20])
    np.testing.assert_array_equal(ds[:], a)
    np.testing.assert_array_equal(ds[10:20:1], a[10:20:1])
    # We don't support step != 1
    with pytest.raises(Exception) as e_info:
        np.testing.assert_array_equal(ds[::2], a[::2])
        assert str(e_info.value) == 'Only step=1 is supported'


@pytest.mark.parametrize("name", ['dir1/ds-2d.b2nd', 'dir2/ds-4d.b2nd'])
def test_dataset_nd(name, services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot[name]
    example = examples_dir / ds.name
    a = blosc2.open(example)[:]
    np.testing.assert_array_equal(ds[1], a[1])
    np.testing.assert_array_equal(ds[:1], a[:1])
    np.testing.assert_array_equal(ds[0:10], a[0:10])
    # The next is out of bounds, but it is supported (by numpy too)
    np.testing.assert_array_equal(ds[10:20], a[10:20])
    np.testing.assert_array_equal(ds[:], a)
    np.testing.assert_array_equal(ds[1:5:1], a[1:5:1])
    # We don't support step != 1
    with pytest.raises(Exception) as e_info:
        np.testing.assert_array_equal(ds[::2], a[::2])
        assert str(e_info.value) == 'Only step=1 is supported'

@pytest.mark.parametrize("name", ['ds-1d.b2nd', 'dir1/ds-2d.b2nd'])
def test_download_b2nd(name, services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot[name]
    path = ds.download()
    assert path == ds.path

    # Data contents
    example = examples_dir / name
    a = blosc2.open(example)
    b = blosc2.open(path)
    np.testing.assert_array_equal(a[:], b[:])

    # Using 2-step download
    urlpath = ds.get_download_url()
    assert urlpath == my_urlpath(ds, None)
    data = httpx.get(urlpath)
    assert data.status_code == 200
    b = blosc2.ndarray_from_cframe(data.content)
    np.testing.assert_array_equal(a[:], b[:])

# TODO: test slices that exceed the array dimensions
@pytest.mark.parametrize("slice_", [slice(1,10), slice(4,8), slice(None), 1])
@pytest.mark.parametrize("name", ['ds-1d.b2nd', 'dir1/ds-2d.b2nd'])
def test_download_b2nd_slice(slice_, name, services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot[name]
    path = ds.download(slice_)
    dspath = my_path(ds.path, slice_)
    assert path == dspath

    # Data contents
    example = examples_dir / name
    a = blosc2.open(example)
    b = blosc2.open(path)
    if isinstance(slice_, int):
        np.testing.assert_array_equal(a[slice_], b[()])
    else:
        np.testing.assert_array_equal(a[slice_], b[:])

    # Using 2-step download
    urlpath = ds.get_download_url(slice_)
    path = my_urlpath(ds, slice_)
    assert urlpath == path
    data = httpx.get(urlpath)
    assert data.status_code == 200
    b = blosc2.ndarray_from_cframe(data.content)
    if isinstance(slice_, int):
        np.testing.assert_array_equal(a[slice_], b[()])
    else:
        np.testing.assert_array_equal(a[slice_], b[:])

def test_download_b2frame(services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['ds-hello.b2frame']
    path = ds.download()
    assert path == ds.path

    # Data contents
    example = examples_dir / ds.name
    a = blosc2.open(example)
    b = blosc2.open(path)
    assert a[:] == b[:]

    # Using 2-step download
    urlpath = ds.get_download_url()
    assert urlpath == f"http://{cat2.sub_host_default}/files/{ds.path}"
    data = httpx.get(urlpath)
    assert data.status_code == 200
    b = blosc2.schunk_from_cframe(data.content)
    assert a[:] == b[:]

# TODO: add an integer slice test when it is supported in blosc2
@pytest.mark.parametrize("slice_", [slice(1,10), slice(15,20), slice(None)])
def test_download_b2frame_slice(slice_, services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['ds-hello.b2frame']
    path = ds.download(slice_)
    dspath = my_path(ds.path, slice_)
    assert path == dspath

    # Data contents
    example = examples_dir / ds.name
    a = blosc2.open(example)
    b = blosc2.open(path)
    assert a[slice_] == b[:]

    # Using 2-step download
    urlpath = ds.get_download_url(slice_)
    path = my_urlpath(ds, slice_)
    assert urlpath == path
    data = httpx.get(urlpath)
    assert data.status_code == 200
    b = blosc2.schunk_from_cframe(data.content)
    assert a[slice_] == b[:]


def test_index_regular_file(services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['README.md']

    # Data contents
    example = examples_dir / ds.name
    a = open(example).read().encode()
    assert ds[:] == a[:]
    assert ord(ds[1]) == a[1]     # TODO: why do we need ord() here?
    assert ds[:1] == a[:1]
    assert ds[0:10] == a[0:10]
    assert ds[10:20] == a[10:20]


def test_download_regular_file(services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['README.md']
    path = ds.download()
    assert path == ds.path

    # Data contents
    example = examples_dir / ds.name
    a = open(example).read()
    b = open(path).read()
    assert a[:] == b[:]

    # Using 2-step download
    urlpath = ds.get_download_url()
    assert urlpath == f"http://{cat2.sub_host_default}/files/downloads/{ds.path}"
    data = httpx.get(urlpath)
    assert data.status_code == 200
    b = data.content.decode()
    assert a[:] == b[:]

@pytest.mark.parametrize("slice_", [slice(1,10), slice(15,20), slice(None)])
def test_download_regular_file_slice(slice_, services, examples_dir):
    myroot = cat2.Root(published_root, host=cat2.sub_host_default)
    ds = myroot['README.md']
    path = ds.download(slice_)
    dspath = my_path(ds.path, slice_)
    assert path == dspath

    # Data contents
    example = examples_dir / ds.name
    a = open(example).read()
    b = open(path).read()
    assert a[slice_] == b[:]

    # Using 2-step download
    urlpath = ds.get_download_url(slice_)
    path = my_urlpath(ds, slice_)
    assert urlpath == path
    data = httpx.get(urlpath)
    assert data.status_code == 200
    b = data.content.decode()
    assert a[slice_] == b[:]
