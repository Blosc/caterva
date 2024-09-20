###############################################################################
# Caterva2 - On demand access to remote Blosc2 data repositories
#
# Copyright (c) 2023 ironArray SLU <contact@ironarray.io>
# https://www.blosc.org
# License: GNU Affero General Public License v3.0
# See LICENSE.txt for details about copyright and rights to use.
###############################################################################

import asyncio
import contextlib
import itertools
import logging
import os
import pathlib
import re
import shutil
import string
import tarfile
import typing
import zipfile
from collections.abc import Awaitable, Callable

# FastAPI
from fastapi import Depends, FastAPI, Form, Request, UploadFile, responses
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import fastapi

# Requirements
import blosc2
import dotenv
import furl
import httpx
import markdown
import numexpr as ne
import numpy as np

# Project
from caterva2 import utils, api_utils, models
from caterva2.services import srv_utils
from caterva2.services.subscriber import db, schemas, users


BASE_DIR = pathlib.Path(__file__).resolve().parent

# Set CATERVA2_SECRET=XXX in .env file in working directory
dotenv.load_dotenv()

# Logging
logger = logging.getLogger('sub')

# Configuration
broker = None
quota: int = 0

# State
statedir = None
cache = None
personal = None
shared = None
public = None
clients = {}       # topic: <PubSubClient>
database = None    # <Database> instance
locks = {}
urlbase: str = ""


class PubDataset(blosc2.ProxySource):
    """
    Class for getting chunks from a dataset on a publisher service.
    """
    def __init__(self, abspath, path, metadata=None):
        self.path = pathlib.Path(path)
        if metadata is not None:
            suffix = abspath.suffix
            if suffix == '.b2nd':
                metadata = models.Metadata(**metadata)
                self.shape = metadata.shape
                self.chunks = metadata.chunks
                self.blocks = metadata.blocks
                dtype = metadata.dtype
                if metadata.dtype.startswith('['):
                    # TODO: eval is dangerous, but we mostly trust the metadata
                    # This is a list, so we need to convert it to a string
                    dtype = eval(dtype)
                self.dtype = np.dtype(dtype)
            else:
                if suffix == '.b2frame':
                    metadata = models.SChunk(**metadata)
                else:
                    abspath = pathlib.Path(f'{abspath}.b2')
                    metadata = models.SChunk(**metadata)
                self.typesize = metadata.cparams.typesize
                self.chunksize = metadata.chunksize
                self.nbytes = metadata.nbytes
            self.abspath = abspath
            if self.abspath is not None:
                self.abspath.parent.mkdir(exist_ok=True, parents=True)

    def _get_request_args(self, nchunk, return_async_client):
        root, *name = self.path.parts
        root = database.roots[root]
        name = pathlib.Path(*name)

        url = f'/api/download/{name}'
        client, url = api_utils.get_client_and_url(root.http, url, return_async_client=return_async_client)
        args = dict(url=url, params={'nchunk': nchunk}, timeout=5)
        return client, args

    def get_chunk(self, nchunk):
        client, req_args = self._get_request_args(nchunk, return_async_client=False)

        response = client.get(**req_args)
        response.raise_for_status()
        chunk = response.content
        return chunk

    async def aget_chunk(self, nchunk):
        client, req_args = self._get_request_args(nchunk, return_async_client=True)

        async with client.stream('GET', **req_args) as resp:
            buffer = []
            async for chunk in resp.aiter_bytes():
                buffer.append(chunk)
            chunk = b''.join(buffer)
            return chunk

def get_disk_usage():
    exclude = {'db.json', 'db.sqlite'}
    return sum(
        path.stat().st_size
        for path, _ in utils.walk_files(statedir, exclude=exclude)
    )

def truncate_path(path, size=35):
    """
    Smart truncaion of a long path for display.
    """
    assert type(path) is str

    if len(path) < size:
        return path

    # If the path is short just truncate at the end
    parts = pathlib.Path(path).parts
    if len(parts) < 3:
        n = len(path) - size
        return path[:-n] + '...'

    # If the path is long be smarter
    first, last = parts[0], parts[-1]
    label = f'{first}/.../{last}'
    n = len(label) - size
    if n > 0:
        last = last[:-n] + '...'

    return f'{first}/.../{last}'


def make_url(request, name, query=None, **path_params):
    url = request.app.url_path_for(name, **path_params)
    url = str(url)  # <starlette.datastructures.URLPath>
    if query:
        url = furl.furl(url).set(query).url

    # urlbase ends with slash (in my opinion it shouldn't, but if we remove the trailing
    # slash then tests fail)
    url = urlbase + url

    return url


async def new_root(data, topic):
    logger.info(f'NEW root {topic} {data=}')
    root = models.Root(**data)
    database.roots[root.name] = root
    database.save()


async def updated_dataset(data, topic):
    name = topic
    relpath = data['path']

    rootdir = cache / name
    abspath = rootdir / relpath
    metadata = data.get('metadata')
    if metadata is None:
        if abspath.suffix not in {'.b2nd', '.b2frame'}:
            abspath = pathlib.Path(f'{abspath}.b2')
        if abspath.is_file():
            abspath.unlink()
    else:
        key = f'{name}/{relpath}'
        init_b2(abspath, key, metadata)


def init_b2(abspath, path, metadata):
    dataset = PubDataset(abspath, path, metadata)
    schunk_meta = metadata.get('schunk', metadata)
    vlmeta = {}
    for k, v in schunk_meta['vlmeta'].items():
        vlmeta[k] = v
    blosc2.Proxy(dataset, urlpath=dataset.abspath, vlmeta=vlmeta, caterva2_env=True)


def open_b2(abspath, path):
    """
    Open a Blosc2 dataset.

    Return a Proxy if the dataset is in a publisher,
    or the LazyExpr or Blosc2 container otherwise.
    """
    if pathlib.Path(path).parts[0] in {'@personal', '@shared', '@public'}:
        container = blosc2.open(abspath)
        if isinstance(container, blosc2.LazyExpr):
            # Open the operands properly
            operands = container.operands
            for key, value in operands.items():
                if 'proxy-source' in value.schunk.meta:
                    # Save operand as Proxy, see blosc2.open doc for more info
                    relpath = srv_utils.get_relpath(value, cache, personal, shared, public)
                    operands[key] = open_b2(value.schunk.urlpath, relpath)
            return container
        else:
            return container

    # Return Proxy
    dataset = PubDataset(abspath, path)
    container = blosc2.open(abspath)
    # No need to pass caterva2_env=True since _cache has already been created
    return blosc2.Proxy(dataset, _cache=container)


#
# Internal API
#

def follow(name: str):
    root = database.roots.get(name)
    if root is None:
        errors = {name: 'This dataset does not exist in the network'}
        return errors

    if not root.subscribed:
        root.subscribed = True
        database.save()

    # Create root directory in the cache
    rootdir = cache / name
    if not rootdir.exists():
        rootdir.mkdir(exist_ok=True)

    # Get list of datasets
    try:
        data = api_utils.get('/api/list', server=root.http)
    except httpx.ConnectError:
        return

    # Initialize the datasets in the cache
    for relpath in data:
        # If-None-Match header
        key = f'{name}/{relpath}'
        val = database.etags.get(key)
        headers = None if val is None else {'If-None-Match': val}

        # Call API
        response = api_utils.get(f'/api/info/{relpath}', headers=headers,
                                 server=root.http, raise_for_status=False,
                                 return_response=True)
        if response.status_code == 304:
            continue

        response.raise_for_status()
        metadata = response.json()

        # Save metadata and create Proxy
        abspath = rootdir / relpath
        print('XXX follow: init_b2', abspath, key, metadata)
        init_b2(abspath, key, metadata)

        # Save etag
        database.etags[key] = response.headers['etag']
        database.save()

    # Subscribe to changes in the dataset
    if name not in clients:
        client = srv_utils.start_client(f'ws://{broker}/pubsub')
        client.subscribe(name, updated_dataset)
        clients[name] = client


#
# HTTP API
#

def user_auth_enabled():
    return bool(os.environ.get('CATERVA2_SECRET'))


current_active_user = (users.current_active_user if user_auth_enabled()
                       else (lambda: None))
"""Depend on this if the route needs an authenticated user (if enabled)."""

optional_user = (users.fastapi_users.current_user(
                     optional=True,
                     verified=False)  # TODO: set when verification works
                 if user_auth_enabled()
                 else (lambda: None))
"""Depend on this if the route may do something with no authentication."""


def _setup_plugin_globals():
    from . import plugins
    # These need to be available for plugins at import time.
    plugins.current_active_user = current_active_user


_setup_plugin_globals()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the (users) database
    if user_auth_enabled():
        await db.create_db_and_tables(statedir)

    # Initialize roots from the broker
    client = None
    if broker:
        try:
            data = api_utils.get('/api/roots', server=broker)
        except httpx.ConnectError:
            logger.warning(f'Broker "{broker}" not available')
        else:
            changed = False
            # Deleted
            d = list(database.roots.items())
            for name, root in d:
                if name not in data:
                    del database.roots[name]
                    changed = True

            # New or updated
            for name, data in data.items():
                root = models.Root(**data)
                if name not in database.roots:
                    database.roots[root.name] = root
                    changed = True
                elif database.roots[root.name].http != root.http:
                    database.roots[root.name].http = root.http
                    changed = True

            if changed:
                database.save()

            # Follow the @new channel to know when a new root is added
            client = srv_utils.start_client(f'ws://{broker}/pubsub')
            client.subscribe('@new', new_root)

            # Resume following
            for path in cache.iterdir():
                if path.is_dir():
                    follow(path.name)

    yield

    # Disconnect from worker
    if client is not None:
        await srv_utils.disconnect_client(client)

# Visualize the size of a file on a compact and human-readable format
def custom_filesizeformat(value):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024.0:
            if unit == 'B':
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


app = FastAPI(lifespan=lifespan)
if user_auth_enabled():
    app.include_router(
        users.fastapi_users.get_auth_router(users.auth_backend),
        prefix="/auth/jwt", tags=["auth"]
    )
    app.include_router(
        users.fastapi_users.get_register_router(
            schemas.UserRead, schemas.UserCreate),
        prefix="/auth", tags=["auth"],
    )
    # TODO: Support user verification, allow password reset and user deletion.


def url(path: str) -> str:
    return f"{urlbase}/{path}"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.filters['filesizeformat'] = custom_filesizeformat
templates.env.globals['url'] = url


@app.get('/api/roots')
async def get_roots(user: db.User = Depends(optional_user)) -> dict:
    """
    Get a dict of roots, with root names as keys and properties as values.

    Returns
    -------
    dict
        The dict of roots.
    """
    # Here we just return the roots that are known by the broker
    # plus the special roots @personal, @shared and @public
    roots = database.roots.copy()
    root = models.Root(name='@public', http='', subscribed=True)
    roots[root.name] = root
    if user:
        for name in ['@personal', '@shared']:
            root = models.Root(name=name, http='', subscribed=True)
            roots[root.name] = root

    return roots


def get_root(name):
    root = database.roots.get(name)
    if root is None:
        srv_utils.raise_not_found(f'{name} not known by the broker')

    return root


@app.post('/api/subscribe/{name}')
async def post_subscribe(
    name: str,
    user: db.User = Depends(optional_user),
):
    """
    Subscribe to a root.

    Parameters
    ----------
    name : str
        The name of the root.

    Returns
    -------
    str
        'Ok' if successful.
    """
    if name == '@public':
        pass
    elif name in {'@personal', '@shared'}:
        if not user:
            raise srv_utils.raise_unauthorized(f"Subscribing to {name} requires authentication")
    else:
        get_root(name)
        print('XXX post_subscribe: follow', name)
        follow(name)
    return 'Ok'


@app.get('/api/list/{path:path}')
async def get_list(
    path: pathlib.Path,
    user: db.User = Depends(optional_user),
):
    """
    List the datasets in a root or directory.

    Parameters
    ----------
    path : Path
        The path to a root or directory.

    Returns
    -------
    list
        The list of datasets, as name strings relative to path.
    """
    # Get the root
    root = path.parts[0]
    if root == '@public':
        rootdir = public
    elif root == '@personal':
        if not user:
            srv_utils.raise_not_found('@personal needs authentication')
        rootdir = personal / str(user.id)
    elif root == '@shared':
        if not user:
            srv_utils.raise_not_found('@shared needs authentication')
        rootdir = shared
    else:
        root = get_root(root)
        rootdir = cache / root.name

    # List the datasets in root or directory
    directory = rootdir / pathlib.Path(*path.parts[1:])
    if directory.is_file():
        name = pathlib.Path(directory.name)
        return [str(name.with_suffix('') if name.suffix == '.b2' else name)]
    return [str(relpath.with_suffix('') if relpath.suffix == '.b2' else relpath)
            for _, relpath in utils.walk_files(directory)]

@app.get('/api/info/{path:path}')
async def get_info(
    path: pathlib.Path,
    user: db.User = Depends(optional_user),
):
    """
    Get the metadata of a dataset.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset.

    Returns
    -------
    dict
        The metadata of the dataset.
    """
    abspath, _ = abspath_and_dataprep(path, user=user)
    return srv_utils.read_metadata(abspath, cache, personal, shared, public)


async def partial_download(abspath, path, slice_=None):
    """
    Download the necessary chunks of a dataset.

    Parameters
    ----------
    abspath : pathlib.Path
        The absolute path to the dataset.
    path : str
        The path to the dataset.
    slice_ : slice, tuple of slices
        The slice to fetch.

    Returns
    -------
    None
        When finished, the dataset is available in cache.
    """
    lock = locks.setdefault(path, asyncio.Lock())
    async with lock:
        proxy = open_b2(abspath, path)
        await proxy.afetch(slice_)


def abspath_and_dataprep(path: pathlib.Path,
                         slice_: (tuple | None) = None,
                         user: (db.User | None) = None,
                         may_not_exist=False) -> tuple[
                             pathlib.Path,
                             Callable[[], Awaitable],
                         ]:
    """
    Get absolute path in local storage and data preparation operation.

    After awaiting the preparation operation to complete, data in the
    dataset should be ready for reading, either that covered by the slice if
    given, or the whole data otherwise.
    """
    parts = path.parts
    if parts[0] == '@personal':
        if not user:
            raise fastapi.HTTPException(status_code=404)  # NotFound

        filepath = personal / str(user.id) / pathlib.Path(*parts[1:])
        abspath = srv_utils.cache_lookup(personal, filepath, may_not_exist)
        async def dataprep():
            pass

    elif parts[0] == '@shared':
        if not user:
            raise fastapi.HTTPException(status_code=404)  # NotFound

        filepath = shared / pathlib.Path(*parts[1:])
        abspath = srv_utils.cache_lookup(shared, filepath, may_not_exist)
        async def dataprep():
            pass

    elif parts[0] == '@public':
        filepath = public / pathlib.Path(*parts[1:])
        abspath = srv_utils.cache_lookup(public, filepath, may_not_exist)
        async def dataprep():
            pass

    else:
        filepath = cache / path
        abspath = srv_utils.cache_lookup(cache, filepath, may_not_exist)
        async def dataprep():
            return await partial_download(abspath, path, slice_)

    return (abspath, dataprep)

@app.get('/api/fetch/{path:path}')
async def fetch_data(
    path: pathlib.Path,
    slice_: str = None,
    user: db.User = Depends(optional_user),
):
    """
    Fetch a dataset.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset.
    slice_ : str
        The slice to fetch.

    Returns
    -------
    FileResponse or StreamingResponse
        The (slice of) dataset as a Blosc2 schunk.  When the whole dataset is
        to be downloaded (instead of some slice which does not cover it fully),
        its stored image is served containing all data and metadata (including
        variable length fields).
    """

    slice_ = api_utils.parse_slice(slice_)
    # Download and update the necessary chunks of the schunk in cache
    abspath, dataprep = abspath_and_dataprep(path, slice_, user=user)
    # This is still needed and will only update the necessary chunks
    await dataprep()
    container = open_b2(abspath, path)

    if isinstance(container, blosc2.Proxy):
        container = container._cache

    if isinstance(container, blosc2.NDArray | blosc2.LazyExpr):
        array = container
        schunk = getattr(array, 'schunk', None)
        typesize = array.dtype.itemsize
        shape = array.shape
    else:
        # SChunk
        array = None
        schunk = container
        typesize = schunk.typesize
        shape = (len(schunk),)

    whole = slice_ is None or slice_ == ()
    if not whole and isinstance(slice_, tuple):
        whole = all(isinstance(sl, slice)
                    and (sl.start or 0) == 0
                    and (sl.stop is None or sl.stop >= sh)
                    and sl.step in (None, 1)
                    for sl, sh in zip(slice_, shape))

    if whole and schunk is not None:  # whole and not lazy expr
        # Send the data in the file straight to the client,
        # avoiding slicing and re-compression.
        return FileResponse(abspath, filename=abspath.name,
                            media_type='application/octet-stream')

    if slice_:
        if array is not None:
            array = container[slice_] if array.ndim > 0 else container[()]
        else:
            if isinstance(slice_, int):
                # TODO: make SChunk support integer as slice
                slice_ = slice(slice_, slice_ + 1)
            schunk = container[slice_]

    # Serialization can be done either as:
    # * a serialized NDArray
    # * a compressed SChunk (bytes, via blosc2.compress2)
    data = array if array is not None else schunk
    if not slice_:
        # data is still a SChunk, so we need to get either a NumPy array, or a bytes object
        data = data[()] if array is not None else data[:]

    if isinstance(data, np.ndarray):
        data = blosc2.asarray(data)
        data = data.to_cframe()
    elif isinstance(data, bytes):
        # A bytes object can still be compressed as a SChunk
        schunk = blosc2.SChunk(data=data, cparams={'typesize': typesize})
        data = schunk.to_cframe()
    downloader = srv_utils.iterchunk(data)

    return responses.StreamingResponse(downloader,
                                       media_type='application/octet-stream')


@app.get('/api/chunk/{path:path}')
async def get_chunk(
    path: pathlib.PosixPath,
    nchunk: int,
    user: db.User = Depends(optional_user),
):
    abspath, _ = abspath_and_dataprep(path, user=user)
    lock = locks.setdefault(path, asyncio.Lock())
    async with lock:
        if user and path.parts[0] == '@personal':
            container = open_b2(abspath, path)
            if isinstance(container, blosc2.LazyArray):
                # We do not support LazyUDF in Caterva2 yet.
                # In case we do, this would have to be changed.
                chunk = container.get_chunk(nchunk)
            else:
                schunk = getattr(container, 'schunk', container)
                chunk = schunk.get_chunk(nchunk)
        else:
            sub_dset = PubDataset(abspath, path)
            chunk = await sub_dset.aget_chunk(nchunk)

    downloader = srv_utils.iterchunk(chunk)
    return responses.StreamingResponse(downloader)


def make_lazyexpr(name: str, expr: str, operands: dict[str, str],
                  user: db.User) -> str:
    """
    Create a lazy expression dataset in personal space.

    This may raise exceptions if there are problems parsing the dataset name
    or expression, or if the expression refers to operands which have not been
    defined.

    Parameters
    ----------
    name : str
        The name of the dataset to be created (without extension).
    expr : str
        The expression to be evaluated.  It must result in a lazy expression.
    operands : dictionary of strings mapping to strings
        The variables used in the expression and which dataset paths they
        refer to.

    Returns
    -------
    str
        The path of the newly created (or overwritten) dataset.
    """

    if not user:
        raise srv_utils.raise_unauthorized("Creating lazy expressions requires authentication")

    # Parse expression
    name = name.strip()
    expr = expr.strip()
    if not name or not expr:
        raise ValueError("Name or expression should not be empty")
    vars = ne.NumExpr(expr).input_names

    # Open expression datasets
    var_dict = {}
    for var in vars:
        path = operands[var]

        # Detect special roots
        path = pathlib.Path(path)
        if path.parts[0] == '@personal':
            abspath = personal / str(user.id) / pathlib.Path(*path.parts[1:])
        elif path.parts[0] == '@shared':
            abspath = shared / pathlib.Path(*path.parts[1:])
        elif path.parts[0] == '@public':
            abspath = public / pathlib.Path(*path.parts[1:])
        else:
            abspath = cache / path
        var_dict[var] = open_b2(abspath, path)

    # Create the lazy expression dataset
    arr = eval(expr, var_dict)
    if not isinstance(arr, blosc2.LazyExpr):
        cname = type(arr).__name__
        raise TypeError(f"Evaluates to {cname} instead of lazy expression")

    path = personal / str(user.id)
    path.mkdir(exist_ok=True, parents=True)
    arr.save(urlpath=f'{path / name}.b2nd', mode="w")

    return f'@personal/{name}.b2nd'


@app.post('/api/lazyexpr/')
async def lazyexpr(
    expr: models.NewLazyExpr,
    user: db.User = Depends(current_active_user),
) -> str:
    """
    Create a lazy expression dataset in personal space.

    The JSON request body must contain a "name" for the dataset to be created
    (without extension), an "expression" to be evaluated, which must result in
    a lazy expression, and an "operands" object which maps variable names used
    in the expression to the dataset paths that they refer to.

    Returns
    -------
    str
        The path of the newly created (or overwritten) dataset.
    """

    def error(msg):
        return fastapi.HTTPException(status_code=400, detail=msg)  # bad request

    try:
        result_path = make_lazyexpr(expr.name, expr.expression, expr.operands,
                                    user)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise error(f'Invalid name or expression: {exc}')
    except KeyError as ke:
        raise error(f'Expression error: {ke.args[0]} is not in the list of available datasets')
    except RuntimeError as exc:
        raise error(str(exc))

    return result_path


@app.post('/api/move/')
async def move(
    payload: models.MoveCopyPayload,
    user: db.User = Depends(current_active_user),
):
    """
    Move a dataset.

    Returns
    -------
    str
        The new path of the dataset.
    """
    if not user:
        raise srv_utils.raise_unauthorized("Moving files requires authentication")

    # Both src and dst should start with a special root
    if not payload.src.startswith(('@personal', '@shared', '@public')):
        raise fastapi.HTTPException(status_code=400, detail=
        "Only moving from @personal or @shared or @public roots is allowed")
    if not payload.dst.startswith(('@personal', '@shared', '@public')):
        raise fastapi.HTTPException(status_code=400, detail=
        "Only moving to @personal or @shared or @public roots is allowed")
    namepath = pathlib.Path(payload.src)
    destpath = pathlib.Path(payload.dst)
    abspath, _ = abspath_and_dataprep(namepath, user=user)
    dest_abspath, _ = abspath_and_dataprep(destpath, user=user, may_not_exist=True)

    # If destination has not an extension, assume it is a directory
    # If user wants something without an extension, she can add a '.b2' extension :-)
    if dest_abspath.is_dir() or not dest_abspath.suffix:
        dest_abspath /= abspath.name
        destpath /= namepath.name

    # Not sure if we should allow overwriting, but let's allow it for now
    # if dest_abspath.exists():
    #     raise fastapi.HTTPException(status_code=409, detail="The new path already exists")

    # Make sure the destination directory exists
    dest_abspath.parent.mkdir(exist_ok=True, parents=True)
    abspath.rename(dest_abspath)

    return str(destpath)


@app.post('/api/copy/')
async def copy(
    payload: models.MoveCopyPayload,
    user: db.User = Depends(current_active_user),
):
    """
    Copy a dataset.

    Returns
    -------
    str
        The path of the copied dataset.
    """
    if not user:
            raise srv_utils.raise_unauthorized("Copying files requires authentication")

    src, dst = payload.src, payload.dst
    # src should start with a special root or known root
    if (not src.startswith(('@personal', '@shared', '@public'))
            and src not in database.roots):
        raise fastapi.HTTPException(status_code=400, detail=
        "Only copying from existing roots is allowed")
    # dst should start with a special root
    if not dst.startswith(('@personal', '@shared', '@public')):
        raise fastapi.HTTPException(status_code=400, detail=
        "Only copying to @personal or @shared or @public roots is allowed")

    namepath, destpath = pathlib.Path(src), pathlib.Path(dst)
    abspath, _ = abspath_and_dataprep(namepath, user=user)
    dest_abspath, _ = abspath_and_dataprep(destpath, user=user, may_not_exist=True)

    # If destination has not an extension, assume it is a directory
    # If user wants something without an extension, she should add a '.b2' extension
    if dest_abspath.is_dir() or not dest_abspath.suffix:
            dest_abspath /= abspath.name
            destpath /= namepath.name

    # Not sure if we should allow overwriting, but let's allow it for now
    # if dest_abspath.exists():
    #     raise fastapi.HTTPException(status_code=409, detail="The new path already exists")

    dest_abspath.parent.mkdir(exist_ok=True, parents=True)
    if abspath.is_dir():
        shutil.copytree(abspath, dest_abspath)
    else:
        shutil.copy(abspath, dest_abspath)

    return str(destpath)


@app.post('/api/upload/{path:path}')
async def upload_file(
        path: pathlib.Path,
        file: UploadFile,
        user: db.User = Depends(current_active_user),
):
    """
    Upload a file to a root.

    Parameters
    ----------
    path : pathlib.Path
        The path to store the uploaded file.
    file : UploadFile
        The file to upload.

    Returns
    -------
    str
        The path of the uploaded file.
    """

    if not user:
        raise srv_utils.raise_unauthorized("Uploading requires authentication")

    # Read the file
    data = await file.read()

    # Replace the root with absolute path
    root = path.parts[0]
    if root == '@personal':
        path2 = personal / str(user.id) / pathlib.Path(*path.parts[1:])
    elif root == '@shared':
        path2 = shared / pathlib.Path(*path.parts[1:])
    elif root == '@public':
        path2 = public / pathlib.Path(*path.parts[1:])
    else:
        raise fastapi.HTTPException(
            status_code=400,  # bad request
            detail="Only uploading to @personal or @shared or @public roots is allowed",
        )

    if path2.is_dir():
        path2 /= file.filename
        path /= file.filename
    path2.parent.mkdir(exist_ok=True, parents=True)

    # If regular file, compress it
    if path2.suffix not in {'.b2', '.b2frame', '.b2nd'}:
        schunk = blosc2.SChunk(data=data)
        data = schunk.to_cframe()
        path2 = path2.with_suffix(path2.suffix + '.b2')

    # Write the file
    with open(path2, 'wb') as f:
        f.write(data)

    # Return the urlpath
    return str(path)


@app.post('/api/remove/{path:path}')
async def remove(
        path: pathlib.Path,
        user: db.User = Depends(current_active_user),
):
    """
    Remove a dataset or a directory path.

    Parameters
    ----------
    path : pathlib.Path
        The path of dataset / directory to remove.

    Returns
    -------
    list
        A list with the paths that have been removed.
    """

    if not user:
        raise srv_utils.raise_unauthorized("Removing files requires authentication")

    # Replace the root with absolute path
    root = path.parts[0]
    if root == '@personal':
        path2 = personal / str(user.id) / pathlib.Path(*path.parts[1:])
    elif root == '@shared':
        path2 = shared / pathlib.Path(*path.parts[1:])
    elif root == '@public':
        path2 = public / pathlib.Path(*path.parts[1:])
    else:
        # Only allow removing from the special roots
        raise fastapi.HTTPException(
            status_code=400,  # bad request
            detail="Only removing from @personal or @shared or @public roots is allowed",
        )

    # If path2 is a directory, remove the contents of the directory
    if path2.is_dir():
        shutil.rmtree(path2)
    else:
        # Try to unlink the file
        try:
            path2.unlink()
        except FileNotFoundError:
            # Try adding a .b2 extension
            path2 = path2.with_suffix(path2.suffix + '.b2')
            try:
                path2.unlink()
            except FileNotFoundError:
                raise fastapi.HTTPException(
                    status_code=404,  # not found
                    detail="The specified path does not exist",
                )

    # Return the path
    return path


#
# HTML interface
#

if user_auth_enabled():
    @app.get("/login", response_class=HTMLResponse)
    async def html_login(
            request: Request,
            user: db.User = Depends(optional_user)
    ):
        if user:
            return RedirectResponse(urlbase, status_code=307)

        return templates.TemplateResponse(request, "login.html")

    @app.get("/logout", response_class=HTMLResponse)
    async def html_logout(
            request: Request,
            user: db.User = Depends(optional_user)
    ):
        if user:
            return RedirectResponse(urlbase, status_code=307)

        return templates.TemplateResponse(request, "logout.html")

    @app.get("/register", response_class=HTMLResponse)
    async def html_register(
            request: Request,
            user: db.User = Depends(optional_user)
    ):
        if user:
            return RedirectResponse(urlbase, status_code=307)

        return templates.TemplateResponse(request, "register.html")

    # TODO: Support user verification, allow password reset and user deletion.


@app.get("/", response_class=HTMLResponse)
@app.get("/roots/{path:path}")
async def html_home(
    request: Request,
    path: str = '',
    # Query parameters
    roots: list[str] = fastapi.Query([]),
    search: str = '',
    # Dependencies
    user: db.User = Depends(optional_user),
):

    # Disk usage
    size = get_disk_usage()
    context = {
        'user_auth_enabled': user_auth_enabled(),
        'roots_url': make_url(request, 'htmx_root_list', {'roots': roots}),
        'username': user.email if user else None,
        # Disk usage
        'usage_total':  custom_filesizeformat(size),
    }

    if quota:
        context['usage_quota'] = custom_filesizeformat(quota)
        context['usage_percent'] = round((size / quota) * 100)

    if roots:
        paths_url = make_url(request, 'htmx_path_list', {'roots': roots, 'search': search})
        context['paths_url'] = paths_url

    if path:
        context["meta_url"] = make_url(request, 'htmx_path_info', path=path)

    return templates.TemplateResponse(request, "home.html", context)


@app.get("/htmx/root-list/")
async def htmx_root_list(
    request: Request,
    # Query
    roots: list[str] = fastapi.Query([]),
    # Depends
    user: db.User = Depends(optional_user),
):

    context = {
        "checked": roots,
        "roots": sorted(database.roots.values(), key=lambda x: x.name),
        "user": user,
    }
    return templates.TemplateResponse(request, "root_list.html", context)


@app.get("/htmx/path-list/", response_class=HTMLResponse)
async def htmx_path_list(
    request: Request,
    # Query parameters
    roots: list[str] = fastapi.Query([]),
    search: str = '',
    # Headers
    hx_current_url: srv_utils.HeaderType = None,
    hx_trigger: srv_utils.HeaderType = None,
    # Depends
    user: db.User = Depends(optional_user),
):

    hx_current_url = furl.furl(hx_current_url)

    # Prepare datasets context
    def get_names():
        n = 1
        while True:
            for name in itertools.product(* [string.ascii_lowercase] * n):
                yield ''.join(name)
            n += 1

    names = get_names()

    query = {'roots': roots, 'search': search}
    datasets = []
    for root in roots:
        if user and root == '@personal':
            rootdir = personal / str(user.id)
        elif user and root == '@shared':
            rootdir = shared
        elif root == '@public':
            rootdir = public
        else:
            if not get_root(root).subscribed:
                follow(root)
            rootdir = cache / root

        for path, relpath in utils.walk_files(rootdir):
            size = path.stat().st_size
            if relpath.suffix == '.b2':
                relpath = relpath.with_suffix('')
            if search in str(relpath):
                path = f'{root}/{relpath}'
                url = make_url(request, "html_home", path=path, query=query)
                datasets.append({
                    'name': next(names),
                    'path': path,
                    'size': size,
                    'url': url,
                    'label': truncate_path(path),
                })

    # Render template
    cmd_url = make_url(request, 'htmx_command')
    search_url = make_url(request, 'htmx_path_list', {'roots': roots})
    context = {
        "datasets": datasets,
        "search_text": search,
        "search_url": search_url,
        "cmd_url": cmd_url,
        "user": user,
    }
    response = templates.TemplateResponse(request, "path_list.html", context)

    # Push URL only when clicked, not on load/reload
    if hx_trigger != 'path-list':
        args = {'roots': roots}
        if search:
            args['search'] = search
        push_url = hx_current_url.set(args).url
        response.headers['HX-Push-Url'] = push_url

    return response


@app.get("/htmx/path-info/{path:path}", response_class=HTMLResponse)
async def htmx_path_info(
    request: Request,
    # Path parameters
    path: pathlib.Path,
    # Headers
    hx_current_url: srv_utils.HeaderType = None,
    hx_trigger: srv_utils.HeaderType = None,
    # Depends
    user: db.User = Depends(optional_user),
):

    try:
        abspath, _ = abspath_and_dataprep(path, user=user)
    except FileNotFoundError:
        return htmx_error(request, 'FileNotFoundError: missing operand(s)')

    meta = srv_utils.read_metadata(abspath, cache, personal, shared, public)

    vlmeta = getattr(getattr(meta, 'schunk', meta), 'vlmeta', {})
    contenttype = vlmeta.get('contenttype') or guess_dset_ctype(path, meta)
    plugin = plugins.get(contenttype)
    if plugin:
        display = {
            "url": f"/plugins/{plugin.name}/display/{path}",
            "label": plugin.label,
        }
    elif path.suffix == ".md":
        display = {
            "url": f"/markdown/{path}",
            "label": "Display",
        }
    else:
        display = None

    context = {
        "path": path,
        "meta": meta,
        "display": display,
        "can_delete": user and path.parts[0] in {"@personal", "@shared", "@public"},
    }

    # XXX
    if hasattr(meta, 'shape'):
        view_url = make_url(request, "htmx_path_view", path=path)
        context.update({
            "view_url": view_url,
            "shape": meta.shape,
        })

    response = templates.TemplateResponse(request, "info.html", context=context)

    # Push URL only when clicked, not on load/reload
    if hx_trigger != 'meta':
        push_url = make_url(request, 'html_home', path=path)
        # Keep query
        current_query = furl.furl(hx_current_url).query
        if current_query:
            push_url = f'{push_url}?{current_query.encode()}'

        response.headers['HX-Push-Url'] = push_url

    return response


@app.post("/htmx/path-view/{path:path}", response_class=HTMLResponse)
async def htmx_path_view(
    request: Request,
    # Path parameters
    path: pathlib.Path,
    # Input parameters
    index: typing.Annotated[list[int], Form()] = None,
    sizes: typing.Annotated[list[int], Form()] = None,
    fields: typing.Annotated[list[str], Form()] = None,
    # Depends
    user: db.User = Depends(optional_user),
):
    abspath, _ = abspath_and_dataprep(path, user=user)
    arr = open_b2(abspath, path)

    # Local variables
    shape = arr.shape
    ndims = len(shape)
    has_ndfields = hasattr(arr, 'fields') and arr.fields != {}

    # Set of dimensions that define the window
    # TODO Allow the user to choose the window dimensions
    dims = list(range(ndims))
    if ndims == 0:
        view_dims = {}
    elif ndims == 1 or has_ndfields:
        view_dims = {dims[-1]}
    else:
        view_dims = {dims[-2], dims[-1]}

    # Default values for input params
    index = (0,) * ndims if index is None else tuple(index)
    if sizes is None:
        sizes = [min(dim, 10) if i in view_dims else 1 for i, dim in enumerate(shape)]

    inputs = []
    tags = []
    for i, (start, size, size_max) in enumerate(zip(index, sizes, shape)):
        mod = size_max % size
        start_max = size_max - (mod or size)
        inputs.append({
            'start': start,
            'start_max': start_max,
            'size': size,
            'size_max': size_max,
            'with_size': i in view_dims,
        })
        if inputs[-1]['with_size']:
            tags.append([k for k in range(start, min(start+size, size_max))])

    if has_ndfields:
        cols = list(arr.fields.keys())
        fields = fields or cols[:5]
        idxs = [cols.index(f) for f in fields]
        rows = [fields]

        # Get array view
        if ndims >= 2:
            arr = arr[index[:-1]]
            i, isize = index[-1], sizes[-1]
            arr = arr[i:i + isize]
            arr = arr.tolist()
        elif ndims == 1:
            i, isize = index[0], sizes[0]
            arr = arr[i:i + isize]
            arr = arr.tolist()
        else:
            arr = [arr[()].tolist()]
        rows += [[row[i] for i in idxs] for row in arr]
    else:
        # Get array view
        cols = None
        if ndims >= 2:
            arr = arr[index[:-2]]
            i, isize = index[-2], sizes[-2]
            j, jsize = index[-1], sizes[-1]
            arr = arr[i:i+isize, j:j+jsize]
            rows = [tags[-1]] + list(arr)
        elif ndims == 1:
            i, isize = index[0], sizes[0]
            arr = [arr[i:i+isize]]
            rows = [tags[-1]] + list(arr)
        else:
            arr = [[arr[()]]]
            rows = list(arr)

    # Render
    context = {
        "view_url": make_url(request, "htmx_path_view", path=path),
        "inputs": inputs,
        "rows": rows,
        "cols": cols,
        "fields": fields,
        "tags": tags if len(tags) == 0 else tags[0],
    }
    return templates.TemplateResponse(request, "info_view.html", context)

@app.post("/htmx/command/", response_class=HTMLResponse)
async def htmx_command(
    request: Request,
    # Body
    command: typing.Annotated[str, Form()],
    names: typing.Annotated[list[str], Form()],
    paths: typing.Annotated[list[str], Form()],
    # Headers
    hx_current_url: srv_utils.HeaderType = None,
    # Depends
    user: db.User = Depends(current_active_user),
):

    operands = dict(zip(names, paths))
    argv = command.split()

    # First check for expressions
    if argv[1] == '=':
        try:
            result_name, expr = command.split('=')
            result_path = make_lazyexpr(result_name, expr, operands, user)
        except (SyntaxError, ValueError):
            return htmx_error(request,
                              'Invalid syntax: expected <varname> = <expression>')
        except TypeError as te:
            return htmx_error(request, f'Invalid expression: {te}')
        except KeyError as ke:
            error = f'Expression error: {ke.args[0]} is not in the list of available datasets'
            return htmx_error(request, error)
        except RuntimeError as exc:
            return htmx_error(request, str(exc))

    # then commands
    elif argv[0] in {'cp', 'copy'}:
        if len(argv) != 3:
            return htmx_error(request, 'Invalid syntax: expected cp/copy <src> <dst>')
        src, dst = operands.get(argv[1], argv[1]), operands.get(argv[2], argv[2])
        payload = models.MoveCopyPayload(src=src, dst=dst)
        try:
            result_path = await copy(payload, user)
        except Exception as exc:
            return htmx_error(request, f'Error copying file: {exc}')
        result_path = await display_first(result_path, user)

    elif argv[0] in {'i', 'info'}:
        if len(argv) != 2:
            return htmx_error(request, 'Invalid syntax: expected i/info <path>')
        path = operands.get(argv[1], argv[1])
        path = pathlib.Path(path)
        try:
            paths = await get_list(path, user)
        except Exception as exc:
            return htmx_error(request, f'Error listing path: {exc}')
        if len(paths) != 1:
            return htmx_error(request, f'dataset "{path}" not found')
        result_path = path

    elif argv[0] in {'ls', 'list'}:
        if len(argv) != 2:
            return htmx_error(request, 'Invalid syntax: expected ls/list <path>')
        path = operands.get(argv[1], argv[1])
        path = pathlib.Path(path)
        try:
            paths = await get_list(path, user)
        except Exception as exc:
            return htmx_error(request, f'Error listing path: {exc}')
        # Get the first path to display
        first_path = next(iter(paths), None)
        if path.name == first_path:
            result_path = path
        else:
            result_path = f'{path}/{first_path}' if first_path else path

    elif argv[0] in {'mv', 'move'}:
        if len(argv) != 3:
            return htmx_error(request, 'Invalid syntax: expected mv/move <src> <dst>')
        src, dst = operands.get(argv[1], argv[1]), operands.get(argv[2], argv[2])
        payload = models.MoveCopyPayload(src=src, dst=dst)
        try:
            result_path = await move(payload, user)
        except Exception as exc:
            return htmx_error(request, f'Error moving file: {exc}')
        result_path = await display_first(result_path, user)

    elif argv[0] in {'rm', 'remove'}:
        if len(argv) != 2:
            return htmx_error(request, 'Invalid syntax: expected rm/remove <path>')
        path = operands.get(argv[1], argv[1])
        path = pathlib.Path(path)
        try:
            # Nothing to show after removing, but anyway
            result_path = await remove(path, user)
        except Exception as exc:
            return htmx_error(request, f'Error removing file: {exc}')

    else:
        return htmx_error(request, f'Invalid command "{argv[0]}" or expression not found')

    # Redirect to display new dataset
    url = make_url(request, "html_home", path=result_path)
    return htmx_redirect(hx_current_url, url)


async def display_first(result_path, user):
    paths = await get_list(pathlib.Path(result_path), user)
    if len(paths) > 1:
        # Display the first path found
        result_path = f'{result_path}/{paths[0]}'
    elif len(paths) == 1 and not result_path.endswith(paths[0]):
        result_path = f'{result_path}/{paths[0]}'
    return result_path


def htmx_error(request, msg):
    context = {'error': msg}
    return templates.TemplateResponse(request, "error.html", context, status_code=400)


def htmx_redirect(current_url, target_url, root=None):
    response = JSONResponse('OK')
    query = furl.furl(current_url).query
    roots = query.params.getlist('roots')

    if root and root not in roots:
        query = query.add({'roots': root})

    response.headers['HX-Redirect'] = f'{target_url}?{query.encode()}'
    return response

@app.post("/htmx/upload/{name}")
async def htmx_upload(
    request: Request,
    name: str,
    # Body
    file: UploadFile,
    # Headers
    hx_current_url: srv_utils.HeaderType = None,
    # Depends
    user: db.User = Depends(current_active_user),
):

    if not user:
        raise srv_utils.raise_unauthorized("Uploading files requires authentication")

    if name == '@personal':
        path = personal / str(user.id)
    elif name == '@shared':
        path = shared
    elif name == '@public':
        path = public
    else:
        raise fastapi.HTTPException(status_code=404)  # NotFound

    path.mkdir(exist_ok=True, parents=True)

    # Read file
    filename = pathlib.Path(file.filename)
    data = await file.read()

    # Check quota
    if quota:
        upload_size = len(data)
        total_size = get_disk_usage() + upload_size
        if total_size > quota:
            error = 'Upload failed because quota limit has been exceeded.'
            return htmx_error(request, error)

    # If a tarball or zipfile, extract the files in path
    # We also filter out hidden files and MacOSX metadata
    suffixes = filename.suffixes
    if suffixes in (['.tar', '.gz'], ['.tar'], ['.tgz'], ['.zip']):
        file.file.seek(0)  # Reset file pointer
        if suffixes == ['.zip']:
            with zipfile.ZipFile(file.file, 'r') as archive:
                members = [m for m in archive.namelist()
                           if (not os.path.basename(m).startswith('.') and
                               not os.path.basename(m).startswith('__MACOSX'))]
                archive.extractall(path, members=members)
                # Convert members elements to Path instances
                members = [pathlib.Path(m) for m in members]
        else:
            mode = 'r:gz' if suffixes[-1] in {'.tgz', '.gz'} else 'r'
            with tarfile.open(fileobj=file.file, mode=mode) as archive:
                members = [m for m in archive.getmembers()
                           if (not os.path.basename(m.name).startswith('.') and
                               not os.path.basename(m.name).startswith('__MACOSX'))]
                archive.extractall(path, members=members)
                # Convert members elements to Path instances
                members = [pathlib.Path(m.name) for m in members]

        # Compress files that are not compressed yet
        new_members = [
            member for member in members
            if not (path / member).is_dir() and not member.suffix in {'.b2', '.b2frame', '.b2nd'}
        ]
        for member in new_members:
            member_path = path / member
            with open(member_path, 'rb') as src:
                data = src.read()
                schunk = blosc2.SChunk(data=data)
                data = schunk.to_cframe()
                member_path2 = f'{member_path}.b2'
            with open(member_path2, 'wb') as dst:
                dst.write(data)
            member_path.unlink()
        # We are done, redirect to home, and show the new files, starting with the first one
        first_member = next((m for m in new_members), None)
        path = f'{name}/{first_member}'
        return htmx_redirect(hx_current_url, make_url(request, "html_home", path=path), root=name)

    if filename.suffix not in {'.b2', '.b2frame', '.b2nd'}:
        schunk = blosc2.SChunk(data=data)
        data = schunk.to_cframe()
        filename = f'{filename}.b2'

    # Save file
    with open(path / filename, 'wb') as dst:
        dst.write(data)

    # Redirect to display new dataset
    path = f'{name}/{filename}'
    url = make_url(request, "html_home", path=path)
    return htmx_redirect(hx_current_url, url, root=name)


@app.delete("/htmx/delete/{path:path}", response_class=HTMLResponse)
async def htmx_delete(
    request: Request,
    # Path parameters
    path: pathlib.Path,
    # Headers
    hx_current_url: srv_utils.HeaderType = None,
    # Depends
    user: db.User = Depends(current_active_user),
):

    # Find absolute path to file
    parts = list(path.parts)
    name = parts[0]
    if name == "@personal":
        parts[0] = str(user.id)
        path = pathlib.Path(*parts)
        abspath = personal / path
    elif name == "@shared":
        path = pathlib.Path(*parts[1:])
        abspath = shared / path
    elif name == "@public":
        path = pathlib.Path(*parts[1:])
        abspath = public / path
    else:
        return fastapi.HTTPException(status_code=400)

    # Remove
    if abspath.suffix not in {'.b2frame', '.b2nd'}:
        abspath = abspath.with_suffix(abspath.suffix + '.b2')
        if not abspath.exists():
            return fastapi.HTTPException(status_code=404)
    abspath.unlink()

    # Redirect to home
    url = make_url(request, "html_home")
    return htmx_redirect(hx_current_url, url, root=name)


@app.get("/markdown/{path:path}", response_class=HTMLResponse)
async def html_markdown(
    request: Request,
    # Path parameters
    path: pathlib.Path,
    user: db.User = Depends(optional_user),
    # Response
    response_class=HTMLResponse,
):

    abspath, _ = abspath_and_dataprep(path, user=user)
    arr = open_b2(abspath, path)
    content = arr[:]
    # Markdown
    return markdown.markdown(content.decode('utf-8'))


#
# Command line interface
#

plugins = {}


def guess_dset_ctype(path: pathlib.Path, meta) -> str | None:
    """Try to guess dataset's content type (given path and metadata)."""
    for (ctype, plugin) in plugins.items():
        if (hasattr(plugin, 'guess')
                and plugin.guess(path, meta)):
            return ctype
    return None


def parse_size(size):
    if size is None:
        return None

    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40 ,
             "":  1, "KiB": 10**3, "MiB": 10**6, "GiB": 10**9, "TiB": 10**12}
    m = re.match(r'^([\d\.]+)\s*([a-zA-Z]{0,3})$', str(size).strip())
    number, unit = float(m.group(1)), m.group(2).upper()
    return int(number*units[unit])


def main():
    conf = utils.get_conf('subscriber', allow_id=True)
    _stdir = '_caterva2/sub' + (f'.{conf.id}' if conf.id else '')
    parser = utils.get_parser(broker=conf.get('broker.http', ''),
                              http=conf.get('.http', 'localhost:8002'),
                              url=conf.get('.url'),
                              loglevel=conf.get('.loglevel', 'warning'),
                              statedir=conf.get('.statedir', _stdir),
                              id=conf.id)
    args = utils.run_parser(parser)

    # Global configuration
    global broker, quota
    broker = args.broker
    quota = parse_size(conf.get('.quota'))

    # Init cache
    global statedir, cache
    statedir = args.statedir.resolve()
    cache = statedir / 'cache'
    cache.mkdir(exist_ok=True, parents=True)
    # Use `download_cached()`, `StaticFiles` does not support authorization.
    #app.mount("/files", StaticFiles(directory=cache), name="files")

    # Shared/Public dirs
    global shared, public
    shared = statedir / 'shared'
    shared.mkdir(exist_ok=True, parents=True)
    public = statedir / 'public'
    public.mkdir(exist_ok=True, parents=True)

    # personal dir
    global personal
    personal = statedir / 'personal'
    personal.mkdir(exist_ok=True, parents=True)
    # Use `download_personal()`, `StaticFiles` does not support authorization.
    #app.mount("/personal", StaticFiles(directory=personal), name="personal")

    # Init database
    global database
    model = models.Subscriber(roots={}, etags={})
    database = srv_utils.Database(statedir / 'db.json', model)

    # Register display plugins
    from .plugins import tomography  # delay module load
    app.mount(f"/plugins/{tomography.name}", tomography.app)
    plugins[tomography.contenttype] = tomography
    tomography.init(abspath_and_dataprep)

    # Run
    global urlbase
    urlbase = args.url
    root_path = str(furl.furl(urlbase).path)
    print('XXX', root_path)
    utils.uvicorn_run(app, args, root_path=root_path)



if __name__ == '__main__':
    main()
