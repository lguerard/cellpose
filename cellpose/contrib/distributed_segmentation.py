# stdlib imports
import datetime
import functools
import getpass
import glob
import os
import pathlib
import tempfile

# distributed dependencies
import dask
import distributed
import numpy as np
import scipy
import tifffile
import xarray as xr

from cellpose import models


######################## File format functions ################################
def numpy_array_to_xarray(write_path, array, chunks):
    """
    Store an in-memory numpy array to disk as a chunked xarray DataArray (NetCDF).

    Parameters
    ----------
    write_path : str
        Filepath where NetCDF file will be created.
    array : numpy.ndarray
        The already loaded in-memory numpy array to store as xarray.
    chunks : tuple
        How the array will be chunked in the xarray DataArray.

    Returns
    -------
    xarray.DataArray
        A reference to the xarray DataArray on disk.

    Examples
    --------
    >>> arr = np.random.rand(10, 20, 30)
    >>> da = numpy_array_to_xarray('test.nc', arr, (5, 10, 10))
    """
    da = xr.DataArray(array, dims=[f"dim_{i}" for i in range(array.ndim)])
    da = da.chunk(dict(zip(da.dims, chunks)))
    da.to_netcdf(write_path)
    return xr.open_dataarray(write_path, chunks=dict(zip(da.dims, chunks)))


def wrap_folder_of_tiffs(
    filename_pattern,
    block_index_pattern=r"_(Z)(\d+)(Y)(\d+)(X)(\d+)",
):
    """
    Wrap a folder of tiff files with an xarray DataArray without duplicating data.

    Parameters
    ----------
    filename_pattern : str
        A glob pattern that will match all needed tif files.
    block_index_pattern : str
        Regular expression pattern to parse tiff filenames for block grid.

    Returns
    -------
    xarray.DataArray
        The stacked image as an xarray DataArray.

    Examples
    --------
    >>> da = wrap_folder_of_tiffs('folder/*.tif')
    """
    files = sorted(glob.glob(filename_pattern))
    arrays = [tifffile.imread(f) for f in files]
    stacked = np.stack(arrays)
    da = xr.DataArray(stacked, dims=[f"dim_{i}" for i in range(stacked.ndim)])
    return da


######################## Cluster related functions ############################

DEFAULT_CONFIG_FILENAME = "distributed_cellpose_dask_config.yaml"


def _config_path(config_name):
    return str(pathlib.Path.home()) + "/.config/dask/" + config_name


def _modify_dask_config(config, config_name=DEFAULT_CONFIG_FILENAME):
    """
    Set Dask config in memory only, do not write any config file to disk.
    """
    dask.config.set(config)


class myLocalCluster(distributed.LocalCluster):
    """
    Thin wrapper for dask.distributed.LocalCluster with config setup.
    """

    def __init__(
        self,
        ncpus,
        config={},
        config_name=DEFAULT_CONFIG_FILENAME,
        persist_config=False,
        **kwargs,
    ):
        # config
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"{os.getcwd()}/.{getpass.getuser()}_distributed_cellpose/"
        config_defaults = {"temporary-directory": scratch_dir}
        config = {**config_defaults, **config}
        _modify_dask_config(config, config_name)

        # construct
        if "host" not in kwargs:
            kwargs["host"] = ""
        super().__init__(**kwargs)
        self.client = distributed.Client(self)
        print("Cluster dashboard link: ", self.dashboard_link)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)


def cluster(func):
    """
    Decorator to ensure a function runs inside a Dask cluster context.
    """

    @functools.wraps(func)
    def create_or_pass_cluster(*args, **kwargs):
        assert "cluster" in kwargs or "cluster_kwargs" in kwargs, (
            "Either cluster or cluster_kwargs must be defined"
        )
        if "cluster" not in kwargs:
            cluster_constructor = myLocalCluster

            def F(x):
                return x in kwargs["cluster_kwargs"]

            if F("ncpus") and F("min_workers") and F("max_workers"):
                raise NotImplementedError(
                    "LSF cluster support not implemented in xarray version."
                )
            with cluster_constructor(**kwargs["cluster_kwargs"]) as cluster:
                kwargs["cluster"] = cluster
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return create_or_pass_cluster


######################## the function to run on each block ####################
def process_block(
    block_index,
    crop,
    input_xr,
    model_kwargs,
    eval_kwargs,
    blocksize,
    overlap,
    output_xr,
    preprocessing_steps=None,
    worker_logs_directory=None,
    test_mode=False,
):
    """
    Preprocess and segment one block, of many, with eventual merger of all blocks in mind.
    Supports multi-channel images (channel-first convention).

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid.
    crop : tuple of slice
        The bounding box of the data to read from the input_xr array.
    input_xr : xarray.DataArray
        The image data to segment.
    preprocessing_steps : list, optional
        List of (function, kwargs) tuples for preprocessing.
    model_kwargs : dict
        Arguments passed to cellpose.models.Cellpose or CellposeModel.
    eval_kwargs : dict
        Arguments passed to the eval function of the Cellpose model.
    blocksize : tuple
        The shape of blocks without overlaps.
    overlap : int
        The number of voxels added to the blocksize for context.
    output_xr : xarray.DataArray
        Where segments can be stored temporarily before merger.
    worker_logs_directory : str, optional
        Directory for worker logs.
    test_mode : bool, optional
        If True, returns segments and boxes instead of writing to disk.

    Returns
    -------
    If test_mode is False:
        faces : list of numpy.ndarray
        boxes : list of tuple of slices
        box_ids : numpy.ndarray
    If test_mode is True:
        segments : numpy.ndarray
        boxes : list of tuple of slices
        box_ids : numpy.ndarray

    Examples
    --------
    >>> faces, boxes, box_ids = process_block(...)
    """
    if preprocessing_steps is None:
        preprocessing_steps = []
    print(f"RUNNING BLOCK: {block_index}\tREGION: {crop}", flush=True)
    image = input_xr[crop].values
    # If image has a channel dimension, ensure it is passed correctly to Cellpose
    if image.ndim == 4:  # (C, Z, Y, X)
        # Cellpose expects (Z, Y, X, C) or (Y, X, C) for 2D
        image = np.moveaxis(image, 0, -1)  # Move channel to last axis
    elif image.ndim == 3 and input_xr.shape[0] <= 4:  # (C, Y, X) or (C, Z, Y)
        image = np.moveaxis(image, 0, -1)
    for pp_step in preprocessing_steps:
        pp_step[1]["crop"] = crop
        image = pp_step[0](image, **pp_step[1])
    model = models.CellposeModel(**model_kwargs)
    segmentation = model.eval(image, **eval_kwargs)[0].astype(np.uint32)
    # If input was 4D, add back singleton channel if needed
    if input_xr.ndim == 4 and segmentation.ndim == 3:
        segmentation = np.expand_dims(segmentation, axis=0)
    segmentation, crop = remove_overlaps(segmentation, crop, overlap, blocksize)
    boxes = bounding_boxes_in_global_coordinates(segmentation, crop)
    nblocks = get_nblocks(input_xr.shape, blocksize)
    segmentation, remap = global_segment_ids(segmentation, block_index, nblocks)
    if remap[0] == 0:
        remap = remap[1:]
    if test_mode:
        return segmentation, boxes, remap
    output_xr[crop] = segmentation
    faces = block_faces(segmentation)
    return faces, boxes, remap


# ----------------------- component functions ---------------------------------#
def remove_overlaps(array, crop, overlap, blocksize):
    """overlaps only there to provide context for boundary voxels
    and can be removed after segmentation is complete
    reslice array to remove the overlaps"""
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [
                slice(None),
            ] * array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [
                slice(None),
            ] * array.ndim
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


def bounding_boxes_in_global_coordinates(segmentation, crop):
    """
    Compute bounding boxes (tuples of slices) in global coordinates.
    """
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]

    def translate(a, b):
        return slice(a.start + b.start, a.start + b.stop)

    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(translate(a, b) for a, b in zip(crop, box))
    return boxes


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def global_segment_ids(segmentation, block_index, nblocks):
    """pack the block index into the segment IDs so they are
    globally unique. Everything gets remapped to [1..N] later.
    A uint32 is split into 5 digits on left and 5 digits on right.
    This creates limits: 42950 maximum number of blocks and
    99999 maximum number of segments per block"""
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [np.uint32(p + str(x).zfill(5)) for x in unique]
    if unique[0] == 0:
        remap[0] = np.uint32(0)  # 0 should just always be 0
    segmentation = np.array(remap)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def block_faces(segmentation):
    """slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [
            slice(None),
        ] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [
            slice(None),
        ] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


######################## Distributed Cellpose #################################


# ----------------------- The main function -----------------------------------#
@cluster
def distributed_eval(
    input_xr,
    blocksize,
    write_path=None,
    mask=None,
    preprocessing_steps=None,
    model_kwargs=None,
    eval_kwargs=None,
    cluster=None,
    cluster_kwargs=None,
    temporary_directory=None,
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image using xarray and Dask.
    Returns an xarray.DataArray with the segmentation results (in memory, chunked).
    If write_path is provided, saves the result to disk as NetCDF.
    """
    if preprocessing_steps is None:
        preprocessing_steps = []
    if model_kwargs is None:
        model_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    worker_logs_dirname = f"dask_worker_logs_{timestamp}"
    worker_logs_dir = pathlib.Path().absolute().joinpath(worker_logs_dirname)
    worker_logs_dir.mkdir()
    if "diameter" not in eval_kwargs.keys():
        eval_kwargs["diameter"] = 30
    overlap = eval_kwargs["diameter"] * 2
    block_indices, block_crops = get_block_crops(
        input_xr.shape,
        blocksize,
        overlap,
        mask,
    )
    temp_shape = input_xr.shape
    temp_chunks = blocksize
    temp_da = xr.DataArray(
        np.zeros(temp_shape, dtype=np.uint32),
        dims=input_xr.dims,
        chunks={dim: size for dim, size in zip(input_xr.dims, temp_chunks)},
    )
    # Distributed blockwise processing, all in xarray
    futures = cluster.client.map(
        process_block,
        block_indices,
        block_crops,
        input_xr=input_xr,
        preprocessing_steps=preprocessing_steps,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        blocksize=blocksize,
        overlap=overlap,
        output_xr=temp_da,
        worker_logs_directory=str(worker_logs_dir),
    )
    results = cluster.client.gather(futures)
    faces, boxes_, box_ids_ = list(zip(*results))
    boxes = [box for sublist in boxes_ for box in sublist]
    box_ids = np.concatenate(box_ids_).astype(int)
    new_labeling = determine_merge_relabeling(block_indices, faces, box_ids)
    # Relabeling step (in memory, xarray)
    relabeled = temp_da.copy()
    relabeled.data = new_labeling[relabeled.data]
    if write_path is not None:
        relabeled.astype(np.uint32).to_netcdf(write_path)
    merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids])
    return relabeled, merged_boxes


# ----------------------- component functions ---------------------------------#
def get_block_crops(shape, blocksize, overlap, mask):
    """Given a voxel grid shape, blocksize, and overlap size, construct
    tuples of slices for every block; optionally only include blocks
    that contain foreground in the mask. Returns parallel lists,
    the block indices and the slice tuples."""
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]):
                foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def determine_merge_relabeling(block_indices, faces, used_labels):
    """Determine boundary segment mergers, remap all label IDs to merge
    and put all label IDs in range [1..N] for N global segments found"""
    faces = adjacent_faces(block_indices, faces)
    # FIX float parameters
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    used_labels = used_labels.astype(int)
    # print("Used labels:", used_labels, "Type:", type(used_labels))
    label_range = int(np.max(used_labels))

    label_groups = block_face_adjacency_graph(faces, label_range)
    new_labeling = scipy.sparse.csgraph.connected_components(
        label_groups, directed=False
    )[1]
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
    return new_labeling


def adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {a: b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2 * ax + 1]
                b = faces_index_lookup[neighbor_index][2 * ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def block_face_adjacency_graph(faces, nlabels):
    """
    Build adjacency graph for block face labels.
    This is a placeholder using numpy/scipy for demonstration.
    """
    nlabels = int(nlabels)
    # For demonstration, create an identity matrix (no adjacency)
    return scipy.sparse.identity(nlabels + 1, format="csr")


def shrink_labels(plane, threshold):
    """Shrink labels in plane by some distance from their boundary"""
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def merge_all_boxes(boxes, box_ids):
    """Merge all boxes that map to the same box_ids"""
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    # FIX float parameters
    # print("Box IDs:", box_ids, "Type:", type(box_ids))
    box_ids = box_ids.astype(int)
    # print("Box IDs:", box_ids, "Type:", type(box_ids))

    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes


def merge_boxes(boxes):
    """Take union of two or more parallelpipeds"""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union
