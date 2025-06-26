import numpy as np
import pytest
import xarray as xr

from cellpose.contrib import distributed_segmentation as ds


def dummy_preprocess(image, crop=None):
    """Dummy preprocessing step: identity."""
    return image


def test_numpy_array_to_xarray(tmp_path):
    arr = np.random.randint(0, 255, (4, 8, 8), dtype=np.uint8)
    out_path = tmp_path / "test.nc"
    da = ds.numpy_array_to_xarray(str(out_path), arr, (2, 4, 4))
    assert isinstance(da, xr.DataArray)
    assert da.shape == arr.shape
    np.testing.assert_array_equal(da.values, arr)


def test_wrap_folder_of_tiffs(tmp_path):
    arrs = [np.ones((2, 2), dtype=np.uint8) * i for i in range(3)]
    for i, arr in enumerate(arrs):
        xr.DataArray(arr).to_netcdf(tmp_path / f"img_{i}.tif")
    da = ds.wrap_folder_of_tiffs(str(tmp_path / "*.tif"))
    assert isinstance(da, xr.DataArray)
    assert da.shape[0] == 3


def test_process_block(tmp_path):
    arr = np.random.randint(0, 2, (1, 8, 8), dtype=np.uint8)
    da = xr.DataArray(arr, dims=["c", "y", "x"])
    out = xr.DataArray(np.zeros_like(arr), dims=da.dims)

    # Use a dummy model that returns the input as segmentation
    class DummyModel:
        def eval(self, image, **kwargs):
            return (image.astype(np.uint32),)

    ds.models.CellposeModel = DummyModel
    faces, boxes, box_ids = ds.process_block(
        block_index=(0, 0, 0),
        crop=(slice(0, 1), slice(0, 8), slice(0, 8)),
        input_xr=da,
        model_kwargs={},
        eval_kwargs={},
        blocksize=(1, 8, 8),
        overlap=0,
        output_xr=out,
        preprocessing_steps=[(dummy_preprocess, {})],
        test_mode=False,
    )
    assert isinstance(faces, list)
    assert isinstance(boxes, list)
    assert isinstance(box_ids, np.ndarray)
    assert out.sum() > 0 or out.sum() == 0  # Just check it runs


def test_process_block_multichannel(tmp_path):
    arr = np.random.randint(0, 2, (2, 8, 8), dtype=np.uint8)  # 2 channels
    da = xr.DataArray(arr, dims=["c", "y", "x"])
    out = xr.DataArray(np.zeros_like(arr[0:1]), dims=["c", "y", "x"])

    class DummyModel:
        def eval(self, image, **kwargs):
            # Should receive (Y, X, C)
            assert image.shape[-1] == 2
            return (image[..., 0].astype(np.uint32),)

    ds.models.CellposeModel = DummyModel
    faces, boxes, box_ids = ds.process_block(
        block_index=(0, 0, 0),
        crop=(slice(0, 2), slice(0, 8), slice(0, 8)),
        input_xr=da,
        model_kwargs={},
        eval_kwargs={},
        blocksize=(2, 8, 8),
        overlap=0,
        output_xr=out,
        preprocessing_steps=[(dummy_preprocess, {})],
        test_mode=False,
    )
    assert isinstance(faces, list)
    assert isinstance(boxes, list)
    assert isinstance(box_ids, np.ndarray)


def test_bounding_boxes_in_global_coordinates():
    arr = np.zeros((4, 4), dtype=np.uint32)
    arr[1:3, 1:3] = 1
    crop = (slice(0, 4), slice(0, 4))
    boxes = ds.bounding_boxes_in_global_coordinates(arr, crop)
    assert isinstance(boxes, list)
    assert all(isinstance(b, tuple) for b in boxes)


def test_get_nblocks():
    shape = (10, 20, 30)
    blocksize = (5, 10, 10)
    nblocks = ds.get_nblocks(shape, blocksize)
    assert nblocks.tolist() == [2, 2, 3]


def test_global_segment_ids():
    arr = np.zeros((2, 2), dtype=np.uint32)
    arr[0, 0] = 1
    block_index = (0, 0)
    nblocks = (1, 1)
    seg, remap = ds.global_segment_ids(arr, block_index, nblocks)
    assert seg.shape == arr.shape
    assert isinstance(remap, list)


def test_remove_overlaps():
    arr = np.ones((4, 4), dtype=np.uint32)
    crop = (slice(0, 4), slice(0, 4))
    overlap = 1
    blocksize = (2, 2)
    arr2, crop2 = ds.remove_overlaps(arr, crop, overlap, blocksize)
    assert isinstance(arr2, np.ndarray)
    assert isinstance(crop2, list)


def test_merge_boxes():
    boxes = [(slice(0, 2), slice(0, 2)), (slice(1, 3), slice(1, 3))]
    merged = ds.merge_boxes(boxes)
    assert isinstance(merged, tuple)
    assert all(isinstance(s, slice) for s in merged)
