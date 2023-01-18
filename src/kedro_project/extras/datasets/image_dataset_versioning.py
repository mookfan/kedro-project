from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np
from PIL import Image

from kedro.io import AbstractVersionedDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path, Version


class ImageDataSet(AbstractVersionedDataSet[np.ndarray, np.ndarray]):
    """``ImageDataSet`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> ImageDataSet(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str, version: Version = None):
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
            version: The version of the dataset being saved and loaded.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        with self._fs.open(load_path) as f: # original: flag mode="r"
            image = Image.open(f).convert("RGBA")
            return np.asarray(image)

    def _save(self, data: np.ndarray) -> None:
        """Saves image data to the specified filepath."""
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        print("save_path", save_path)
        with self._fs.open(save_path, mode="wb") as f:
            image = Image.fromarray(data)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(f)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath, version=self._version, protocol=self._protocol
        )