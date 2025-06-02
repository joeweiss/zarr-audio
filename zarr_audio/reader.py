import io
from typing import Dict, Literal, Optional

import fsspec
import numpy as np
import soundfile as sf
import zarr
from pydub import AudioSegment


class AudioReader:
    """
    Reader for Zarr-encoded audio arrays with support for raw NumPy extraction
    and re-encoding into audio formats like FLAC, MP3, or WAV.

    Assumes the Zarr group has an 'audio' dataset and metadata in attrs including
    samplerate, bit_depth, compression, etc.
    """

    def __init__(self, zarr_uri: str, storage_options: Optional[dict] = None):
        """
        Initialize the reader from a Zarr URI.

        Args:
            zarr_uri: fsspec-compatible path to the Zarr group.
            storage_options: Optional dictionary of fsspec credentials or options.
        """
        self.zarr_uri = zarr_uri
        self.storage_options = storage_options or {}
        store = fsspec.get_mapper(zarr_uri, **self.storage_options)
        self.z = zarr.open_group(store, mode="r")
        self.audio = self.z["audio"]
        self.samplerate: int = self.z.attrs["samplerate"]
        self.channels: int = self.audio.shape[0]

    def read_array(self, start_time: float, duration: float) -> np.ndarray:
        """
        Return a (channels, samples) NumPy array from the given time range.

        Args:
            start_time: Start time in seconds.
            duration: Duration in seconds.

        Returns:
            NumPy array of shape (channels, samples).
        """
        start = int(start_time * self.samplerate)
        end = int((start_time + duration) * self.samplerate)
        return self.audio[:, start:end]

    def read_encoded(
        self,
        start_time: float,
        duration: float,
        format: Literal["flac", "mp3"] = "flac",
    ) -> bytes:
        """
        Read and encode a time segment as audio bytes in the requested format.

        Args:
            start_time: Start time in seconds.
            duration: Duration in seconds.
            format: Audio format to encode as ("flac" or "mp3").

        Returns:
            Byte stream of encoded audio.
        """
        segment = self.read_array(start_time, duration).T  # (samples, channels)
        buf = io.BytesIO()
        sf.write(buf, segment, self.samplerate, format=format)
        buf.seek(0)
        return buf.read()

    def info(self) -> Dict[str, object]:
        """
        Return metadata about the Zarr-encoded audio.

        Returns:
            Dictionary with sample rate, channels, samples, duration, dtype, and compression info.
        """
        return {
            "samplerate": self.samplerate,
            "channels": self.audio.shape[0],
            "samples": self.audio.shape[1],
            "duration_sec": self.audio.shape[1] / self.samplerate,
            "dtype": str(self.audio.dtype),
            "bit_depth": self.z.attrs.get("bit_depth"),
            "compression": self.z.attrs.get("compression"),
        }

    @classmethod
    def read_attrs(
        cls, zarr_uri: str, storage_options: Optional[dict] = None
    ) -> Dict[str, object]:
        """
        Read only the metadata (attrs) from a Zarr URI without loading the audio dataset.

        Args:
            zarr_uri: fsspec-compatible path to the Zarr group.
            storage_options: Optional dictionary of fsspec credentials or options.

        Returns:
            Dictionary of Zarr group attributes.
        """
        storage_options = storage_options or {}
        store = fsspec.get_mapper(zarr_uri, **storage_options)
        z = zarr.open_group(store, mode="r")
        return dict(z.attrs)

    def read_as_segment(
        self, start_time: float, duration: float, target_format: str = "wav"
    ) -> AudioSegment:
        """
        Return a Pydub AudioSegment for applying effects, playback, or re-export.

        Args:
            start_time: Start time in seconds.
            duration: Duration in seconds.
            target_format: Format to encode to for intermediate buffer (e.g. "wav").

        Returns:
            pydub.AudioSegment object.
        """
        pcm = self.read_array(start_time, duration).T  # (samples, channels)
        buf = io.BytesIO()
        sf.write(buf, pcm, self.samplerate, format=target_format)
        buf.seek(0)
        return AudioSegment.from_file(buf, format=target_format)
