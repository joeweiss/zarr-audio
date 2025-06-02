import glob
import os
import pathlib
import shutil

import fsspec
import numpy as np
import pytest
import soundfile as sf
import zarr

from zarr_audio import AudioEncoder  # adjust import as needed

# watchmedo shell-command -c "pytest -x --capture=no tests/test_encode_class.py" --recursive -W

skip_unless_local = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_TESTS") != "1",
    reason="Only runs in local environment (set RUN_LOCAL_TESTS=1)",
)


inspect_output = True  # Set to False to skip copying .zarr files to tests/output/


def get_dir_size(path):
    """Return total size of directory in bytes (portable)."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


def get_audio_test_files():
    """Discover all .wav and .flac files in tests/test_files/."""
    return sorted(
        glob.glob("tests/test_files/*.mp3")
        + glob.glob("tests/test_files/*.flac")
        + glob.glob("tests/test_files/*.wav")
    )


# Add a chunk_duration param
@skip_unless_local
@pytest.mark.parametrize("chunk_duration", [5, 11])
@pytest.mark.parametrize("fs_scheme", ["file", "memory"])
@pytest.mark.parametrize("encoding_read_duration", [1, 60 * 10])
@pytest.mark.parametrize("filename", get_audio_test_files())
def test_audio_encoder_roundtrip_explicit_fsspec(
    filename, fs_scheme, encoding_read_duration, chunk_duration, tmp_path
):
    print(filename)
    with sf.SoundFile(filename) as f:
        subtype = f.subtype
        samplerate = f.samplerate
        frames = f.frames
        channels = f.channels

    subtype_to_dtype = {
        "PCM_16": ("int16", 16),
        "PCM_24": ("int32", 24),
        "FLOAT": ("float32", 32),
        "DOUBLE": ("float64", 64),
    }

    if subtype == "PCM_32":
        with sf.SoundFile(filename) as f:
            probe = f.read(dtype="int32", always_2d=True, frames=1)
            sample = probe[0, 0]
            if abs(sample) < 2**24:
                dtype = "float32"
            else:
                dtype = "int32"
    else:
        dtype = subtype_to_dtype.get(subtype, ("int16", 16))[0]

    with sf.SoundFile(filename) as f:
        y_orig = f.read(always_2d=True, dtype=dtype).T

    n_samples = y_orig.shape[1]
    segment_len = int(samplerate * 5)
    start = n_samples // 2 - segment_len // 2
    end = start + segment_len

    if fs_scheme == "file":
        input_uri = f"file://{os.path.abspath(filename)}"
        output_path = tmp_path / "encoded.zarr"
        output_uri = f"file://{output_path}"
    elif fs_scheme == "memory":
        input_uri = f"file://{os.path.abspath(filename)}"
        output_uri = "memory://test.zarr"
    else:
        raise ValueError("Unknown fs_scheme")

    encoder = AudioEncoder(
        input_uri=input_uri,
        output_uri=output_uri,
        storage_options={"auto_mkdir": True},
        chunk_duration=chunk_duration,
        encoding_read_duration=encoding_read_duration,
    )
    encoder.encode()

    store = fsspec.get_mapper(output_uri)
    z = zarr.open_group(store, mode="r")
    z_segment = z["audio"][:, start:end]
    expected = y_orig[:, start:end]

    print(f"🧪 Comparing segment range {start}-{end}")

    if z_segment.dtype.kind in {"f"} or z_segment.dtype.itemsize > 2:
        assert np.allclose(z_segment, expected, rtol=1e-4, atol=10), (
            f"Segment mismatch for file: {filename} with chunk_duration={chunk_duration}"
        )
    else:
        assert np.array_equal(z_segment, expected), (
            f"Segment mismatch for file: {filename} with chunk_duration={chunk_duration}"
        )

    assert z.attrs["compression"] in {"flac", "blosc"}

    print(
        f"\n[{fs_scheme.upper()} → {os.path.basename(filename)} | Chunk: {chunk_duration}s]"
    )
    print(f"  Samples:       {n_samples}")
    print(f"  Sample rate:   {samplerate}")
    print(f"  Segment match: ✅")

    if inspect_output and fs_scheme == "file":
        output_dir = pathlib.Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        target_name = (
            f"{pathlib.Path(filename).stem}-{fs_scheme}-chunk{chunk_duration}.zarr"
        )
        target_path = output_dir / target_name
        shutil.copytree(output_path, target_path, dirs_exist_ok=True)
        print(f"📁 Zarr copy saved to: {target_path}")
