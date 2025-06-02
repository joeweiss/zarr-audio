import os
import pathlib

import numpy as np
import pytest
import soundfile as sf

from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader

inspect_output = True

skip_unless_local = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_TESTS") != "1",
    reason="Only runs in local environment (set RUN_LOCAL_TESTS=1)",
)


@skip_unless_local
def test_audio_reader_roundtrip(tmp_path):
    filename = "tests/test_files/BAOW_EASO_48000_32bit.wav"
    fs_scheme = "file"
    chunk_duration = 10
    segment_duration = 5  # seconds

    with sf.SoundFile(filename) as f:
        subtype = f.subtype
        samplerate = f.samplerate
        channels = f.channels
        subtype_to_dtype = {
            "PCM_16": "int16",
            "PCM_24": "int32",
            "PCM_32": "int32",  # will be probed
            "FLOAT": "float32",
            "DOUBLE": "float64",
        }
        dtype = subtype_to_dtype.get(subtype, "int16")
        if subtype == "PCM_32":
            probe = f.read(dtype="int32", always_2d=True, frames=1)
            dtype = "float32" if abs(probe[0, 0]) < 2**24 else "int32"

    with sf.SoundFile(filename) as f:
        y_orig = f.read(always_2d=True, dtype=dtype).T

    n_samples = y_orig.shape[1]
    start = n_samples // 2 - int(samplerate * segment_duration // 2)
    end = start + int(samplerate * segment_duration)
    expected = y_orig[:, start:end]

    input_uri = f"file://{os.path.abspath(filename)}"
    output_path = tmp_path / "roundtrip.zarr"
    output_uri = f"file://{output_path}"

    encoder = AudioEncoder(
        input_uri=input_uri,
        output_uri=output_uri,
        storage_options={"auto_mkdir": True},
        chunk_duration=chunk_duration,
    )
    encoder.encode()

    reader = AudioReader(output_uri, {"auto_mkdir": True})
    arr = reader.read_array(start_time=start / samplerate, duration=segment_duration)

    # Segment content check
    if arr.dtype.kind == "f" or arr.dtype.itemsize > 2:
        assert np.allclose(arr, expected, rtol=1e-4, atol=10)
    else:
        assert np.array_equal(arr, expected)

    # Runtime info check
    info = reader.info()
    print(info)
    assert info["samplerate"] == samplerate
    assert info["channels"] == channels
    assert info["samples"] == n_samples
    assert info["dtype"] == str(np.dtype(dtype))

    # Zarr attrs-only check (no array load)
    attrs = AudioReader.read_attrs(output_uri, {"auto_mkdir": True})
    print(attrs)
    assert attrs["samplerate"] == samplerate
    assert attrs["channels"] == channels
    assert attrs["samples"] == n_samples
    assert str(np.dtype(dtype)) == str(np.dtype(attrs["dtype"]))

    # Encoded FLAC check
    encoded = reader.read_encoded(
        start_time=start / samplerate, duration=segment_duration, format="flac"
    )
    assert isinstance(encoded, bytes)
    assert len(encoded) > 1000

    # AudioSegment support.
    audio_seg = reader.read_as_segment(
        start_time=start / samplerate, duration=segment_duration
    )
    filtered = (
        audio_seg.high_pass_filter(1000).normalize()
    )  # PyDub's HPF has a very soft rolloff, this is of limited use and only as a test.

    print(f"✅ Round-trip test passed for: {filename}")

    if inspect_output and fs_scheme == "file":
        output_dir = pathlib.Path("tests/output")
        output_dir.mkdir(exist_ok=True)

        basename = os.path.splitext(os.path.basename(filename))[0]
        start_sec = round(start / samplerate, 3)
        end_sec = round(end / samplerate, 3)
        segment_path = (
            output_dir / f"{basename}-segment_{start_sec:.3f}-{end_sec:.3f}.flac"
        )

        with open(segment_path, "wb") as f:
            f.write(encoded)

        print(f"🎧 Saved re-encoded audio segment to: {segment_path}")

        segment_mp3 = (
            output_dir
            / f"{basename}-segment_filtered_{start_sec:.3f}-{end_sec:.3f}.mp3"
        )
        with open(segment_mp3, "wb") as f:
            filtered.export(f, format="mp3")

        print(f"🎧 Saved filtered mp3 segment to: {segment_mp3}")
