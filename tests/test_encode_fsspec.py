import pytest
import shutil
import pathlib
import numpy as np
import soundfile as sf
import zarr
import fsspec
import os

from zarr_audio import AudioEncoder  # adjust if needed

inspect_output = True  # Set to False to skip copying .zarr files to tests/output/

TEST_BUCKET = os.getenv("TEST_BUCKET")

# Replace this with your actual S3 URIs
S3_AUDIO_URIS = [
    f"s3://{TEST_BUCKET}/HOME/BAOW_EASO_48000_16bit.wav",
    f"s3://{TEST_BUCKET}/HOME/soundscape.wav",
    f"s3://{TEST_BUCKET}/HOME/long.flac",
    f"s3://{TEST_BUCKET}/HOME/long.mp3",
    # "s3://birdnetlib-django-test-storage/HOME/20230716_180000.wav",
]

skip_unless_local = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_TESTS") != "1",
    reason="Only runs in local environment (set RUN_LOCAL_TESTS=1)",
)


@skip_unless_local
@pytest.mark.parametrize("chunk_duration", [5])
@pytest.mark.parametrize("input_uri", S3_AUDIO_URIS)
def test_audio_encoder_roundtrip_s3(input_uri, chunk_duration, tmp_path):
    # Download locally to probe metadata and read original
    print("test_audio_encoder_roundtrip_s3", input_uri)
    fs, path = fsspec.core.url_to_fs(input_uri, anon=True)
    with fs.open(path, "rb") as f:
        with sf.SoundFile(f) as sf_info:
            subtype = sf_info.subtype
            samplerate = sf_info.samplerate
            frames = sf_info.frames
            channels = sf_info.channels
            dtype = "int16"  # fallback default
            if subtype == "PCM_32":
                f.seek(0)
                probe = sf_info.read(dtype="int32", always_2d=True, frames=1)
                sample = probe[0, 0]
                if abs(sample) < 2**24:
                    dtype = "float32"
                else:
                    dtype = "int32"
            else:
                subtype_to_dtype = {
                    "PCM_16": ("int16", 16),
                    "PCM_24": ("int32", 24),
                    "FLOAT": ("float32", 32),
                    "DOUBLE": ("float64", 64),
                }
                dtype = subtype_to_dtype.get(subtype, ("int16", 16))[0]

    # Read full audio for comparison
    with fs.open(path, "rb") as f:
        with sf.SoundFile(f) as sf_info:
            y_orig = sf_info.read(always_2d=True, dtype=dtype).T

    n_samples = y_orig.shape[1]
    segment_len = int(samplerate * 5)
    start = n_samples // 2 - segment_len // 2
    end = start + segment_len

    output_path = tmp_path / "encoded.zarr"
    output_uri = f"file://{output_path}"

    encoder = AudioEncoder(
        input_uri=input_uri,
        output_uri=output_uri,
        storage_options={"anon": True},  # update with real credentials if needed
        chunk_duration=chunk_duration,
    )
    encoder.encode()

    store = fsspec.get_mapper(output_uri)
    z = zarr.open_group(store, mode="r")

    # ✅ Check number of channels
    assert z["audio"].shape[0] == y_orig.shape[0], (
        f"Channel mismatch for file: {input_uri} — original: {y_orig.shape[0]}, encoded: {z['audio'].shape[0]}"
    )

    z_segment = z["audio"][:, start:end]
    expected = y_orig[:, start:end]

    if z_segment.dtype.kind in {"f"} or z_segment.dtype.itemsize > 2:
        assert np.allclose(z_segment, expected, rtol=1e-4, atol=10), (
            f"Segment mismatch for file: {input_uri} with chunk_duration={chunk_duration}"
        )
    else:
        assert np.array_equal(z_segment, expected), (
            f"Segment mismatch for file: {input_uri} with chunk_duration={chunk_duration}"
        )

    assert z.attrs["compression"] in {"flac", "blosc"}

    print(f"\n[S3 → {input_uri} | Chunk: {chunk_duration}s]")
    print(f"  Samples:       {n_samples}")
    print(f"  Sample rate:   {samplerate}")
    print(f"  Segment match: ✅")

    if inspect_output:
        output_dir = pathlib.Path("tests/output")
        output_dir.mkdir(exist_ok=True)
        target_name = f"{pathlib.Path(input_uri).stem}-s3-chunk{chunk_duration}.zarr"
        target_path = output_dir / target_name
        shutil.copytree(output_path, target_path, dirs_exist_ok=True)
        print(f"📁 Zarr copy saved to: {target_path}")
