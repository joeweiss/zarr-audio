import numpy as np
import pytest
from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader
from tests.test_roundtrip import generate_test_audio


def test_spectrogram_roundtrip(tmp_path):
    """Test that spectrograms are computed, stored, and retrieved correctly."""
    samplerate = 48000
    duration = 10
    channels = 2
    dtype = "int16"
    fmt = "wav"
    n_fft = 2048
    hop_length = 512

    audio_path = tmp_path / f"test_spec.{fmt}"
    output_path = tmp_path / "test_spec.zarr"

    # Generate test audio
    y = generate_test_audio(audio_path, samplerate, duration, channels, dtype, fmt)

    input_uri = f"file://{audio_path}"
    output_uri = f"file://{output_path}"

    # Encode with spectrogram computation
    encoder = AudioEncoder(
        input_uri=input_uri,
        output_uri=output_uri,
        storage_options={"auto_mkdir": True},
        chunk_duration=10,
        compute_spectrogram=True,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    encoder.encode()

    # Read back and verify spectrogram exists
    reader = AudioReader(output_uri, {"auto_mkdir": True})
    assert reader.has_spectrogram, "Spectrogram should be available"

    # Check spectrogram parameters
    params = reader.get_spectrogram_params()
    assert params["n_fft"] == n_fft
    assert params["hop_length"] == hop_length
    assert params["freq_bins"] == n_fft // 2 + 1
    assert params["time_frames"] > 0

    # Read a spectrogram segment
    start_time = 2.0
    segment_duration = 3.0
    S_db = reader.read_spectrogram_array(start_time=start_time, duration=segment_duration)

    # Verify shape
    expected_time_frames = int((segment_duration * samplerate) / hop_length)
    assert S_db.shape[0] == n_fft // 2 + 1
    # Allow some tolerance for frame boundaries
    assert abs(S_db.shape[1] - expected_time_frames) <= 2

    # Verify dB values are in reasonable range
    assert S_db.dtype == np.float32
    assert np.all(np.isfinite(S_db))
    assert S_db.min() < 0  # dB scale should have negative values
    assert S_db.max() <= 0  # dB scale normalized to max ref

    print(f"✅ Spectrogram roundtrip test passed")
    print(f"   Shape: {S_db.shape}")
    print(f"   dB range: [{S_db.min():.2f}, {S_db.max():.2f}]")


def test_spectrogram_disabled(tmp_path):
    """Test that spectrogram computation can be disabled."""
    samplerate = 48000
    duration = 5
    channels = 1
    dtype = "int16"
    fmt = "wav"

    audio_path = tmp_path / "test_no_spec.wav"
    output_path = tmp_path / "test_no_spec.zarr"

    generate_test_audio(audio_path, samplerate, duration, channels, dtype, fmt)

    input_uri = f"file://{audio_path}"
    output_uri = f"file://{output_path}"

    # Encode WITHOUT spectrogram computation
    encoder = AudioEncoder(
        input_uri=input_uri,
        output_uri=output_uri,
        storage_options={"auto_mkdir": True},
        chunk_duration=10,
        compute_spectrogram=False,
    )
    encoder.encode()

    # Read back and verify spectrogram does NOT exist
    reader = AudioReader(output_uri, {"auto_mkdir": True})
    assert not reader.has_spectrogram, "Spectrogram should not be available"

    # Verify attempting to read spectrogram raises error
    with pytest.raises(ValueError, match="Spectrogram data not available"):
        reader.read_spectrogram_array(start_time=0, duration=1.0)

    with pytest.raises(ValueError, match="Spectrogram data not available"):
        reader.get_spectrogram_params()

    print(f"✅ Spectrogram disabled test passed")


def test_spectrogram_boundary_cases(tmp_path):
    """Test spectrogram reading at audio boundaries."""
    samplerate = 48000
    duration = 10
    channels = 1
    dtype = "int16"
    fmt = "wav"

    audio_path = tmp_path / "test_boundary.wav"
    output_path = tmp_path / "test_boundary.zarr"

    generate_test_audio(audio_path, samplerate, duration, channels, dtype, fmt)

    input_uri = f"file://{audio_path}"
    output_uri = f"file://{output_path}"

    encoder = AudioEncoder(
        input_uri=input_uri,
        output_uri=output_uri,
        storage_options={"auto_mkdir": True},
        compute_spectrogram=True,
    )
    encoder.encode()

    reader = AudioReader(output_uri, {"auto_mkdir": True})

    # Test reading from start
    S_start = reader.read_spectrogram_array(start_time=0, duration=1.0)
    assert S_start.shape[1] > 0

    # Test reading from end
    S_end = reader.read_spectrogram_array(start_time=duration - 1.0, duration=1.0)
    assert S_end.shape[1] > 0

    # Test reading entire spectrogram
    S_full = reader.read_spectrogram_array(start_time=0, duration=duration)
    params = reader.get_spectrogram_params()
    assert S_full.shape[1] == params["time_frames"]

    # Test reading beyond boundaries (should clamp)
    S_beyond = reader.read_spectrogram_array(start_time=duration - 0.5, duration=2.0)
    assert S_beyond.shape[1] > 0  # Should return what's available

    print(f"✅ Spectrogram boundary cases test passed")
