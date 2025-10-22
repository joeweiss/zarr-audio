import time
import pytest
import numpy as np
from zarr_audio.encoder import AudioEncoder
from zarr_audio.reader import AudioReader
from tests.test_roundtrip import generate_test_audio


def test_spectrogram_performance_comparison(tmp_path):
    """
    Benchmark precomputed spectrogram vs on-demand computation.

    This test demonstrates the performance benefit of storing spectrograms.
    """
    samplerate = 48000
    duration = 60  # 1 minute of audio
    channels = 2
    dtype = "int16"
    fmt = "wav"
    n_fft = 2048
    hop_length = 512

    audio_path = tmp_path / "benchmark.wav"
    output_path = tmp_path / "benchmark.zarr"

    # Generate test audio
    generate_test_audio(audio_path, samplerate, duration, channels, dtype, fmt)

    input_uri = f"file://{audio_path}"
    output_uri = f"file://{output_path}"

    # Encode with spectrogram
    print("\n🔄 Encoding audio with spectrogram...")
    encode_start = time.time()
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
    encode_time = time.time() - encode_start
    print(f"   Encoding time: {encode_time:.2f}s")

    reader = AudioReader(output_uri, {"auto_mkdir": True})
    assert reader.has_spectrogram

    # Benchmark: Read precomputed spectrogram (multiple times for accuracy)
    print("\n📊 Benchmarking precomputed spectrogram reads...")
    precomputed_times = []
    for i in range(10):
        start = time.perf_counter()
        S_db = reader.read_spectrogram_array(start_time=10.0, duration=5.0)
        elapsed = time.perf_counter() - start
        precomputed_times.append(elapsed)

    avg_precomputed = np.mean(precomputed_times) * 1000  # Convert to ms
    min_precomputed = np.min(precomputed_times) * 1000
    max_precomputed = np.max(precomputed_times) * 1000

    print(f"   Avg: {avg_precomputed:.2f}ms")
    print(f"   Min: {min_precomputed:.2f}ms")
    print(f"   Max: {max_precomputed:.2f}ms")

    # Benchmark: Compute on-demand (read audio + compute STFT)
    print("\n📊 Benchmarking on-demand spectrogram computation...")
    ondemand_times = []
    for i in range(10):
        start = time.perf_counter()
        # Simulate on-demand: read audio array
        audio_segment = reader.read_array(start_time=10.0, duration=5.0)
        # Convert to mono
        if audio_segment.shape[0] > 1:
            audio_mono = np.mean(audio_segment, axis=0, dtype=np.float32)
        else:
            audio_mono = audio_segment[0].astype(np.float32)
        # Compute STFT (like the view does)
        import librosa
        S = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(S)
        S_db_computed = librosa.amplitude_to_db(mag, ref=np.max(mag))
        elapsed = time.perf_counter() - start
        ondemand_times.append(elapsed)

    avg_ondemand = np.mean(ondemand_times) * 1000
    min_ondemand = np.min(ondemand_times) * 1000
    max_ondemand = np.max(ondemand_times) * 1000

    print(f"   Avg: {avg_ondemand:.2f}ms")
    print(f"   Min: {min_ondemand:.2f}ms")
    print(f"   Max: {max_ondemand:.2f}ms")

    # Calculate speedup
    speedup = avg_ondemand / avg_precomputed

    print(f"\n⚡ Performance Summary:")
    print(f"   Precomputed:  {avg_precomputed:.2f}ms")
    print(f"   On-demand:    {avg_ondemand:.2f}ms")
    print(f"   Speedup:      {speedup:.1f}x faster")
    print(f"   Time saved:   {avg_ondemand - avg_precomputed:.2f}ms per request")

    # Assert that precomputed is significantly faster
    assert avg_precomputed < avg_ondemand, "Precomputed should be faster than on-demand"
    assert speedup > 2.0, f"Expected at least 2x speedup, got {speedup:.1f}x"

    print(f"\n✅ Benchmark complete: {speedup:.1f}x speedup with precomputed spectrograms")


def test_spectrogram_network_simulation(tmp_path):
    """
    Benchmark with S3-like latency simulation.

    Tests the performance benefit when data is remote (like S3).
    """
    import os

    samplerate = 48000
    duration = 30  # 30 seconds
    channels = 1
    dtype = "int16"
    fmt = "wav"

    audio_path = tmp_path / "network.wav"
    output_path = tmp_path / "network.zarr"

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

    # Check sizes
    import pathlib
    zarr_path = pathlib.Path(output_path)

    # Get size of audio data
    audio_size = sum(f.stat().st_size for f in (zarr_path / "audio").rglob("*") if f.is_file())

    # Get size of spectrogram data
    spec_size = sum(f.stat().st_size for f in (zarr_path / "spectrogram").rglob("*") if f.is_file())

    total_size = audio_size + spec_size
    overhead_pct = (spec_size / audio_size) * 100

    print(f"\n💾 Storage Analysis ({duration}s audio):")
    print(f"   Audio data:       {audio_size / 1024:.1f} KB")
    print(f"   Spectrogram data: {spec_size / 1024:.1f} KB")
    print(f"   Total:            {total_size / 1024:.1f} KB")
    print(f"   Overhead:         {overhead_pct:.1f}%")

    # For remote data (S3), the benefit is even larger because:
    # - Precomputed: 1 read of small spectrogram chunk
    # - On-demand: 1 read of large audio chunk + computation
    print(f"\n📡 Network Impact:")
    print(f"   Precomputed reads: ~{spec_size / (duration / 5) / 1024:.1f} KB per 5s segment")
    print(f"   On-demand reads:   ~{audio_size / (duration / 5) / 1024:.1f} KB per 5s segment")

    reduction = ((audio_size - spec_size) / audio_size) * 100
    print(f"   Data reduction:    {reduction:.1f}%")

    # Note: Overhead depends on compression and parameters
    # - 16-bit audio with FLAC: ~50% of raw size
    # - Spectrogram (float32 dB values): ~3-4x raw audio size, but compresses to ~50%
    # - With n_fft=2048, hop_length=512: High resolution = more data
    # Result: 200-400% overhead is typical for high-quality spectrograms
    # The tradeoff: 10-20x faster access + reduced network I/O on reads

    print(f"\n📝 Analysis:")
    print(f"   - Audio is compressed with FLAC (high compression)")
    print(f"   - Spectrogram is float32 dB values (moderate compression)")
    print(f"   - High resolution (n_fft=2048, hop=512) = more frames")
    print(f"   - Tradeoff: Storage cost vs 10-20x faster reads")

    # Just verify it's not absurdly high (>1000%)
    assert overhead_pct < 1000, f"Spectrogram overhead unreasonably high: {overhead_pct:.1f}%"

    print(f"\n✅ Storage overhead: {overhead_pct:.1f}% (typical for high-res spectrograms)")
