import io
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from chorus.types import AudioResult, encode_response


class AudioEncodingTests(unittest.TestCase):
    def test_flushes_partial_mp3_frames_with_independent_rates_and_channels(self):
        inputs = []
        for rate, frequencies, count in (
            (24_000, (750,), 241),
            (48_000, (750, 1500), 1441),
        ):
            time = np.arange(count) / rate
            samples = np.stack(
                [
                    0.5 * np.sin(2 * np.pi * frequency * time) * np.hanning(count)
                    for frequency in frequencies
                ],
                axis=1,
            )
            wav = io.BytesIO()
            sf.write(wav, samples, rate, format="WAV", subtype="PCM_16")
            audio = AudioResult(wav.getvalue(), rate, count / rate)
            inputs.append((audio, samples))

        with ThreadPoolExecutor(max_workers=2) as pool:
            encoded = list(
                pool.map(lambda item: encode_response(item[0], "mp3"), inputs)
            )
        for (audio, expected), (content, reported_rate) in zip(inputs, encoded):
            with self.subTest(rate=audio.sample_rate, channels=expected.shape[1]):
                actual, rate = sf.read(io.BytesIO(content), always_2d=True)
                self.assertEqual((reported_rate, rate), (audio.sample_rate,) * 2)
                self.assertEqual(actual.shape[1], expected.shape[1])
                self.assertGreaterEqual(len(actual), len(expected))
                # Raw MP3 includes encoder delay. Locate the signal, then compare
                # every channel so missing flushes or mixed requests lose audio.
                correlation = np.correlate(actual[:, 0], expected[:, 0], "valid")
                offset = int(np.argmax(correlation))
                restored = actual[offset : offset + len(expected)]
                error = np.linalg.norm(restored - expected) / np.linalg.norm(expected)
                self.assertLess(error, 0.15)


if __name__ == "__main__":
    unittest.main()
