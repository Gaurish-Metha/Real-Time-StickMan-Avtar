import sounddevice as sd
import numpy as np

class AudioProcessor:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.volume = 0.0
        self.stream = None

    def callback(self, indata, frames, time, status):
        """
        Audio callback function.
        """
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(indata**2))
        # Normalize roughly to 0-1 range (with some gain)
        self.volume = np.clip(rms * 5, 0, 1)

    def start(self):
        """Starts the audio stream."""
        try:
            self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.rate, blocksize=self.chunk)
            self.stream.start()
        except Exception as e:
            print(f"Warning: Could not start audio stream: {e}")

    def get_volume(self):
        """Returns the current volume level (0.0 to 1.0)."""
        return self.volume

    def stop(self):
        """Stops the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
