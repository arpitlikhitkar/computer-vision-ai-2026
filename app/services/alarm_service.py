"""
Laptop Audio Alarm Service (Phase 6.9 Sound Alert & Stop/Mute Controls)

Plays loud security alarm sound on laptop speaker when an UNKNOWN person is detected.
Includes Stop Alarm, Mute Alarm, and Sound Enabled/Disabled Toggles.
"""

import sys
import time
import threading
import wave
import math
import struct
import tempfile
import os

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import QApplication
from app.config.settings import config


def generate_security_alarm_wav():
    """Generates a loud two-tone security alarm siren WAV file in temp folder."""
    temp_wav_path = os.path.join(tempfile.gettempdir(), "household_unknown_alarm.wav")
    if os.path.exists(temp_wav_path):
        return temp_wav_path

    try:
        sample_rate = 22050
        duration = 1.0  # 1 second siren
        num_samples = int(sample_rate * duration)

        with wave.open(temp_wav_path, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            raw_bytes = bytearray()
            for i in range(num_samples):
                t = i / sample_rate
                # Siren alternating frequency between 880Hz and 1200Hz
                freq = 880 if (int(t * 4) % 2 == 0) else 1200
                val = int(16000 * math.sin(2 * math.pi * freq * t))
                raw_bytes.extend(struct.pack("<h", val))

            wav_file.writeframes(raw_bytes)
        return temp_wav_path
    except Exception as e:
        print(f"[ALARM] Error generating alarm WAV: {e}")
        return None


class AlarmSoundWorker(QThread):
    """Background QThread worker to play laptop speaker alarm without freezing UI."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wav_path = generate_security_alarm_wav()
        self.stopped = False

    def run(self):
        if self.stopped:
            return
        try:
            if HAS_WINSOUND and self.wav_path and os.path.exists(self.wav_path):
                # Play alarm siren WAV on Windows
                winsound.PlaySound(self.wav_path, winsound.SND_FILENAME)
            elif HAS_WINSOUND:
                # Beep at 1200Hz for 600ms
                winsound.Beep(1200, 600)
            else:
                QApplication.beep()
        except Exception as e:
            print(f"[ALARM] Speaker sound error: {e}")
            try:
                QApplication.beep()
            except Exception:
                pass

    def stop(self):
        self.stopped = True
        if HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass


class AlarmService:
    """
    Non-blocking Laptop Speaker Alarm Manager with Cooldown and Stop/Mute Controls.
    """
    def __init__(self, cooldown_seconds=5.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alarm_time = 0.0
        self.active_workers = []

    def trigger_unknown_alarm(self):
        """
        Triggers laptop audio alarm if enabled and cooldown elapsed.
        """
        if not config.enable_audio_alarm:
            return

        current_time = time.time()
        if (current_time - self.last_alarm_time) < self.cooldown_seconds:
            return

        self.last_alarm_time = current_time
        print("[ALARM SERVICE] UNKNOWN PERSON DETECTED! Playing laptop security alarm siren...")

        worker = AlarmSoundWorker()
        worker.start()
        self.active_workers.append(worker)

        # Cleanup finished workers
        for w in list(self.active_workers):
            if not w.isRunning():
                self.active_workers.remove(w)

    def stop_alarm(self):
        """
        Immediately stops any currently playing audio alarm.
        """
        print("[ALARM SERVICE] Stopping all active alarm sirens...")
        if HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        for w in self.active_workers:
            w.stop()
            w.quit()
        self.active_workers.clear()

    def toggle_mute(self) -> bool:
        """
        Toggles alarm sound on/off. Returns new enabled status.
        """
        config.enable_audio_alarm = not config.enable_audio_alarm
        config.save_to_json()
        if not config.enable_audio_alarm:
            self.stop_alarm()
        print(f"[ALARM SERVICE] Alarm sound active: {config.enable_audio_alarm}")
        return config.enable_audio_alarm


# Global singleton instance
alarm_service = AlarmService(cooldown_seconds=5.0)
