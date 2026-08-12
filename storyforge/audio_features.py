from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .media import TranscriptSegment


@dataclass
class AudioWindow:
    start_seconds: float
    end_seconds: float
    rms_db: float
    peak_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    speech_present: bool
    transcript: str = ""
    text_emotion: str = "中性"
    text_emotion_confidence: float = 0.0
    acoustic_state: str = "中等"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BoundaryCandidate:
    time_seconds: float
    score: float
    reasons: list[str] = field(default_factory=list)
    source: str = "audio"

    def to_dict(self) -> dict:
        return asdict(self)


class AudioFeatureTool:
    """Dependency-light PCM window analysis and persistent change detection."""

    EMOTION_WORDS = {
        "兴奋": {
            "win", "won", "great", "amazing", "excited", "success", "happy",
            "赢", "胜利", "成功", "开心", "高兴", "精彩", "厉害",
        },
        "紧张": {
            "danger", "warning", "urgent", "fight", "attack", "run", "careful",
            "危险", "警告", "紧张", "快跑", "攻击", "战斗", "小心",
        },
        "低落": {
            "sad", "sorry", "lost", "lose", "failed", "pain", "cry", "miss",
            "难过", "抱歉", "失败", "输了", "痛苦", "遗憾", "失去",
        },
        "轻松/幽默": {
            "funny", "laugh", "joke", "haha", "hilarious", "lol",
            "搞笑", "好笑", "哈哈", "笑死", "玩笑", "有趣",
        },
    }

    @classmethod
    def analyse(
        cls,
        audio: tuple[np.ndarray, int] | None,
        duration_seconds: float,
        transcript: list[TranscriptSegment],
        *,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
    ) -> list[AudioWindow]:
        if audio is None or duration_seconds <= 0:
            return []
        samples, rate = audio
        window_seconds = max(0.5, float(window_seconds))
        hop_seconds = max(0.25, min(window_seconds, float(hop_seconds)))
        windows: list[AudioWindow] = []
        start = 0.0
        while start < duration_seconds:
            end = min(duration_seconds, start + window_seconds)
            chunk = samples[max(0, int(start * rate)):min(len(samples), int(end * rate))]
            if len(chunk) < max(16, int(rate * 0.05)):
                break
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            peak = float(np.max(np.abs(chunk)))
            signs = np.signbit(chunk)
            zcr = float(np.count_nonzero(signs[1:] != signs[:-1]) / max(1, len(chunk) - 1))
            centroid = cls._spectral_centroid(chunk, rate)
            overlapping = [
                item.text for item in transcript
                if min(end, item.end_seconds) > max(start, item.start_seconds)
            ]
            text = " ".join(dict.fromkeys(overlapping)).strip()
            emotion, confidence = cls.text_emotion(text)
            windows.append(AudioWindow(
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                rms_db=round(20 * math.log10(max(rms, 1e-8)), 2),
                peak_db=round(20 * math.log10(max(peak, 1e-8)), 2),
                zero_crossing_rate=round(zcr, 5),
                spectral_centroid_hz=round(centroid, 2),
                speech_present=bool(text),
                transcript=text,
                text_emotion=emotion,
                text_emotion_confidence=confidence,
            ))
            if end >= duration_seconds:
                break
            start += hop_seconds
        cls._assign_acoustic_states(windows)
        return windows

    @staticmethod
    def _spectral_centroid(samples: np.ndarray, rate: int) -> float:
        if not len(samples):
            return 0.0
        # Limit FFT cost on long windows without changing the feature's purpose.
        stride = max(1, len(samples) // 16_384)
        values = samples[::stride]
        spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values))))
        total = float(spectrum.sum())
        if total <= 1e-12:
            return 0.0
        frequencies = np.fft.rfftfreq(len(values), d=stride / rate)
        return float(np.dot(frequencies, spectrum) / total)

    @classmethod
    def text_emotion(cls, text: str) -> tuple[str, float]:
        lower = text.lower()
        english_tokens = set(re.findall(r"[a-z']+", lower))
        matches = {
            label: sum(
                1 for word in words
                if (word in english_tokens if word.isascii() else word in lower)
            )
            for label, words in cls.EMOTION_WORDS.items()
        }
        label, count = max(matches.items(), key=lambda item: item[1], default=("中性", 0))
        if count <= 0:
            return "中性", 0.0
        return label, round(min(0.95, 0.55 + 0.12 * count), 2)

    @staticmethod
    def _assign_acoustic_states(windows: list[AudioWindow]) -> None:
        if not windows:
            return
        energies = np.asarray([item.rms_db for item in windows], dtype=np.float32)
        low = float(np.percentile(energies, 35))
        high = float(np.percentile(energies, 70))
        for item in windows:
            if item.rms_db <= low:
                item.acoustic_state = "平静/低能量"
            elif item.rms_db >= high:
                item.acoustic_state = "活跃/高能量"
            else:
                item.acoustic_state = "中等"

    @classmethod
    def change_boundaries(
        cls,
        windows: list[AudioWindow],
        *,
        threshold: float = 0.55,
        persistence_windows: int = 2,
        merge_seconds: float = 0.8,
    ) -> list[BoundaryCandidate]:
        persistence = max(1, int(persistence_windows))
        if len(windows) < persistence * 2:
            return []
        candidates: list[BoundaryCandidate] = []
        for index in range(persistence, len(windows) - persistence + 1):
            left = windows[index - persistence:index]
            right = windows[index:index + persistence]
            energy_delta = min(1.0, abs(cls._median(left, "rms_db") - cls._median(right, "rms_db")) / 12.0)
            centroid_left = cls._median(left, "spectral_centroid_hz")
            centroid_right = cls._median(right, "spectral_centroid_hz")
            centroid_delta = min(1.0, abs(math.log((centroid_right + 50) / (centroid_left + 50))) / math.log(4))
            zcr_delta = min(1.0, abs(cls._median(left, "zero_crossing_rate") - cls._median(right, "zero_crossing_rate")) / 0.12)
            left_emotion = cls._dominant_emotion(left)
            right_emotion = cls._dominant_emotion(right)
            left_state = cls._dominant_state(left)
            right_state = cls._dominant_state(right)
            emotion_changed = (
                left_emotion != right_emotion
                and (left_emotion != "中性" or right_emotion != "中性")
            )
            acoustic_state_changed = left_state != right_state
            speech_changed = all(item.speech_present for item in left) != all(
                item.speech_present for item in right
            )
            score = (
                0.30 * energy_delta
                + 0.12 * centroid_delta
                + 0.05 * zcr_delta
                + 0.50 * float(emotion_changed)
                + 0.15 * float(acoustic_state_changed)
                + 0.03 * float(speech_changed)
            )
            reasons: list[str] = []
            if emotion_changed:
                reasons.append(f"文本情绪变化:{left_emotion}→{right_emotion}")
            if energy_delta >= 0.45:
                reasons.append("声音能量持续变化")
            if acoustic_state_changed:
                reasons.append(f"声学状态变化:{left_state}→{right_state}")
            if centroid_delta >= 0.35:
                reasons.append("声音频谱持续变化")
            if speech_changed:
                reasons.append("语音/非语音状态变化")
            if score >= threshold and reasons:
                left_center = (left[-1].start_seconds + left[-1].end_seconds) / 2
                right_center = (right[0].start_seconds + right[0].end_seconds) / 2
                candidates.append(BoundaryCandidate(
                    time_seconds=round((left_center + right_center) / 2, 3),
                    score=round(min(1.0, score), 3),
                    reasons=reasons,
                    source="audio_emotion",
                ))
        return cls._merge_candidates(candidates, merge_seconds)

    @staticmethod
    def _median(windows: list[AudioWindow], name: str) -> float:
        return float(np.median([float(getattr(item, name)) for item in windows]))

    @staticmethod
    def _dominant_emotion(windows: list[AudioWindow]) -> str:
        labels = [item.text_emotion for item in windows]
        emotional = [label for label in labels if label != "中性"]
        if not emotional:
            return "中性"
        label = max(sorted(set(emotional)), key=lambda value: (emotional.count(value), value))
        # A non-neutral state must occupy a majority of the persistence window.
        return label if emotional.count(label) > len(labels) / 2 else "中性"

    @staticmethod
    def _dominant_state(windows: list[AudioWindow]) -> str:
        labels = [item.acoustic_state for item in windows]
        return max(sorted(set(labels)), key=lambda label: (labels.count(label), label))

    @staticmethod
    def _merge_candidates(
        candidates: list[BoundaryCandidate],
        merge_seconds: float,
    ) -> list[BoundaryCandidate]:
        merged: list[BoundaryCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.time_seconds):
            if merged and candidate.time_seconds - merged[-1].time_seconds <= merge_seconds:
                previous = merged[-1]
                if candidate.score > previous.score:
                    candidate.reasons = list(dict.fromkeys([*previous.reasons, *candidate.reasons]))
                    merged[-1] = candidate
                else:
                    previous.reasons = list(dict.fromkeys([*previous.reasons, *candidate.reasons]))
            else:
                merged.append(candidate)
        return merged
