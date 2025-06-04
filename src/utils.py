
import logging
import re

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def is_silent(data):
    if np.abs(data).max() < 3e-3:
        return True
    else:
        return False


def sentence_end(txt):
    for c in [".", "。", "!", "?", "!", "?"]:
        if c in txt:
            if c == ".":  # check not number before it like 1.
                idx = txt.find(c)
                if idx > 0:
                    if txt[idx - 1].isdigit():
                        continue
            return c
    return ""


class NumberToTextConverter:
    r"""
    A helper class to ensure text-to-speech (TTS) systems read numeric digits
    in the desired language (Chinese, English, Vietnamese) digit-by-digit. It forcibly
    replaces all numeric substrings in text with their language-specific
    textual representations, thereby reducing the likelihood of TTS mistakes
    on numbers.

    Note: Khaanh only use this in streaming mode.
    
    Attributes:
        num_to_chinese (dict):
            Mapping from digit (str) to its Chinese textual form (str).
        num_to_english (dict):
            Mapping from digit (str) to its English textual form (str).
    
    Example:
        >>> converter = NumberToTextConverter()
        >>> converter.replace_numbers_with_text("Tôi có 2 quả táo", language="vietnamese")
        'Tôi có hai quả táo'
        >>> converter.replace_numbers_with_text("I have 23 books", language="english")
        'I have two three books'
        >>> converter.replace_numbers_with_text("我有2個蘋果", language="chinese")
        '我有兩個蘋果'
    """

    def __init__(self):
        self.num_to_chinese = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }
        self.num_to_english = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        self.num_to_vietnamese = {
            "0": "không",
            "1": "một",
            "2": "hai",
            "3": "ba",
            "4": "bốn",
            "5": "năm",
            "6": "sáu",
            "7": "bảy",
            "8": "tám",
            "9": "chín",
        }

    def number_to_chinese_digit_by_digit(self, num_str):
        result = ""
        for char in num_str:
            if char in self.num_to_chinese:
                result += self.num_to_chinese[char]
        return result

    def number_to_english_digit_by_digit(self, num_str):
        result = []
        for char in num_str:
            if char in self.num_to_english:
                result.append(self.num_to_english[char])
        return " ".join(result)

    def detect_language(self, text):
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_count = len(re.findall(r"[a-zA-Z]", text))
        return "chinese" if chinese_count >= english_count else "english"

    def replace_numbers_with_text(self, text, language=None):
        if language is None:
            language = self.detect_language(text)
        numbers = re.findall(r"\d+", text)

        for num in numbers:
            if language == "chinese":
                replacement = self.number_to_chinese_digit_by_digit(num)
            else:
                replacement = self.number_to_english_digit_by_digit(num)
            text = text.replace(num, replacement, 1)

        return text


class VoiceChecker:
    r"""
    A simple utility class to detect silence or low variation in consecutive audio chunks by comparing
    the mel-spectrogram distances. It keeps track of consecutive zero-distance and low-distance chunks
    to decide if the audio is considered "bad" (e.g., overly silent or not changing enough).
    Attributes:
        previous_mel (`np.ndarray` or `None`):
            Holds the previously observed mel-spectrogram in decibel scale. Used to compute
            the next distance; reset via :meth:`reset`.
        consecutive_zeros (`int`):
            The number of consecutive chunks that were detected as silent (distance = 0).
        consecutive_low_distance (`int`):
            The number of consecutive chunks whose distance was below the threshold.
    Example:
        >>> checker = VoiceChecker()
        >>> # Suppose we have audio_wav (list or np.ndarray) and mel_spec (np.ndarray)
        >>> # We split them into chunks and call checker.is_bad(...)
        >>> is_audio_bad = checker.is_bad(audio_wav, mel_spec, chunk_size=2560, thresh=100.0)
        >>> if is_audio_bad:
        ...     print("Audio deemed bad!")
        >>> # Reset states if needed
        >>> checker.reset()
    """

    def __init__(self):
        self.previous_mel = None
        self.consecutive_zeros = 0
        self.consecutive_low_distance = 0

    def compute_distance(self, audio_chunk, mel_spec):
        if is_silent(audio_chunk):
            return 0.0 # Kiểm tra xem đó có phải là đoạn trống không

        mel_db = librosa.power_to_db(mel_spec)
        if self.previous_mel is None:
            self.previous_mel = mel_db
            return -1.0

        distance = np.linalg.norm(np.mean(mel_db, axis=1) - np.mean(self.previous_mel, axis=1))
        self.previous_mel = mel_db
        return distance

    def is_bad(self, audio_wav, mel_spec, chunk_size=2560, thresh=100.0):
        num_chunks = len(audio_wav) // chunk_size
        mel_chunk_size = mel_spec.shape[-1] // num_chunks
        for i in range(num_chunks):
            audio_chunk = audio_wav[i * chunk_size : (i + 1) * chunk_size]
            mel_spec_chunk = mel_spec[:, i * mel_chunk_size : (i + 1) * mel_chunk_size]

            distance = self.compute_distance(audio_chunk, mel_spec_chunk)
            logger.warning(
                f"mel dist: {distance:.1f}, zero: {self.consecutive_zeros}, low: {self.consecutive_low_distance}"
            )
            if distance == 0:
                self.consecutive_low_distance = 0  # reset
                self.consecutive_zeros += 1
                if self.consecutive_zeros >= 12:
                    logger.warning("VoiceChecker detected 1.2 s silent. Marking as failed.")
                    return True
            elif distance < thresh:
                self.consecutive_zeros = 0
                self.consecutive_low_distance += 1
                if self.consecutive_low_distance >= 5:
                    logger.warning("VoiceChecker detected 5 consecutive low distance chunks. Marking as failed.")
                    return True
            else:
                self.consecutive_low_distance = 0
                self.consecutive_zeros = 0

        return False

    def reset(self):
        self.previous_mel = None
        self.consecutive_zeros = 0
        self.consecutive_low_distance = 0