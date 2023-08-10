from typing import Optional

import demucs.separate
from basic_pitch.inference import predict_and_save, predict


def split(in_path: str, out_path: str):
    demucs.separate.main(["--two-stems", "vocals", "-n", "htdemucs_ft", "-o", out_path, in_path])


def to_midi(in_path: str, out_path: Optional[str], onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=58):
    """
    Convert audio to midi. Uses a frequency range from 80 Hz to 1kHz.
    :param in_path:
    :param out_path:
    :param onset_threshold:
    :param frame_threshold:
    :param minimum_note_length: Min note length in milliseconds
    """
    if out_path is not None:
        predict_and_save(
            [in_path],
            out_path,
            save_midi=True,
            sonify_midi=True,
            save_model_outputs=False,
            save_notes=False,
            minimum_frequency=80,
            maximum_frequency=1000,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length
        )
    else:
        return predict(
            in_path,
            minimum_frequency=80,
            maximum_frequency=1000,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length
        )
