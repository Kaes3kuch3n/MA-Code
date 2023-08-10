import mir_eval.transcription
import numpy as np


# Prepare MIDI for use with mir_eval
class EvalData:
    def __init__(self):
        self.pitches = []
        self.intervals = []

    def __repr__(self) -> str:
        return "%s(pitches=%r, intervals=%r" % (self.__class__.__name__, self.pitches, self.intervals)


def prepare_eval_data(midi_notes):
    """
    Prepare MIDI data to be usable as input for mir_eval
    :param midi_notes: The PrettyMIDI notes array
    :return: An EvalData object containing a pitches array and an intervals array
    """
    eval_data = EvalData()

    # Since the resulting MIDI data contains a pitch value and the USDX files do not always use the correct octave,
    # we have to normalize it before we can evaluate it using mir_eval. To do this, we calculate the modulo of the
    # pitch and 12, which is the range of an octave. After that we add 60 to normalize the pitch value to the
    # middle C octave.

    for note in midi_notes:
        eval_data.pitches.append((note.pitch % 12) + 60)
        eval_data.intervals.append((note.start, note.end))

    eval_data.pitches = np.array(eval_data.pitches)
    eval_data.intervals = np.array(eval_data.intervals)
    return eval_data


def evaluate(reference_data: EvalData, estimated_data: EvalData):
    return mir_eval.transcription.evaluate(
        reference_data.intervals,
        reference_data.pitches,
        estimated_data.intervals,
        estimated_data.pitches
    )
