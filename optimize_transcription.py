import os.path
import random
import shutil

import pretty_midi
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from evaluate_midi import prepare_eval_data, evaluate, EvalData

# Monkey patch numpy to avoid skopt error
# (see https://github.com/scikit-optimize/scikit-optimize/issues/1138 for details)
import numpy

from transcription import transcribe_vocals

numpy.int = int

songs_path = 'data/test'
iterations = 100
batch_size = 30

progress = {
    'step': 0,
}


def optimize_transcription(songs: list):
    def evaluate_transcription(vocals_path: str, ref_notes: EvalData, params):
        tmp_dir = os.path.join(os.getcwd(), "transcription_tmp")
        os.mkdir(tmp_dir)
        est_notes = transcribe_vocals(
            vocals_path, tmp_dir,
            params[0], params[1], params[2], params[3], params[4], params[5],
            params[6], params[7], params[8], params[9], params[10], params[11],
        )
        shutil.rmtree(tmp_dir)
        if len(est_notes) == 0:
            # optimization is trying to minimize the result
            # return max value to indicate invalid result
            return 1
        scores = evaluate(ref_notes, prepare_eval_data(est_notes))
        return 1.0 - scores['F-measure']

    def load_reference_notes(notes_path: str):
        midi = pretty_midi.PrettyMIDI(notes_path)
        original_midi = midi.instruments[0].notes
        return prepare_eval_data(original_midi)

    def evaluate_transcriptions(params):
        progress['step'] += 1
        print(f"Step {progress['step']} of {iterations}")
        results = []
        for song in random.sample(songs, batch_size):
            ref_notes = load_reference_notes(song["notes"])
            result = evaluate_transcription(song["vocals"], ref_notes, params)
            results.append(result)
        return sum(results) / len(results)

    # Parameter ranges for prediction options
    param_ranges = [
        # Noise gate
        Real(-32.0, -6.0, name="noise_gate_threshold"),
        Real(70.0, 1000.0, name="noise_gate_attack"),  # 1/16 - 1/2 note
        Real(500.0, 2000.0, name="noise_gate_release"),  # 1/4 - 1 note
        # Lowpass filter
        Real(80.0, 1500.0, name="lowpass_cutoff"),  # Lowest vocal frequency - include some overtones
        # Compressor
        Real(-18.0, -3.0, name="compressor_threshold"),
        Real(2.0, 10.0, name="compressor_ratio"),
        Real(1.0, 80.0, name="compressor_attack"),  # 1 ms - little more than 1/16 note
        Real(80.0, 500.0, name="compressor_release"),  # little more than 1/16 note - 1/4 note
        # ML model
        Categorical([None, "metricgan", "mtl"], name="ml_model"),
        # Basic Pitch
        Real(0.2, 0.8, name="basic_pitch_onset_threshold"),
        Real(0.2, 0.8, name="basic_pitch_frame_threshold"),
        Integer(80, 250, name="basic_pitch_minimum_note_length")
    ]

    results = gp_minimize(evaluate_transcriptions, param_ranges, n_calls=iterations)
    print(results)


if __name__ == "__main__":
    full_songs_path = os.path.join(os.getcwd(), songs_path)
    songs = [{'vocals': os.path.join(full_songs_path, song, 'vocals.wav'),
              'notes': os.path.join(full_songs_path, song, f'{song}.mid')} for song
             in os.listdir(full_songs_path)]
    print(f'optimizing over {len(songs)} songs')
    optimize_transcription(songs)
