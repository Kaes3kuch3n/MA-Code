import os

import tensorflow as tf
import torch
import torchaudio
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from pedalboard import Pedalboard, NoiseGate, LowpassFilter, Compressor
from pedalboard.io import AudioFile
from pydub import AudioSegment, effects
from speechbrain.pretrained import SpectralMaskEnhancement, WaveformEnhancement

# Load models in advance to cache them between function calls
basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
metricgan_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="models/metricgan-plus-voicebank",
)
mtl_model = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="models/mtl-mimic-voicebank",
)


def transcribe_vocals(
        vocals_path: str,
        workdir: str,
        noise_gate_threshold: float = -18.0,
        noise_gate_attack: float = 500.0,
        noise_gate_release: float = 1500.0,
        lowpass_cutoff: float = 500.0,
        compressor_threshold: float = -6.0,
        compressor_ratio: float = 5.0,
        compressor_attack: float = 1.0,
        compressor_release: float = 100.0,
        ml_model: str = None,
        basic_pitch_onset_threshold: float = 0.5,
        basic_pitch_frame_threshold: float = 0.3,
        basic_pitch_minimum_note_length: float = 127.7,
):
    def normalize_audio(in_path: str, out_path: str):
        audio = AudioSegment.from_file(in_path)
        normalized = effects.normalize(audio)
        normalized.export(out_path, format="wav")

    def apply_pedalboard(board, in_path, out_path):
        with AudioFile(in_path) as audio:
            with AudioFile(out_path, 'w', audio.samplerate, audio.num_channels) as output:
                # Loop over file and apply pedalboard effects
                while audio.tell() < audio.frames:
                    chunk = audio.read(int(audio.samplerate))
                    effected = board.process(chunk, audio.samplerate, reset=False)
                    output.write(effected)

    def apply_metricgan(in_path, out_path):
        noisy = metricgan_model.load_audio(in_path).unsqueeze(0)
        enhanced = metricgan_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
        torchaudio.save(out_path, enhanced.cpu(), 16000)

    def apply_mtl(in_path, out_path):
        enhanced = mtl_model.enhance_file(in_path)
        torchaudio.save(out_path, enhanced.unsqueeze(0).cpu(), 16000)

    def optimize_audio(in_path: str, out_path: str):
        # Setup pedalboard
        board = Pedalboard([
            NoiseGate(threshold_db=noise_gate_threshold, attack_ms=noise_gate_attack, release_ms=noise_gate_release),
            LowpassFilter(cutoff_frequency_hz=lowpass_cutoff),
            Compressor(
                threshold_db=compressor_threshold,
                ratio=compressor_ratio,
                attack_ms=compressor_attack,
                release_ms=compressor_release,
            ),
        ])

        if ml_model:
            tmp_path = os.path.join(workdir, "vocals_tmp.wav")
            apply_pedalboard(board, in_path, tmp_path)
            if ml_model == "metricgan":
                apply_metricgan(tmp_path, out_path)
            elif ml_model == "mtl":
                apply_mtl(tmp_path, out_path)
            else:
                raise ValueError(f"Unknown ML model: {ml_model}")
        else:
            apply_pedalboard(board, in_path, out_path)

    def predict_notes(in_path: str):
        model_output, midi_data, note_events = predict(
            in_path,
            basic_pitch_model,
            minimum_frequency=80,
            maximum_frequency=1000,
            onset_threshold=basic_pitch_onset_threshold,
            frame_threshold=basic_pitch_frame_threshold,
            minimum_note_length=basic_pitch_minimum_note_length,
        )
        if len(midi_data.instruments) == 0:
            return []
        return midi_data.instruments[0].notes

    norm_vocals = os.path.join(workdir, "vocals_normalized.wav")
    normalize_audio(vocals_path, norm_vocals)
    opt_vocals = os.path.join(workdir, "vocals_optimized.wav")
    optimize_audio(norm_vocals, opt_vocals)
    return predict_notes(opt_vocals)
