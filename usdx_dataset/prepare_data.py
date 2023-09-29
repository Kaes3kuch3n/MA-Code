import os

import pydub

from usdx_dataset.usdx_tools import check_validity, usdx_to_midi

input_dir = 'data/usdx/'


def get_files(files: list[str]) -> tuple[None, None] | tuple[str, str]:
    """
    # Return the first .mp3 and the first .txt file found in the list
    # If one of the two is not found, return None
    :param files: The list of files to search in
    :return: The first .mp3 and .txt file found in the list. If one of the two is not found, returns None.
    """
    found_mp3 = None
    found_txt = None
    for file in files:
        if file.endswith('.mp3'):
            found_mp3 = file
        elif file.endswith('.txt'):
            found_txt = file
    if found_mp3 is None or found_txt is None:
        return None, None
    return found_mp3, found_txt


def prepare_audio(mp3_path: str) -> str:
    # Convert the MP3 file to a WAV file since MT3 expects .wav files
    mp3_filename = os.path.basename(mp3_path)[:-4]
    wav_path = f'data/prepared/{mp3_filename}/{mp3_filename}.wav'
    pydub.AudioSegment.from_mp3(mp3_path).export(wav_path, format='wav')
    return wav_path


def prepare_midi(usdx_path: str) -> tuple[bool, str]:
    """
    Convert the USDX .txt file to a MIDI file
    :param usdx_path: The path to the .txt file
    :return: A tuple containing a boolean indicating whether the song is suitable for training and a string.
    If the song is suitable for training, the string contains the path to the MIDI file, otherwise it contains the
    reason why the song is not suitable for training.
    """
    # Convert the usdx .txt file to a MIDI file
    usdx_filename = os.path.basename(usdx_path)[:-4]
    midi_path = f'data/prepared/{usdx_filename}/{usdx_filename}.mid'
    valid, reason = check_validity(usdx_path)
    if not valid:
        # Skip songs that are not suitable for training
        return False, reason
    usdx_to_midi(usdx_path, midi_path)
    return True, midi_path


def prepare_data(input_dir: str):
    song_rejection_reasons = {}
    # Walk through all the subdirectories in the input_dir
    for root, _, files in os.walk(input_dir, onerror=print):
        # Skip top directory
        if root == input_dir:
            continue
        print(f'Preparing {root}...')
        mp3_file, usdx_file = get_files(files)
        if mp3_file is None:
            print(f'MP3/USDX file missing in {root}, skipping')
            continue

        # Directory contains necessary files, prepare them for the dataset
        usdx_path = os.path.join(root, usdx_file)
        success, other = prepare_midi(usdx_path)
        if not success:
            # Song is not suitable for training, skip it
            print(f'Song {root} is not suitable for training ({other}), skipping')
            song_rejection_reasons[other] = song_rejection_reasons.get(other, 0) + 1
            continue

        mp3_path = os.path.join(root, mp3_file)
        prepare_audio(mp3_path)
        print(f'Prepared {root}')

    print('Skipped songs:')
    for reason, count in song_rejection_reasons.items():
        print(f'{reason}: {count}')


if __name__ == '__main__':
    prepare_data(input_dir)
