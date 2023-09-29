import codecs
import os

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
import chardet


def get_file_encoding(filepath: str) -> str:
    """
    Get the encoding of a file
    :param filepath: The path of the file
    :return: The encoding of the file
    """
    with open(filepath, 'rb') as f:
        return chardet.detect(f.read())['encoding']


def check_validity(filepath: str) -> tuple[bool, str]:
    """
    Check if a USDX song is suitable for model training.
    The following songs are currently not suitable:
    - Songs with rap notes
    - Duet songs
    :param filepath: The path of the USDX song file
    :return: True if the song is suitable for model training, otherwise false
    """
    with (codecs.open(filepath, 'r', encoding=get_file_encoding(filepath)) as song_file):
        for line in song_file:
            if line.startswith('f') or line.startswith('F'):
                return False, 'RAP_NOTES'
            if line.startswith('P1') or line.startswith('p1'):
                return False, 'DUET'
            if line.startswith('#RELATIVE:YES'):
                return False, 'RELATIVE'
        return True, ''


def to_float(value: str) -> float:
    return float(value.replace(',', '.'))


def normalize_notes(notes: list):
    lowest_pitch = 999
    for note in notes:
        if note['pitch'] < lowest_pitch:
            lowest_pitch = note['pitch']
    # Calculate the lowest possible offset
    offset_factor = int(lowest_pitch / 12.0)
    if lowest_pitch < 0:
        # If there are notes with negative pitch, move notes up one more octave so that all pitches are positive
        offset_factor -= 1
    offset = offset_factor * 12
    # Apply offset to all notes to normalize them
    for note in notes:
        note['pitch'] -= offset
    return notes


def load_song_data(filepath: str) -> dict:
    """
    Load the relevant data of a USDX song file
    :param filepath: The path of the USDX song file
    :return: The data of the song
    """
    data = {
        'bpm': 0,
        'gap': 0,
        'notes': []
    }

    with codecs.open(filepath, 'r', encoding=get_file_encoding(filepath)) as song_file:
        for line in song_file:
            if line.startswith('#BPM:'):
                data['bpm'] = to_float(line.split(':')[1])

            if line.startswith('#GAP:'):
                gap = to_float(line.split(':')[1])
                data['start_offset'] = int(second2tick(gap / 1000.0, 4, bpm2tempo(data['bpm'])))

            if not line.startswith(':') and not line.startswith('*'):
                continue

            parts = line.split()
            start = int(parts[1])
            length = int(parts[2])
            pitch = int(parts[3])

            data['notes'].append({
                'start': start,
                'length': length,
                'pitch': pitch
            })

    normalize_notes(data['notes'])
    return data


def usdx_to_midi(in_filepath: str, out_filepath: str):
    """
    Convert a USDX song file to a MIDI file
    :param in_filepath: The path of the USDX song file
    :param out_filepath: The output path of the generated MIDI file
    """
    data = load_song_data(in_filepath)

    # Create MIDI file and set initial values
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 4
    track.append(Message('program_change', program=1, time=0))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(data['bpm']), time=0))

    current_position = 0
    first_note = True
    start_offset = data['start_offset']

    for note in data['notes']:
        start = note['start']
        length = note['length']
        note = 60 + note['pitch']

        # Calculate the onset time of the note
        # The onset time is the delta time since the last note event
        if first_note:
            onset_time = start + start_offset
            first_note = False
        else:
            onset_time = start - current_position

        # Sanity check for the onset time
        if onset_time < 0:
            raise Exception(f'Invalid midi note time: {onset_time} '
                            f'(start time: {start}, current position: {current_position})')

        note_start_msg = Message('note_on', note=note, velocity=127, time=onset_time)
        # Since times in MIDI are delta times, the offset time for the note is equal to the length of the note
        note_end_msg = Message('note_off', note=note, velocity=127, time=length)
        track.append(note_start_msg)
        track.append(note_end_msg)

        current_position = start + length

    # Create parent diretories if they don't exist
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    mid.save(out_filepath)
