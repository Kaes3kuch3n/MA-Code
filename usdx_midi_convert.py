from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

mid.ticks_per_beat = 4

track.append(Message('program_change', program=1, time=0))

current_position = 0
first_note = True
start_modifier = 0

with open('data/in/Avicii - Wake Me Up.txt', 'r') as song_file:
    for line in song_file:
        if line.startswith('#BPM:'):
            bpm = int(line.split(':')[1])
            track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))

        if line.startswith('#GAP:'):
            gap = int(line.split(':')[1])
            start_offset = int(second2tick(gap / 1000.0, 4, bpm2tempo(bpm)))

        if line.startswith('- 1000'):
            start_modifier = - (16 * 4)

        if not line.startswith(':') and not line.startswith('*'):
            continue

        parts = line.split()
        start = int(parts[1]) + start_modifier
        length = int(parts[2])
        pitch = int(parts[3])

        note = 60 + pitch

        if first_note:
            note_start_msg = Message('note_on', note=note, velocity=127, time=start + start_offset)
            first_note = False
        else:
            note_start_msg = Message('note_on', note=note, velocity=127, time=start - current_position)

        note_end_msg = Message('note_off', note=note, velocity=127, time=length)
        track.append(note_start_msg)
        track.append(note_end_msg)

        current_position = start + length

mid.save('data/out/usdx-midi/wake-me-up.mid')
