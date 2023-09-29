# Dev Log

## Further Ideas
- Consider note probabilities (if possible) for deciding between notes that occur simultaneously
  - Probably not possible: The probability doesn't seem to be included in the model output
- Combine contiguous notes (if they have the same pitch and are very, very close to each other)
  - Might lead to unwanted/too many note merges
- Lower min bound for onset and frame threshold optimization

## Implementation

### Implementing USDX to MIDI converter
- Figuring out MIDI timing: Tempo and beat resolution
  - Ticks per beat
  - Delta time
- Writing MIDI with mido
- Problems
  - Rap notes can't be used because they may not be in the right octave
    - Solution: Skip songs with rap notes
  - Pitch may either be zero-based (which could lead to problems in the learning process) or
    based on the actual notes of the vocals, but MIDI can only contain pitch values between 0 and 127
    - Solution: Convert pitches to zero-based variant and offset them by 60 (to use C4 "middle C" as base note)

### Using scikit-optimize for optimizing basic_pitch
- Converting midi data to mir_eval format
- Writing mir_eval transcription evaluation loop
- Used gp_minimize (Bayesian optimization) to optimize params:
  - minimum note length
  - onset threshold
  - frame threshold
- Set frequency range to 80Hz-1kHz based on Blythe: Attention, Balance and Coordination

### Optimizing audio and improving evaluation
- Applied noise gate to vocals file to get rid of some instrumental artefacts bleeding through
- Evaluation improvements: Fixed pitch comparison, fixed BPM, checked midi alignments
  - TODO: General BPM fix (Current fix is hardcoded; find sensible solution)
- Ran gp_minimize again using improved audio pipeline and improved evaluation algorithm
  - Results improved from F1 = 0.06 or so to F1 = 0.1084 :)
  - TODO: Onset and Frame Thresholds might need some more improvement, maybe lower min bound

### Relevant features used in MT3 datasets
- MT3 datasets are tensorflow datasets
- Features:
  - "audio": wav file bytes
  - "sequence": note_seq sequence in serialized string format
  - "id": wav filename + colon + midi filename
- Finding out how: Dissecting the GuitarSet from MT3
  - audio
    - "audio" features start with a string containing `RIFF` and `WAVEfmt`
    - Probably wav file, try saving bytes as wav
    - Listen to file -> Sounds like a song!
  - sequence
    - No useful header, features start with MIDI filename
    - Maybe MIDI, try saving as mid file
    - File doesn't work in GarageBand
    - Paper mentions "MIDI-like tokens", so maybe some custom format
    - Look at how model output is converted back to MIDI
      - `note_seq` library is used -> no PyPI description, but created by the guys from the MT3 project
    - Tried converting feature to `note_seq` `NoteSequence` and saving that as MIDI file
    - Still doesn't work
    - Tried it the other way around: Loading MIDI file and converting it to `NoteSequence`
    - Discovered that `NoteSequence` objects have a `SerializeToString` method
    - No `Deserialize`, but found counterpart method `MergeFromString`
    - Converted feature to `NoteSequence` using `MergeFromString` and saved resulting sequence as MIDI file
    - MIDI file works! And sounds exactly like melody from corresponding wav file, so seems to be correct

### Add dataset to MT3
- Instrument label for vocals set
- MIDI does not have "vocals" as isntrument, use one that's most likely unused -> Bird tweet (124, cite General MIDI)