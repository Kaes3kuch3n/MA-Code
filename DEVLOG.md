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