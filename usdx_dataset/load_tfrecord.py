import tensorflow as tf
import note_seq

files = ["data/datasets/usdx_vocals.tfrecord"]

# Create dataset of filenames
dataset = tf.data.Dataset.from_tensor_slices(files)


def parse_fn(*args):
    pb = args[-1]  # Some readers have more than 1 arg.
    # Parse data using a feature list to convert it into an Example proto
    return tf.io.parse_single_example(pb, {
        'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
        'audio': tf.io.FixedLenFeature([], dtype=tf.string),
        # 'velocity_range': tf.io.FixedLenFeature([], dtype=tf.string),
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
    })


def reader(filename):
    # Load data from file with given name and parse the loaded dataset
    return tf.data.TFRecordDataset(filename).map(parse_fn)


dataset = dataset.interleave(reader, cycle_length=16, block_length=16)

for record in dataset:
    print("Example record:")
    print(record['id'])
    sequence_bytes = record['sequence'].numpy()
    sequence = note_seq.NoteSequence()
    sequence.MergeFromString(sequence_bytes)
    note_seq.note_sequence_to_midi_file(sequence, 'data/out/test.mid')
    tf.io.write_file('data/out/test.wav', record['audio'])
    break
