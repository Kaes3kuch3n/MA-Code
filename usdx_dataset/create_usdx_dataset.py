import os
from typing import Literal

import tensorflow as tf
import note_seq
from sklearn.model_selection import train_test_split

input_dir = 'data/prepared/'
dataset_dir = 'data/datasets/usdx_vocals/'


def load_binary_file(file_path):
    with open(file_path, "rb") as file:
        file_bytes = file.read()
    return file_bytes


def serialize_example(id, audio, sequence):
    features = {
        'audio': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio])),
        'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sequence])),
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def write_song_to_tfrecord(song: os.DirEntry, writer: tf.io.TFRecordWriter):
    wav_file = f'{song.name}.wav'
    midi_file = f'{song.name}.mid'
    entry_id = f"{wav_file}:{midi_file}".encode('utf-8')
    audio = load_binary_file(os.path.join(song.path, wav_file))
    sequence = note_seq.midi_file_to_sequence_proto(os.path.join(song.path, midi_file)).SerializeToString()
    example = serialize_example(entry_id, audio, sequence)
    writer.write(example)


def generate_dataset(songs: [str], dataset_type: Literal['train', 'test']):
    os.makedirs(os.path.join(dataset_dir, dataset_type), exist_ok=True)

    shard_size = 3
    num_shards = len(songs) // shard_size + 1
    for shard_index in range(num_shards):
        shard_file_name = f'{dataset_type}.tfrecord-{shard_index:05d}-of-{num_shards:05d}'
        shard_file_path = os.path.join(dataset_dir, dataset_type, shard_file_name)
        with (tf.io.TFRecordWriter(shard_file_path) as train_writer):
            start_idx = shard_index * len(songs) // num_shards
            end_idx = (shard_index + 1) * len(songs) // num_shards

            for song in songs[start_idx:end_idx]:
                write_song_to_tfrecord(song, train_writer)


def generate_datasets():
    os.makedirs(dataset_dir, exist_ok=True)

    songs = [f for f in os.scandir(input_dir) if f.is_dir()]
    train_songs, test_songs = train_test_split(songs, test_size=0.25, random_state=42)
    print(f'Generating datasets with {len(train_songs)} training songs and {len(test_songs)} test songs')

    generate_dataset(train_songs, 'train')
    generate_dataset(test_songs, 'test')


if __name__ == '__main__':
    generate_datasets()
