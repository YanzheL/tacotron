import tensorflow as tf
from time import time
import os
import glob
from multiprocessing import cpu_count
from util import audio
from functools import partial
import scipy as sp
from text import text_to_sequence, text_to_sequence_tf


class Handler(object):
    def __init__(self):
        self.output_spec = {}

    def _set_prev(self, handler):
        self.prev = handler

    def set_output_spec(self):
        pass

    def _handle(self, dataset):
        raise NotImplementedError

    def _apply_output_spec(self, dataset):
        if 'shapes' in self.output_spec:
            shapes = tuple(
                [tf.TensorShape(i) for i in self.output_spec['shapes']]
            )
            dataset = dataset.apply(tf.contrib.data.assert_element_shape(shapes))

        return dataset

    def handle(self, dataset):
        dataset = self._handle(dataset)
        dataset = self._apply_output_spec(dataset)
        return dataset

    def exception_caught(self, exception):
        print(exception)
        pass


class CsvMetadataLoadHandler(Handler):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir

    def _handle(self, dataset):
        dataset = tf.data.experimental.CsvDataset(
            os.path.join(self.dir, 'metadata.csv'),
            [
                tf.string,
                tf.string,
            ],
            field_delim='|',
            use_quote_delim=False,
            select_cols=[0, 1]
        )
        dataset = dataset.map(
            self._path_convert,
            num_parallel_calls=cpu_count()
        )
        return dataset

    def _path_convert(self, filename, text):
        filename = tf.strings.format(os.path.join(self.dir, 'wavs', '{}.wav'), filename)
        filename = tf.regex_replace(filename, '"', '')
        return filename, text


class DataLoadHandler(Handler):
    def _handle(self, dataset):
        dataset = dataset.map(
            self._load_wav,
            num_parallel_calls=cpu_count()
        )
        return dataset

    @staticmethod
    def _load_wav(wav_path, text):
        raw_wav = tf.read_file(wav_path)
        wav = tf.audio.decode_wav(
            raw_wav,
            # desired_samples=hparams.sample_rate,
            name='audio_decode'
        )[0]
        return tf.squeeze(wav), text


class PyDataLoadHandler(Handler):
    def _handle(self, dataset):
        dataset = dataset.map(
            lambda wav_path, text: tf.py_func(
                self._load_wav,
                (wav_path, text),
                (tf.float32, tf.string)
            ),
            num_parallel_calls=cpu_count()
        )
        return dataset

    @staticmethod
    def _load_wav(wav_path, text):
        wav = audio.load_wav(wav_path)
        return wav, text


class FeatureExtractHandler(Handler):
    def __init__(self, cleaners, num_mels, num_freq, method='std'):
        super().__init__()
        self._cleaner_names = [x.strip() for x in cleaners.split(',')]
        self.num_mels = num_mels
        self.num_freq = num_freq
        self.method = method

    def set_output_spec(self):
        self.output_spec = {
            'shapes': (
                [None],
                [1],
                [None, self.num_mels],
                [None, self.num_freq],
                [1],
            )
        }

    def _handle(self, dataset):
        if self.method == 'tensorflow':
            dataset = dataset.map(
                self._extract_features_tf,
                num_parallel_calls=cpu_count()
            )
        else:
            dataset = dataset.map(
                lambda wav, text: tf.py_func(
                    self._extract_features_std,
                    (wav, text),
                    (tf.int32, tf.int32, tf.float32, tf.float32, tf.int32)
                ),
                num_parallel_calls=cpu_count()
            )
        return dataset

    def _extract_features_tf(self, wav, text):
        # Compute the linear-scale spectrogram from the wav:
        wav_pre = audio.preemphasis_tensorflow(wav)
        # wav_pre = wav
        linear_spec = audio.spectrogram_tensorflow(wav_pre)
        mel_spec = audio.melspectrogram_tensorflow(wav_pre)
        input_data = tf.py_func(
            partial(text_to_sequence_tf, cleaner_names=self._cleaner_names),
            [text],
            tf.int64
        )
        # input_data = [1, 2, 3, 4]
        input_data = tf.cast(input_data, tf.int32)
        input_length = tf.shape(input_data)[0]
        return input_data, [input_length], mel_spec, linear_spec, [tf.shape(linear_spec)[0]]

    def _extract_features_std(self, wav, text):
        wav_pre = audio.preemphasis(wav)
        linear_target = audio.spectrogram(wav_pre).astype(sp.float32)
        mel_target = audio.melspectrogram(wav_pre).astype(sp.float32)
        input_data = sp.asarray(text_to_sequence(str(text, encoding='utf8'), self._cleaner_names), dtype=sp.int32)
        input_length = sp.int32(len(input_data))
        return input_data, [input_length], mel_target.T, linear_target.T, [sp.int32(len(linear_target.T))]


class PaddedBatchHandler(Handler):
    def __init__(self, batch_size, group_size, outputs_per_step):
        super().__init__()
        self.batch_size = batch_size
        self.group_size = group_size
        self.outputs_per_step = outputs_per_step

    def set_output_spec(self):
        def _squeeze(shape):
            return [i for i in shape if i != 1]

        prev_shapes = self.prev.output_spec['shapes'] if 'shapes' in self.prev.output_spec else None
        if prev_shapes is not None:
            shapes = []
            # Output 4 elements
            for i in range(4):
                pshape = prev_shapes[i]
                cshape = [self.batch_size]
                cshape.extend(pshape)
                cshape = _squeeze(cshape)
                shapes.append(cshape)
            self.output_spec = {'shapes': tuple(shapes)}

    def _handle(self, dataset):
        dataset = dataset.prefetch(self.batch_size * self.group_size)
        dataset = dataset.padded_batch(
            self.batch_size * self.group_size,
            (
                [-1],
                1,
                [-1, -1],
                [-1, -1],
                1
            ),
        )
        dataset = dataset.map(
            self._squeeze
        )
        dataset = dataset.map(
            self._sort_by_length_tf
        )
        # dataset = dataset.map(
        #     lambda inputs, input_lengths, mel_targets, linear_targets, target_lengths: tf.py_func(
        #         self._sort_by_length,
        #         (inputs, input_lengths, mel_targets, linear_targets, target_lengths),
        #         (tf.int32, tf.int32, tf.float32, tf.float32, tf.int32)
        #     )
        # )
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.map(
            self._unpadding
        )
        dataset = dataset.padded_batch(
            self.batch_size,
            (
                [-1],
                1,
                [-1, -1],
                [-1, -1],
            ),
            drop_remainder=True
        )
        dataset = dataset.map(
            self._squeeze
        )
        dataset = dataset.map(
            self._align_batch
        )
        return dataset

    def _squeeze(self, *tensors):
        return tuple([tf.squeeze(t) for t in tensors])

    def _unpadding(self, input, input_length, mel_target, linear_target, target_length):
        return input[:input_length], [input_length], mel_target[:target_length, :], linear_target[:target_length, :]

    def _sort_by_length(self, inputs, input_lengths, mel_targets, linear_targets, target_lengths):
        # print(inputs.shape)
        # print(input_lengths.shape)
        # print(mel_targets.shape)
        # print(linear_targets.shape)
        # print(target_lengths.shape)
        indices = target_lengths.argsort()
        # print(indices)
        return inputs[indices, :], input_lengths[indices], mel_targets[indices, :, :], linear_targets[indices, :,
                                                                                       :], target_lengths[indices]

    def _sort_by_length_tf(self, inputs, input_lengths, mel_targets, linear_targets, target_lengths):
        indices = tf.argsort(target_lengths)
        return tf.gather(inputs, indices), \
               tf.gather(input_lengths, indices), \
               tf.gather(mel_targets, indices), \
               tf.gather(linear_targets, indices), \
               tf.gather(target_lengths, indices)

    def _align_batch(self, inputs, lengths, mel_targets, linear_targets):
        mel_targets = self._align_targets(mel_targets, self.outputs_per_step)
        linear_targets = self._align_targets(linear_targets, self.outputs_per_step)
        return inputs, tf.squeeze(lengths), mel_targets, linear_targets

    def _align_targets(self, targets, pack):
        length = tf.shape(targets)[1]
        packed_len = self._round_up(length + 1, pack)
        targets = tf.pad(
            targets,
            [
                [0, 0],
                [0, packed_len - length],
                [0, 0]
            ],
            "CONSTANT"
        )
        return targets

    @staticmethod
    def _round_up(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder


class TFRecordSaveHandler(Handler):
    def __init__(self, field_names):
        super().__init__()
        self.field_names = field_names

    def _handle(self, dataset):
        dataset = dataset.map(
            self._save_op,
            num_parallel_calls=cpu_count()
        )
        return dataset

    def _save_op(self, *fields):
        prev_shapes = self.prev.output_spec['shapes'] if 'shapes' in self.prev.output_spec else None
        if prev_shapes is not None:
            for pshape, field in zip(prev_shapes, fields):
                field.set_shape(pshape)
        batch_tensor_dict = {}
        for idx, field in enumerate(fields):
            name = self.field_names[idx]
            batch_tensor_dict[name] = tf.io.serialize_tensor(field)
        return batch_tensor_dict

    @staticmethod
    def run(dataset, save_dir, sess):
        one_element = dataset.make_one_shot_iterator().get_next()
        try:
            ct = 0
            while True:
                t1 = time()
                batch_dict = sess.run(one_element)
                features_dict = {}
                for k, v in batch_dict.items():
                    features_dict[k] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[v])
                    )
                example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                os.makedirs(save_dir, exist_ok=True)
                with tf.io.TFRecordWriter(
                        os.path.join(save_dir, 'batch_{}.tfrecord'.format(ct)),
                        options=tf.io.TFRecordOptions(
                            compression_type=tf.io.TFRecordCompressionType.ZLIB,
                            compression_level=9,
                        )
                ) as writer:
                    writer.write(example.SerializeToString())
                t2 = time()
                ct += 1
                print("Saved {} batches, time = {}s".format(ct, t2 - t1))
        except tf.errors.OutOfRangeError:
            print("Finished!")


class TFRecordLoadHandler(Handler):

    def __init__(self, dir, field_spec):
        super().__init__()
        self.dir = dir
        self.field_spec = field_spec
        self.feature_description = {}
        for name in field_spec.keys():
            self.feature_description[name] = tf.FixedLenFeature([], dtype=tf.string)

    def _handle(self, dataset):
        files = list(glob.iglob(os.path.join(self.dir, '**/*.tfrecord'), recursive=True))
        dataset = tf.data.TFRecordDataset(
            files,
            compression_type='ZLIB'
        )
        dataset = dataset.map(
            self._parse_example
        )
        dataset = dataset.map(
            self._parse_tensor
        )
        return dataset

    def _parse_example(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, self.feature_description)

    def _parse_tensor(self, parsed_dict):
        ret = []
        for name, spec in self.field_spec.items():
            tensor = tf.io.parse_tensor(
                parsed_dict[name],
                spec['dtype']
            )
            tensor.set_shape(spec['shape'])
            ret.append(
                tensor
            )
        return tuple(ret)


class StandardBatchHandler(Handler):
    def _handle(self, dataset):
        dataset = dataset.map(
            self._pack
        )
        dataset = dataset.shuffle(10)
        dataset = dataset.repeat()
        return dataset

    @staticmethod
    def _pack(inputs, lengths, mel_targets, linear_targets):
        return (
            {
                'inputs': inputs,
                'lengths': lengths
            },
            {
                'mel_targets': mel_targets,
                'linear_targets': linear_targets
            }
        )


from pipelines.pipeline import Pipeline
from hparams import hparams


def debug():
    # tf.enable_eager_execution()
    handlers = [
        CsvMetadataLoadHandler('../LJSpeech-1.1'),
        DataLoadHandler(),
        FeatureExtractHandler(hparams.cleaners, hparams.num_mels, hparams.num_freq, method='tensorflow'),
        PaddedBatchHandler(hparams.batch_size, 32, hparams.outputs_per_step),
        # TFRecordLoadHandler('../tfrecords_std', {
        #     'inputs': {
        #         'dtype': tf.int32,
        #         'shape': [hparams.batch_size, None]
        #     },
        #     'lengths': {
        #         'dtype': tf.int32,
        #         'shape': [hparams.batch_size]
        #     },
        #     'mel_targets': {
        #         'dtype': tf.float32,
        #         'shape': [hparams.batch_size, None, hparams.num_mels]
        #     },
        #     'linear_targets': {
        #         'dtype': tf.float32,
        #         'shape': [hparams.batch_size, None, hparams.num_freq]
        #     }
        # }),
        # StandardBatchHandler()
    ]
    pipe = Pipeline(handlers)
    dataset = pipe.process()

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        ct = 0
        try:
            t1 = time()
            while True:
                res = sess.run(one_element)
                ct += 1
                if ct % 10 == 0:
                    t2 = time()
                    print('Time = {}'.format(t2 - t1))
                    t1 = t2
                    break
                # print(type(res))
                # print((len(res[0]), res[1].shape, res[2].shape))
        except tf.errors.OutOfRangeError:
            print("end!")


if __name__ == '__main__':
    debug()
