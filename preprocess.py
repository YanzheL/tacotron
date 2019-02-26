import tensorflow as tf
from pipelines.pipeline import Pipeline
from pipelines.handlers import CsvMetadataLoadHandler, DataLoadHandler, FeatureExtractHandler, PaddedBatchHandler, \
    TFRecordSaveHandler, PyDataLoadHandler
from hparams import hparams, SESS_CFG
from time import time
import argparse


def load_from_csv(dir):
    return [
        CsvMetadataLoadHandler(dir)
    ]


def preprocess(handlers, output_dir):
    handlers.extend(
        [
            DataLoadHandler(),
            FeatureExtractHandler(hparams.cleaners, hparams.num_mels, hparams.num_freq, method='tensorflow'),
            PaddedBatchHandler(hparams.batch_size, hparams.group_size, hparams.outputs_per_step),
            TFRecordSaveHandler(['inputs', 'lengths', 'mel_targets', 'linear_targets'])
        ]
    )
    dataset = Pipeline(handlers).process()
    with tf.Session(config=SESS_CFG) as sess:
        t1 = time()
        TFRecordSaveHandler.run(dataset, output_dir, sess)
        t2 = time()
        print('Total Time = {}s'.format(t2 - t1))


def main(args):
    handlers = globals()['load_from_{}'.format(args.src_type)](args.data_dir)
    preprocess(handlers, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='LJSpeech-1.1')
    parser.add_argument('--output_dir', default='tfrecords_tf')
    parser.add_argument('--src_type', default='csv')
    args = parser.parse_args()
    main(args)
