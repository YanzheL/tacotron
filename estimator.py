import tensorflow as tf
from models.tacotron import Tacotron
from tensorflow.estimator import CheckpointSaverHook, SummarySaverHook, LoggingTensorHook, RunConfig
from pipelines.pipeline import Pipeline
from pipelines.handlers import TFRecordLoadHandler, StandardBatchHandler
from hparams import hparams, SESS_CFG
from text import text_to_sequence
import scipy as sp
from util.audio import inv_spectrogram, save_wav, inv_spectrogram_tensorflow, inv_preemphasis, find_endpoint
from argparse import ArgumentParser
import os

tf.logging.set_verbosity(tf.logging.INFO)


def train_input_fn(src_dir):
    handlers = [
        # CsvMetadataLoadHandler('/home/trinity/PycharmProjects/tacotron_ng/LJSpeech-1.1'),
        # DataLoadHandler(),
        # FeatureExtractHandler(hparams.cleaners, hparams.num_mels, hparams.num_freq),
        # PaddedBatchHandler(hparams.batch_size, 2, hparams.outputs_per_step),
        TFRecordLoadHandler(src_dir, {
            'inputs': {
                'dtype': tf.int32,
                'shape': [hparams.batch_size, None]
            },
            'lengths': {
                'dtype': tf.int32,
                'shape': [hparams.batch_size]
            },
            'mel_targets': {
                'dtype': tf.float32,
                'shape': [hparams.batch_size, None, hparams.num_mels]
            },
            'linear_targets': {
                'dtype': tf.float32,
                'shape': [hparams.batch_size, None, hparams.num_freq]
            }
        }),
        StandardBatchHandler()
    ]
    pipe = Pipeline(handlers)
    dataset = pipe.process()
    return dataset


def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None, config=None):
    inputs = features['inputs']
    lengths = features['lengths']
    mel_targets = None
    linear_targets = None
    train_hooks = []
    global_step = tf.train.get_global_step()
    if mode == tf.estimator.ModeKeys.TRAIN:
        mel_targets = labels['mel_targets']
        linear_targets = labels['linear_targets']
    with tf.variable_scope('model'):
        model = Tacotron(params)
        model.initialize(inputs, lengths, mel_targets, linear_targets)
        if mode == tf.estimator.ModeKeys.TRAIN:
            model.add_loss()
            model.add_optimizer(global_step)
            # train_hooks.extend([
            #     LoggingTensorHook(
            #         [global_step, model.loss, tf.shape(model.linear_outputs)],
            #         every_n_secs=60,
            #     )
            # ])
        outputs = tf.map_fn(
            inv_spectrogram_tensorflow,
            model.linear_outputs
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope('stats') as scope:
            tf.summary.histogram('linear_outputs', model.linear_outputs)
            tf.summary.histogram('linear_targets', model.linear_targets)
            tf.summary.histogram('mel_outputs', model.mel_outputs)
            tf.summary.histogram('mel_targets', model.mel_targets)
            tf.summary.scalar('loss_mel', model.mel_loss)
            tf.summary.scalar('loss_linear', model.linear_loss)
            tf.summary.scalar('learning_rate', model.learning_rate)
            tf.summary.scalar('loss', model.loss)
            gradient_norms = [tf.norm(grad) for grad in model.gradients]
            tf.summary.histogram('gradient_norm', gradient_norms)
            tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
            tf.summary.audio(
                'outputs',
                outputs,
                hparams.sample_rate,
                max_outputs=1
            )
            tf.summary.merge_all()

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=outputs,
        loss=getattr(model, 'loss', None),
        train_op=getattr(model, 'optimize', None),
        eval_metric_ops=None,
        export_outputs=None,
        training_chief_hooks=None,
        training_hooks=train_hooks,
        scaffold=None,
        evaluation_hooks=None,
        prediction_hooks=None
    )


def predict_input_fn(texts):
    def make_dataset(features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)
        # Return the dataset.
        return dataset

    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seqs = [text_to_sequence(text, cleaner_names) for text in texts]
    lengths = [len(seq) for seq in seqs]
    features = {
        'inputs': tf.convert_to_tensor(seqs, dtype=tf.int32),
        'lengths': tf.convert_to_tensor(lengths, dtype=tf.int32)
    }
    return make_dataset(features, None, len(texts))


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=hparams,
        config=RunConfig(
            save_summary_steps=args.summary_interval,
            save_checkpoints_steps=args.checkpoint_interval,
            session_config=SESS_CFG,
            # log_step_count_steps=100,
            keep_checkpoint_max=2
        )
    )
    if args.mode == 'train':
        os.makedirs(args.data_dir, exist_ok=True)
        estimator.train(
            input_fn=lambda: train_input_fn(args.data_dir)
        )
    elif args.mode == 'predict':
        assert len(args.texts), "No text to predict"
        results = estimator.predict(
            input_fn=lambda: predict_input_fn(args.texts)
        )
        for idx, wav in enumerate(results):
            wav = inv_preemphasis(wav)
            wav = wav[:find_endpoint(wav)]
            # sp.save('wav_{}.npy'.format(idx), wav, allow_pickle=False)
            save_wav(wav, 'output_{}.wav'.format(idx))
            # break
    elif args.mode == 'export':
        os.makedirs(args.export_dir, exist_ok=True)
        estimator.export_saved_model(
            args.export_dir,
            tf.estimator.export.build_raw_serving_input_receiver_fn(
                {
                    'inputs': tf.placeholder(dtype=tf.int32, shape=(None, None), name='inputs'),
                    'lengths': tf.placeholder(dtype=tf.int32, shape=(None,), name='lengths'),
                },
                default_batch_size=None
            ),
            # assets_extra=None,
            # as_text=False,
            # checkpoint_path=None,
            # experimental_mode=ModeKeys.PREDICT
        )
    else:
        raise KeyError('Unknown Mode <{}>'.format(args.mode))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='tfrecords_tf')
    parser.add_argument('--export_dir', default='export')
    parser.add_argument('--model_dir', default='logdir/20190227')
    parser.add_argument('--checkpoint_interval', '-c', type=int, default=100)
    parser.add_argument('--summary_interval', '-s', type=int, default=10)
    parser.add_argument('--mode', '-m', required=True, choices=['train', 'predict', 'export'])
    parser.add_argument('texts', nargs='*')
    args = parser.parse_args()
    # print(args)
    main(args)
