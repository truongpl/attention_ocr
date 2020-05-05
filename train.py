import argparse
import os

import tensorflow as tf

from input_fn import train_input_fn, val_input_fn
from model_fn import model_fn
from utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # python train.py --model_dir ./save/
    params = Params('./params.json')

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the model
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=50,
                                    save_checkpoints_steps=10000)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    # # Evaluate the model on the test set
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(params))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: val_input_fn(params), throttle_secs=100)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # estimator.train(input_fn=lambda: train_input_fn(params))