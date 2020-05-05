"""Train the model"""

import argparse
import os

import tensorflow as tf

from input_fn import serving_input_fn
from model_fn import model_fn
from utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--export_dir', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # python export.py --model_dir ./save/ --export_dir ./serving/
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True

    # Load the parameters from json file
    params = Params('./params.json')
    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=50,
                                    save_checkpoints_steps=100,
                                    session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # [truong] Export model to serving
    estimator.export_savedmodel(args.export_dir, serving_input_receiver_fn=serving_input_fn)

