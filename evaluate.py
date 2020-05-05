"""Train the model"""

import argparse
import os

import tensorflow as tf

from input_fn import predict_input_fn, val_input_fn
from model_fn import model_fn
from utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--val_file', type=str)
args = parser.parse_args()

label_map = {'cor_start':0, 'cor_end':1, 'O':2}
unused_charset =  ["=","!","*","_",",",">","<",'"',"?","+","%","@","|","{","}","^","~",";"]

if __name__ == '__main__':
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

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.predict(lambda: predict_input_fn(args.val_file, params, label_map, unused_charset))
    # res = estimator.evaluate(lambda: val_input_fn(args.val_file, params, label_map, unused_charset))
    print(res)
    for preds in res:
        print('XXXXX ', preds)
