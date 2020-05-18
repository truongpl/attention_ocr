import argparse
import os
import glob

import tensorflow as tf

from input_fn import train_input_fn, val_input_fn
from model_fn import model_fn
from utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--coco_path', type=str)
args = parser.parse_args()

charIdx = open('./data/char_table.txt', encoding='utf-8')
char_list = charIdx.read().split("\n")

def read_coco(coco_path):
    coco_imgs = glob.glob(coco_path+'train_words/'+'*.jpg')

    coco_anno = open(coco_path+'train_words_gt.txt','r')
    lines = coco_anno.readlines()
    coco_anno.close()

    coco_imgs = list()
    for line in lines:
        splitter = line.rstrip().split(',')

        if len(splitter) > 1:
            img_name = coco_path + 'train_words/' + splitter[0] + '.jpg'
            word_anno = splitter[1]

            flag = True
            for w in word_anno:
                if w not in char_list:
                    flag = False
                    break

            if flag == True:
                coco_imgs.append((img_name, word_anno))

    return coco_imgs


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

    # Read real image data
    coco_imgs = read_coco(args.coco_path)

    # Train the model
    # # Evaluate the model on the test set
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(params, coco_imgs))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: val_input_fn(params), throttle_secs=100)

    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.train(input_fn=lambda: train_input_fn(params, coco_imgs))