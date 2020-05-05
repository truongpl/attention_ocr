Implement fullflow of attention ocr using Tensorflow's seq2seq API and tf.Estimator

Migrate from bitbucket's repo

# Source structure

| Location             |  Content                                   |
|----------------------|--------------------------------------------|
| `/data`              | Fonts file, character table and corpus     |
| `train.py        `   | Train script                               |
| `word_generator.py ` | Utility to generate synthetic word from corpus |
| `params.json ` | Configuration |

# Train OCR model
```
python train.py --model_dir ./model_dir/
```
# Evaluate and predict
```
TBD
```
