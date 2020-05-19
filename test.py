import requests
import cv2
import numpy as np
import json
from utils import Params


SERVER_URL = 'http://localhost:8501/v1/models/ocr:predict'

f = open('./data/char_table.txt','r')
lines = f.readlines()
f.close()

char_list = list()
for line in lines:
    char_list.append(line.rstrip())


def image_std(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def resize_padding(img, new_width, new_height, interp=0):
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 1), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw,0] = img

    return image_padded, resize_ratio, dw, dh



def convert_idx2char(pred):
    def idx2Word(predictions):
        result = ''
        for idx in predictions:
            if char_list[idx] == 'UNK':
                result += ''
            else:
                result += char_list[idx]
        return result

    logits = pred[:]
    predWord = list(logits)

    start_idx = 0
    try:
        end_idx = predWord.index(char_list.index('EOS'))
    except:
        end_idx = len(predWord)

    predWordIdx = predWord[start_idx:end_idx]
    pred_word = idx2Word(predWordIdx)
    return pred_word, end_idx


def extract(img_list, host):
    image = np.reshape(img_list,(1,80,100,1))
    x = {"signature_name": "serving_default", "inputs": {'image': image.tolist()}}

    headers = {"content-type": "application/json"}

    data = json.dumps(x)

    resp = requests.post("http://127.0.0.1:8501/v1/models/ocr:predict", data=data, headers=headers)
    if resp.status_code == 200:
        predictions = json.loads(resp.text)['outputs']
        result = []
        for pred in predictions:
            result.append(convert_idx2char(pred)[0])
        return result
    else:
        print('Cannot call to serving ', resp.status_code)
        return None


# Test a single image
image = cv2.imread('./sample_img/1.png',0)
image = cv2.resize(image, (150,120))
image = image_std(image)

result = extract([image], SERVER_URL)
print(result)
