import glob
import random
import numpy as np
import os
import math
import cv2
import imgaug.augmenters as iaa

from PIL import Image, ImageDraw, ImageFont

START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

global xxx
xxx = 0

# ref_font_size_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
ref_font_size_list = [30, 35, 45, 50, 55, 60, 65]
ref_fonts = glob.glob('./data/fonts/' +"*.ttf") + glob.glob('./data/fonts/' +"*.TTF") + glob.glob('./data/fonts/' +"*.otf") + glob.glob('./data/fonts/' +"*.OTF")
ref_date_format = ['MM/DD/YYYY', 'MM/DD/YY', 'MMM\'DDYY']
ref_month_list = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
ref_time_format = ['HH:MM:SS','HH:MM:AMPM']
ref_number_list = '0123456789'
ref_float_point = ['.',',']
ref_currency = ['$','€','¥','£']
ref_max_num_length = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
ref_special_template = ['X','*','(',':',','',.']
ref_www_ext = ['.com','.org','.gov','.net']

word_list = []
f = open('./data/full_corpus.txt','r')
for line in f:
    line = line.rstrip()
    word_list.append(line)
f.close()

charIdx = open('./data/char_table.txt', encoding='utf-8')
char_list = charIdx.read().split("\n")


# def whitening(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
#     y = np.multiply(np.subtract(x, mean), 1/std_adj)
#     return y

def whitening(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225), is_gray=True):
    # rgb order of VGG
    if is_gray == False:
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    else:
        mean = np.mean(in_img)
        std = np.std(in_img)
        std_adj = np.maximum(std, 1.0/np.sqrt(in_img.size))
        img = np.multiply(np.subtract(in_img, mean), 1/std_adj)

    return img



def get_augmenter():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    augmenter = iaa.Sequential(
        [
            sometimes(iaa.OneOf(
                [
                    iaa.GaussianBlur((0.8, 1.2)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.AveragePooling([2, 2]),
                    iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 2.0)),
                    iaa.AdditiveLaplaceNoise(scale=0.05*255, per_channel=True),
                    iaa.LinearContrast((0.5, 2.0), per_channel=True),
                    iaa.Clouds(),
                    iaa.Fog(),
                    iaa.PiecewiseAffine(scale=0.02),
                                            iaa.Affine(
                        scale={"x": (0.8, 1), "y": (0.8, 1)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-10, 10),
                        shear=(-5, 5),
                        order=[0, 1],
                        cval=(0, 255),
                        mode='constant'
                    ),
                ]
            )),
            # sometimes(iaa.Alpha(
            #             (0.3, 0.6),
            #             iaa.Affine(rotate=(-4, 4)),
            #             per_channel=0.5
            #         )
            # ),
            sometimes(iaa.OneOf(
                [
                    iaa.Crop(px=(2, 6)),
                    iaa.CoarseDropout((0.0, 0.01), size_percent=(0.02, 0.1)),
                ]
            )),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.02),  keep_size=False))
        ],
        random_order=True
    )

    return augmenter


def generate_image_v1(word, font, augmenter, img_shape=(128,32,1), imgSize=(5000, 5000), position=(100, 100)):
    # Loop until generate correct image
    bgColor = np.random.randint(low=100, high=255)
    textColor = max(0, bgColor-np.random.randint(low=20, high=255))
    pilImage = Image.new("RGB", imgSize, (bgColor, bgColor, bgColor))
    draw = ImageDraw.Draw(pilImage)

    # Draw raw text
    font_size = random.choice(ref_font_size_list)
    font_truetype = ImageFont.truetype(font, font_size)
    draw.text(position, word, (textColor, textColor, textColor), font=font_truetype)

    cvImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2GRAY)

    # colorImage = np.array(pilImage)
    colorImage = cvImage # Temporary use gray image since color doesn't provide good result

    _, bwImage = cv2.threshold(cvImage, bgColor-1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bwImage, cv2.RETR_EXTERNAL, 2)[-2:]

    blobs = [cv2.boundingRect(cnt) for cnt in contours]
    if not blobs:
        # print("Possible error here.........................")
        result = cv2.resize(colorImage, (img_shape[0], img_shape[1]))
        result = whitening(result)
        return result

    minx = min([blob[0] for blob in blobs])
    miny = min([blob[1] for blob in blobs])
    maxx = max([blob[2]+blob[0] for blob in blobs])
    maxy = max([blob[3]+blob[1] for blob in blobs])

    cvImage = cvImage[
        max(0,miny-10):maxy+10,
        max(0,minx-10):maxx+10
    ]

    colorImage = colorImage[
        max(0,miny-10):maxy+10,
        max(0,minx-10):maxx+10
    ]

    # new color result
    if augmenter is not None:
        colorImage = augmenter.augment_image(colorImage)

    # Resize and whitening
    result = cv2.resize(colorImage, (img_shape[0], img_shape[1]))
    result = whitening(result)

    return result


def generate_random_number(length_list):
    f = ""
    is_floating = random.choice([True, False])
    # Randomize number length
    length = random.choice(length_list)
    for _ in range(length):
        f = f + random.choice(ref_number_list)

    result = ""
    if is_floating == True:
        if f[0] == '0':
            result = f[0] + '.' + f[1:]
        else:
            index = np.random.randint(0, length)
            if index == 0:
                result = '0' + '.' + f
            else:
                result = f[0:index] + '.' + f[index:]
    else:
        result = f

    return result


def generate_random_currency():
    result = generate_random_number([2,3,4,5])

    currency = random.choice(ref_currency)

    is_space = random.choice([True, False])
    is_front = random.choice([True, False])

    if is_space == True:
        if is_front == True:
            result = currency + ' ' + result
        else:
            result = result + ' ' + currency
    else:
        if is_front == True:
            result = currency + result
        else:
            result = result + currency

    return result


def generate_date():
    date_format = random.choice(ref_date_format)
    # Generate random date
    yy = np.random.randint(0,100)
    if yy < 10:
        yy = '0' + str(yy)
    else:
        yy = str(yy)

    dd = np.random.randint(1,30)
    if dd < 10:
        dd = '0' + str(dd)
    else:
        dd = str(dd)

    mm = np.random.randint(1,12)
    if mm < 10:
        mm = '0' + str(mm)
    else:
        mm = str(mm)

    result = ''
    if date_format == ref_date_format[0]:
        # MM/DD/YYYY
        yyyy = np.random.randint(0,99)
        if yyyy < 10:
            yyyy = '0' + str(yyyy)
        else:
            yyyy = str(yyyy)
        yyyy = yyyy + yy
        result = mm + '/' + dd + '/' + yyyy
    elif date_format == ref_date_format[1]:
        # MM/DD/YY
        result = mm + '/' + dd + '/' + yy
    else:
        # MMM'DD
        mm = int(mm) - 1
        mm = ref_month_list[mm]
        result = mm + dd + '\'' + yy

    return result


def generate_time():
    time_format = random.choice(ref_time_format)

    # Generate random time
    # HH:
    hh = np.random.randint(0,24)
    if hh < 10:
        hh = '0' + str(hh)
    else:
        hh = str(hh)

    mm = np.random.randint(0,60)
    if mm < 10:
        mm = '0' + str(mm)
    else:
        mm = str(mm)

    ss = np.random.randint(0,60)
    if ss < 10:
        ss = '0' + str(ss)
    else:
        ss = str(ss)

    result = ''
    if time_format == ref_time_format[0]:
        result = hh + ':' + mm + ':' + ss
    else:
        time_frame = random.choice(['AM','PM','am','pm'])
        is_space = random.choice([True,False])

        if is_space == True:
            result = hh + ':' + mm + ' ' + time_frame
        else:
            result = hh + ':' + mm + time_frame

    return result


def generate_word():
    # Randomly pick word from corpus
    num_word = len(word_list)
    word_idx = np.random.randint(0, num_word)
    return word_list[word_idx]


def generate_double_word(self):
    # Pick two words from corpus
    num_word = len(word_list)

    word_idx1 = np.random.randint(0, num_word)
    word_idx2 = np.random.randint(0, num_word)

    result = word_list[word_idx1] + ' ' + word_list[word_idx2]
    return result


def generate_special_template():
    result = ''
    template = random.choice(ref_special_template)

    if template == ref_special_template[0] or template == ref_special_template[1]:
        # *** or XXX

        length = np.random.randint(4,9)
        for i in range(0, length):
            result += template

        # Generate for digit
        for _ in range(4):
            result += random.choice(ref_number_list)
    elif template == ref_special_template[2]:
        result += '('
        for i in range(0,3):
            result += random.choice(ref_number_list)
        result += ')'
    else:
        choice = random.choice([True,False])
        result = generate_word()

        if choice == True:
            result += template
        else:
            result = template + result

    return result

def generate_site():
    result = ''
    ext = random.choice(ref_www_ext)
    result += 'www.' + generate_word().replace('.','') + ext
    return result


def pad_char_idx(wordList, token=0, converter=None):
    # Sequences = [[1,2,3,4],[5,6,7,8,9]]
    max_length_word = max(map(lambda x: len(x), wordList))

    decoder_input = token * np.ones([len(wordList), max_length_word], dtype=np.int32)
    target_output = token * np.ones([len(wordList), max_length_word+1], dtype=np.int32)
    word_length = np.zeros(len(wordList), dtype=np.int32)

    for idx, word in enumerate(wordList):
        for charIdx, char in enumerate(word):
            try:
                index = converter.index(char)
            except:
                index = UNK_TOKEN

            decoder_input[idx, charIdx] = index
            target_output[idx, charIdx] = index

        decoder_input[idx, len(word):max_length_word] = token
        target_output[idx, len(word)] = END_TOKEN # At end token at end of word for loss

        word_length[idx] = len(word)

    return decoder_input, target_output, word_length


# Read COCO list
def generate_data(batch_size, epochs, augmenter, real_img_data=None, img_size=(128,32,1)):
    # [TBD] This generator should be a font-based generator, not image-based
    imgs = np.zeros((batch_size, img_size[0], img_size[1], img_size[2]))
    words = list()

    ret_imgs = np.zeros((batch_size, img_size[0], img_size[1], img_size[2]))
    ret_words = list()

    indexes = list(range(batch_size))

    cnt = 0
    print('Begin of 1 epoch')
    while cnt < epochs:

        word_rate = int(batch_size*4/8) # 4/8
        number_rate = int(batch_size*1/8) # 1/8
        currency_rate = int(batch_size*1/8) # 1/8
        time_rate = int(batch_size*1/8*2/4) # 2/4 of 1/8 for time 2/4 of 1/8 for date
        date_rate = int(batch_size*1/8*2/4)
        special_rate = int(batch_size*1/8*2/4)
        site_rate = int(batch_size*1/8*2/4)
        # double_word_rate = int(batch_size*1/8) # 1/8

        start_idx = 0

        if real_img_data is not None:
            # Generate some image of coco
            for j in range(word_rate//2):
                coco_data = random.choice(real_img_data)

                image = cv2.imread(coco_data[0])

                h, w , dim = image.shape
                if img_size[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to adapt with VGG norm
                image = cv2.resize(image, (img_size[0], img_size[1]))

                image = whitening(image)
                image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

                word = coco_data[1]

                imgs[start_idx+j][:] = image
                words.append(word)

            start_idx += word_rate//2

            for j in range(word_rate//2):
                font = random.choice(ref_fonts)
                word =  generate_word()
                # image = generate_image(word, font)
                # image = np.reshape(image, (128,32,1))
                image = generate_image_v1(word, font, augmenter, img_shape=img_size)
                image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

                imgs[start_idx+j][:] = image

                if "_upper" in font:
                    word = word.upper()
                elif "_lower" in font:
                    word = word.lower()
                words.append(word)

            start_idx += word_rate//2
        else:
            # Generate full synthetic word
            for j in range(word_rate):
                font = random.choice(ref_fonts)
                word =  generate_word()
                # image = generate_image(word, font)
                # image = np.reshape(image, (128,32,1))
                image = generate_image_v1(word, font, augmenter, img_shape=img_size)
                image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

                imgs[start_idx+j][:] = image

                if "_upper" in font:
                    word = word.upper()
                elif "_lower" in font:
                    word = word.lower()
                words.append(word)
            start_idx = word_rate

        # Generate some synthetic data
        for j in range(number_rate):
            font = random.choice(ref_fonts)
            word =  generate_random_number(ref_max_num_length)
            # image = generate_image(word, font)
            # image = np.reshape(image, (128,32,1))
            image = generate_image_v1(word, font, augmenter, img_shape=img_size)
            image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        start_idx = word_rate + number_rate
        for j in range(currency_rate):
            font = random.choice(ref_fonts)
            word =  generate_random_currency()
            # image = generate_image(word, font)
            # image = np.reshape(image, (128,32,1))

            image = generate_image_v1(word, font, augmenter, img_shape=img_size)
            image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        start_idx = word_rate + number_rate + currency_rate
        for j in range(time_rate):
            font = random.choice(ref_fonts)
            word =  generate_time()
            # image = generate_image(word, font)
            # image = np.reshape(image, (128,32,1))

            image = generate_image_v1(word, font, augmenter, img_shape=img_size)
            image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        start_idx = word_rate + number_rate + currency_rate + time_rate
        for j in range(date_rate):
            font = random.choice(ref_fonts)
            word =  generate_date()
            # image = generate_image(word, font)
            # image = np.reshape(image, (128,32,1))

            image = generate_image_v1(word, font, augmenter, img_shape=img_size)
            image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        start_idx = word_rate + number_rate + currency_rate + time_rate + date_rate
        for j in range(special_rate):
            font = random.choice(ref_fonts)
            word =  generate_special_template()
            # image = generate_image(word, font)
            # image = np.reshape(image, (128,32,1))

            image = generate_image_v1(word, font, augmenter, img_shape=img_size)
            image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        start_idx = word_rate + number_rate + currency_rate + time_rate + date_rate + special_rate
        for j in range(site_rate):
            font = random.choice(ref_fonts)
            word =  generate_site()
            # image = generate_image(word, font)
            # image = np.reshape(image, (128,32,1))

            image = generate_image_v1(word, font, augmenter, img_size)
            image = np.reshape(image, (img_size[0], img_size[1], img_size[2]))

            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        # Reshuffle the order
        random.shuffle(indexes)

        for i in range(0, batch_size):
            ret_imgs[i][:] = imgs[indexes[i]][:]
            ret_words.append(words[indexes[i]])

        # Pad ret words
        decoder_input, target_output, seq_length = pad_char_idx(ret_words, PAD_TOKEN, char_list)
        gen_imgs = np.reshape(ret_imgs, [len(ret_imgs), img_size[1], img_size[0], img_size[2]]) # Reshape to 120, 150

        yield (gen_imgs, decoder_input, seq_length), target_output
        words.clear()
        ret_words.clear()

        cnt += 1
