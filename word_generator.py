import glob
import random
import numpy as np
import os
import math
import cv2

from PIL import Image, ImageDraw, ImageFont

START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

ref_font_size_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
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
ref_font_size_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

word_list = []
f = open('./data/full_corpus.txt','r')
for line in f:
    line = line.rstrip()
    word_list.append(line)
f.close()

charIdx = open('./data/char_table.txt', encoding='utf-8')
char_list = charIdx.read().split("\n")


def whitening(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def generate_image(word, font, fontSize=(18,80), imgSize=(1000, 60), position=(10, 10)):
    bgColor = np.random.randint(low=100, high=255)
    textColor = max(0, bgColor-np.random.randint(low=20, high=255))
    pilImage = Image.new("RGB", imgSize, (bgColor, bgColor, bgColor))
    draw = ImageDraw.Draw(pilImage)

    font_size = random.choice(ref_font_size_list)
    font = ImageFont.truetype(font, font_size)
    draw.text(position, word, (textColor, textColor, textColor), font=font)
    cvImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2GRAY)
    _, bwImage = cv2.threshold(cvImage, bgColor-1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bwImage, cv2.RETR_EXTERNAL, 2)[-2:]
    blobs = [cv2.boundingRect(cnt) for cnt in contours]
    if len(blobs) <= 0:
        result = cv2.resize(cvImage, (128, 32))
        return result
    minx = min([blob[0] for blob in blobs])
    miny = min([blob[1] for blob in blobs])
    maxx = max([blob[2]+blob[0] for blob in blobs])
    maxy = max([blob[3]+blob[1] for blob in blobs])
    cvImage = cvImage[
        max(0, miny-position[0]):maxy+position[0],
        max(0, minx-position[1]):maxx+position[1]
    ]
    cvImage = cv2.blur(cvImage, (random.choice([1, 3]), random.choice([1, 3])))
    h, w = cvImage.shape
    pts1 = np.float32([
        [np.random.randint(position[1]), np.random.randint(position[0])],
        [w-np.random.randint(position[1]-1), np.random.randint(position[0])],
        [w-np.random.randint(position[1]-1), h-np.random.randint(position[0]-1)],
        [np.random.randint(position[1]), h-np.random.randint(position[0]-1)]
    ])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    cvImage = cv2.warpPerspective(cvImage, M, (int(w), int(h)))
    if np.random.randint(3) == 0: # add noise with probability equals 1/3
        cvImage = cvImage.astype(np.int32)-np.random.randint(20, size=cvImage.shape)
        cvImage[cvImage < 0] = 0
        cvImage = cvImage.astype(np.uint8)
    if np.random.randint(10) == 0: # revert image with probability equals 1/10
        cvImage = 255-cvImage

    result = cv2.resize(cvImage, (128,32))
    # Whitening
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


def generate_data(batch_size, epochs):
    # [TBD] This generator should be a font-based generator, not image-based
    imgs = np.zeros((batch_size, 128, 32, 1))
    words = list()

    ret_imgs = np.zeros((batch_size, 128, 32, 1))
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
        for j in range(word_rate):
            font = random.choice(ref_fonts)
            word =  generate_word()
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
            imgs[start_idx+j][:] = image

            if "_upper" in font:
                word = word.upper()
            elif "_lower" in font:
                word = word.lower()
            words.append(word)

        start_idx = word_rate
        for j in range(number_rate):
            font = random.choice(ref_fonts)
            word =  generate_random_number(ref_max_num_length)
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
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
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
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
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
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
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
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
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
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
            image = generate_image(word, font)
            image = np.reshape(image, (128,32,1))
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
        gen_imgs = np.reshape(ret_imgs, [len(ret_imgs), 32, 128, 1])
        # yield {'imgs': ret_imgs, 'words': ret_words}
        # [None, 128, 32, 3], []
        yield (gen_imgs, decoder_input, seq_length), target_output
        words.clear()
        ret_words.clear()

        cnt += 1
