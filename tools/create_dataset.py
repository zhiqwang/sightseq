import os
import lmdb
import cv2
import numpy as np
import argparse

def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    image_buf = np.fromstring(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[0], img.shape[1]
    if h * w == 0:
        return False
    return True

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def create_dataset(output_path, img_names, img_labels,
                   lexicon_list=None, check_valid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        output_path    : LMDB output path
        img_names      : list of image path
        img_labels     : list of corresponding groundtruth texts
        lexicon_list   : (optional) list of lexicon lists
        check_valid    : if true, check the validity of every image
    """
    # print (len(img_names) , len(img_labels))
    assert (len(img_names) == len(img_labels))
    nsamples = len(img_names)
    env = lmdb.open(output_path, map_size=1099511627776)

    cache = {}
    cnt = 1
    for i in range(nsamples):
        img_name = img_names[i]
        img_label = img_labels[i]
        if not os.path.exists(img_name):
            print('%s does not exist' % img_name)
            continue
        with open(img_name, 'rb') as f:
            image_bin = f.read()
        if check_valid:
            if not check_image_is_valid(image_bin):
                print('%s is not a valid image' % img_name)
                continue

        image_key = 'image-%09d' % cnt
        label_key = 'label-%09d' % cnt
        cache[image_key] = image_bin
        cache[label_key] = img_label.encode()
        if lexicon_list:
            lexicon_key = 'lexicon-%09d' % cnt
            cache[lexicon_key] = ' '.join(lexicon_list[i]).encode()
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nsamples))
        cnt += 1
    nsamples = cnt - 1
    cache['num-samples'] = str(nsamples).encode()
    write_cache(env, cache)
    print('Created dataset with %d samples' % nsamples)
