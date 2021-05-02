import os, json
import random
from glob import glob
from pprint import pprint

import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
import ray
ray.init()

def get_file_paths(root='D:\PHOTOS\*'):
    queue = [root]
    files = []
    while len(queue) > 0:
        dir = queue.pop(0)
        for file in glob(dir):
            if os.path.isfile(file) and file.endswith('JPG'):
                files.append(file)
            else:
                queue.append(file + '\*')
    return files


def get_my_image(dir=os.path.abspath('database')):
    db_imgs = []
    db_enc = []
    for file in glob(dir + '\*'):
        suff = os.path.split(file)[-1]
        if suff.startswith('arham'):
            img = cv2.imread(file)
            encoding = face_recognition.face_encodings(img)[0]
            db_imgs.append(img)
            db_enc.append(encoding)

    return db_imgs, np.array(db_enc)

@ray.remote
def get_similar_imgs(files):
    my_imgs, my_enc = get_my_image()
    matches = []
    pbar = tqdm(total=len(files))
    tc = 1
    for file in files:
        # file = r'D:\PHOTOS\DAY 2\5 BARAT\_IMK5917.JPG'
        img = cv2.imread(file) # r'D:\PHOTOS\1 day 1 haldi mehandi\_IMK4982.JPG' r'D:\PHOTOS\DAY 2\5 BARAT\_IMK5917.JPG'
        img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)

        face_locations = face_recognition.face_locations(img)
        encoding = face_recognition.face_encodings(img)
        # print(face_locations)
        # print("==> Matching ", len(encoding), "Faces")
        # print(my_enc)

        idx2 = 0
        for enc in encoding:
            results = face_recognition.compare_faces(my_enc, enc, tolerance=0.3)
            # print("results=>", results)
            idx = 0
            for res in results:
                if res:
                    matches.append(file)
                    # print("Found match in", file)
                    y, xx, yy, x = face_locations[idx2]
                    # cv2.imshow(f'img1_{random.randint(1, 1000)}', my_imgs[idx])
                    # cv2.imshow(f'img2_{random.randint(1, 1000)}', img[y: yy, x: xx])
                    # cv2.imshow('img3', img)
                    # cv2.waitKey(1)
                idx += 1
            idx2 += 1
        if tc%10 == 0:
            cv2.destroyAllWindows()
        pbar.update(tc)
        tc += 1
        # break
    pbar.close()
    # print("Matched files=>", matches)
    print(f"[INFO] processed {len(matches)} images")
    return matches


def main():
    files = get_file_paths(root=r'G:\PHOTOS\*')
    print("files count: ", len(files))
    pprint(files)
    process_count = 8
    chunk = len(files)//process_count
    futures = []
    print("len->", len(files))
    for i in range(process_count):
        start, end = i*chunk, (i+1)*chunk
        print(start,end)
        fs = files[start: end]
        futures.append(get_similar_imgs.remote(fs))
    
    matches = ray.get(futures)
    json.dump(matches, open('matches_2.json', 'w'))


if __name__ == "__main__":
    main()