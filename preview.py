import os, json
import cv2

matches1 = json.load(open('matches_1.json'))
matches2 = json.load(open('matches_2.json'))

# low_acc = []
# for match in matches1:
#     for m in match:
#         print(m)
#         img = cv2.imread(m)
#         cv2.imshow('img', cv2.resize(img, (0, 0), fx=0.3, fy=0.3))
#         key = cv2.waitKey(0)
#         if key == ord('a'):
#             low_acc.append(m)
#         elif key == ord('q'):
#             break
#         cv2.destroyAllWindows()

# json.dump(low_acc, open('low_acc.json', 'w'))

# input("----------- EOF -------------\n\n --------- Hit Enter ---------")

high_acc = []
for match in matches2:
    for m in match:
        print(m)
        img = cv2.imread(m)
        cv2.imshow('img', cv2.resize(img, (0, 0), fx=0.3, fy=0.3))
        key = cv2.waitKey(0)
        if key == ord('a'):
            high_acc.append(m)
        elif key == ord('q'):
            break
        cv2.destroyAllWindows()

json.dump(high_acc, open('high_acc.json', 'w'))