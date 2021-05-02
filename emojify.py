import os, json
import cv2
import face_recognition


def main(matches, out_dir=os.path.abspath('low_output')):
    emoji = cv2.imread('emoji.jpg')
    for m in matches:
        print(m)
        img = cv2.imread(m)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(img)
        cv2.imshow('img', img)
        for y, xx, yy, x in face_locations:
            cv2.imshow('face', img[y: yy, x: xx])
            key = cv2.waitKey(0)
            if key == ord('a'):
                # blur = cv2.GaussianBlur(img[y-10: yy+10, x-10: xx+10], (51,51), 0)
                try:
                    roi = img[y-20: yy+20, x-20: xx+20, :]
                    h, w, _ = roi.shape
                    img[y-20: yy+20, x-20: xx+20, :] = cv2.resize(emoji.copy(), (w, h)) 
                except Exception as e:
                    print("[ERROR] ", e)
        cv2.destroyAllWindows()
        file = os.path.split(m)[-1]
        cv2.imwrite(os.path.join(out_dir, file), img)
        #     break
        # break


if __name__ == '__main__':
    matches1 = json.load(open('low_acc.json'))
    matches2 = json.load(open('high_acc.json'))
    # main(matches1, out_dir=os.path.abspath('low_output'))
    main(matches2, out_dir=os.path.abspath('high_output'))