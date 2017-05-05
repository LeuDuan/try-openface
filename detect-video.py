from __future__ import print_function
# from classifier import getRep, infer
import sys, time, cv2, pickle, argparse
import numpy as np
from sklearn.mixture import GMM
import logging
logging.basicConfig(level=logging.DEBUG)

import openface
net_model_path  = "models/nn4.small2.v1.t7"
dlib_model_path = "models/shape_predictor_68_face_landmarks.dat"
net = openface.TorchNeuralNet(net_model_path, imgDim=96, cuda=True)
align = openface.AlignDlib(dlib_model_path)



def getRep(bgrImg, multiple=False):
    '''
    bgrImg is numpy array which usually returned by called : imread()
    '''
    start_getRep = time.time()
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        end_getRep = time.time()
        logging.info("Get Represent: {0}".format(end_getRep - start_getRep))
        return None
    reps = []
    for bb in bbs:
       
        cv2.rectangle(bgrImg, (bb.left(), bb.bottom()),
                      (bb.right(), bb.top()), (255, 0, 0), 3)
        # cv2.imshow("G", bgrImg)
        alignedFace = align.align(
            96,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    end_getRep = time.time()
    logging.info("Get Represent: {0}".format(end_getRep - start_getRep))
    return sreps



def infer(bgrImgs,args, multiple=False):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')

    for bgrImg in bgrImgs:
        
        reps = getRep(bgrImg, multiple=False)
        if reps is None:
            return
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            if args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            if False:
                print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx,
                                                                         confidence))
            else:
                print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
                sss = "Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence)
                cv2.putText(bgrImg,sss,(100,150),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),1)
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))



if __name__ == "__main__":
    # config :
    parser = argparse.ArgumentParser()
    parser.add_argument('v', type=str, help="video path")
    parser.add_argument('f', type=int, help="from frame ?")
    parser.add_argument('classifierModel',type=str, help="path to classification model", default="./REPS/lfw/classifier.pkl")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.v)
    logging.info(args.v)

    i = 0
    while True:
        ret, frame = cap.read()
        i += 1
        if i < args.f:
            continue
        infer([frame],args,False)
        if not ret:
            break
        print( i)
        cv2.imshow(args.v, frame)
        cv2.waitKey(1)
