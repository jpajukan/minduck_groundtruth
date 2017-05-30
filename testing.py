# coding=utf-8
# Importteina ainakin PIL, numpy filereadit ja writet
import numpy
from PIL import Image
from os import listdir
from os.path import isfile, join
from algoritmimockup import mockupalgorithm
import sys
from sop import segmentation, blur, contourfindrectangle
import cv2
import ast
import math
import itertools


def coordinate_distance(c1, c2):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]

    distance = math.sqrt((dx**2) + (dy**2))

    return distance


def coordinate_distance_sum(array1, array2):
    c = 0
    distance_sum = 0
    for i in array1:
        distance_sum = distance_sum + coordinate_distance(i, array2[c])
        c = c + 1

    return distance_sum

def smallest_result(values, groundtruth):
    result = 10000000000000

    coordinatepermutations = itertools.permutations(values)

    for i in coordinatepermutations:
        possibleresult = coordinate_distance_sum(i, groundtruth)

        if (possibleresult < result):
            result = possibleresult

    return result

def rmse(algo_data, gt_data):  #https://stackoverflow.com/questions/21926020/how-to-calculate-rmse-using-ipython-numpy
    return numpy.sqrt(((algodata - gt_data) ** 2).mean())    #https://stackoverflow.com/questions/17197492/root-mean-square-error-in-python

def app(argv):
    # Tää on ajettava python 2.7 ja opencv 3.1 (myös 3.x pitäis kelvata

    # Kuvakansion tiedostonimet
    # Jos siellä sattuu olee muutakin roskaa niin joudut erottelemaan .png päätteiset
    folder = "oikeakuvakansio"
    groundtruthfile = 'groundtruth.txt'

    try:
        folder = argv[0]
        groundtruthfile = argv[1]
    except IndexError:
        pass

    imagenames = [f for f in listdir(folder) if isfile(join(folder, f))]

    imagenames.sort()

    # Alusta lista testituloksia varten
    testitulokset = []

    # Looppaa kuvatiedostot
    for image_file_name in imagenames:
        # Lataa kuva

        im = Image.open(folder + "/" + image_file_name)
        im = im.convert('RGB')

        im_numpy = numpy.asarray(im) # http://stackoverflow.com/questions/384759/pil-and-numpy

        im_numpy = im_numpy[:, :, ::-1]  # PNG RGBA muunto BGR

        # Syötä numpy array algorimille
        # Oikeasti pitäis alustaa ohjelmaluokka ja kaikki
        image = im_numpy

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # muuta kuva hsv vareiksi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image_gray = blur(3, image_gray);

        image_bw = segmentation(2, image_gray);

        cnt, origin = contourfindrectangle(image, image_bw)  # eka frame tai kokoajan

        result = []
        for piste in cnt:
            result.append((piste[0][0],piste[0][1]))
        # result = mockupalgorithm(im_numpy)

        testitulokset.append(result)


    print testitulokset

    gt_data = []
    analysis_data = []
    
    with open(groundtruthfile, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            gt_data.append(ast.literal_eval(line))

    c = 0

    for t in testitulokset:
        r = smallest_result(t, gt_data[c])
        print r

        if (r > 100):
            print t
            print gt_data[c]
            tulos = rmse(t, gt_data[c])
            analysis_data.append(tulos)

        c = c + 1
    return


if __name__ == '__main__':
    app(sys.argv[1:])
