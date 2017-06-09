# coding=utf-8
# Importteina ainakin PIL, numpy filereadit ja writet
import numpy
from PIL import Image, ImageDraw
from os import listdir
from os.path import isfile, join
from algoritmimockup import mockupalgorithm
import sys
from sop import segmentation, blur, contourfindrectangle
import cv2
import ast
import math
import itertools
from timeit import default_timer as timer


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

def coordinate_distance_array(array1, array2):
    c = 0
    distance_array = []
    for i in array1:
        distance_array.append(coordinate_distance(i, array2[c]))
        c = c + 1

    return distance_array

def smallest_result(values, groundtruth):
    result = 10000000000000000000

    coordinatepermutations = itertools.permutations(values)

    selectedpermutation = []

    selecteddistances = []

    for i in coordinatepermutations:
        #possibleresult = coordinate_distance_sum(i, groundtruth)

        e = coordinate_distance_array(i, groundtruth)

        possibleresult = rmse_errors(numpy.array(e))
        if (possibleresult < result):
            result = possibleresult
            selectedpermutation = i
            selecteddistances = e

    return result, selectedpermutation, selecteddistances


def rmse(algo_data, gt_data):  #https://stackoverflow.com/questions/21926020/how-to-calculate-rmse-using-ipython-numpy
    return numpy.sqrt(((algo_data - gt_data) ** 2).mean())    #https://stackoverflow.com/questions/17197492/root-mean-square-error-in-python


def rmse_errors(errors):
    return numpy.sqrt((errors ** 2).mean())


def app(argv):
    # Tää on ajettava python 2.7 ja opencv 3.1 (myös 3.x pitäis kelvata

    # Kuvakansion tiedostonimet
    # Jos siellä sattuu olee muutakin roskaa niin joudut erottelemaan .png päätteiset
    main_folder = "mallitesti"
    groundtruthfile = 'groundtruth.txt'

    input_folder = "input"
    output_folder = "output"

    output_file = "result.txt"
    output_file_large = "result_large.txt"

    output_file_time = "result_running_times.txt"

    try:
        main_folder = argv[0]
    except IndexError:
        pass

    imagenames = [f for f in listdir(main_folder + "/" + input_folder) if isfile(join(main_folder + "/" + input_folder, f))]

    imagenames.sort()

    # Alusta lista testituloksia varten
    testitulokset = []
    testiajat = []

    # Looppaa kuvatiedostot
    for image_file_name in imagenames:
        # Lataa kuva

        im = Image.open(main_folder + "/" + input_folder + "/" + image_file_name)
        im = im.convert('RGB')

        im_numpy = numpy.asarray(im) # http://stackoverflow.com/questions/384759/pil-and-numpy

        im_numpy = im_numpy[:, :, ::-1]  # PNG RGBA muunto BGR

        # Syötä numpy array algorimille
        # Oikeasti pitäis alustaa ohjelmaluokka ja kaikki

        # Ajanoton alku
        start = timer()
        image = im_numpy

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # muuta kuva hsv vareiksi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image_gray = blur(3, image_gray);

        image_bw = segmentation(2, image_gray);

        cnt, origin = contourfindrectangle(image, image_bw)  # eka frame tai kokoajan
        # Ajanoton loppu
        end = timer()

        testiajat.append(end - start)

        draw = ImageDraw.Draw(im)


        result = []

        if(not isinstance(cnt, (int, long ))):

            for piste in cnt:
                result.append((piste[0][0],piste[0][1]))
                draw.line((piste[0][0] - 2, piste[0][1], piste[0][0] + 2, piste[0][1]), fill=128)
                draw.line((piste[0][0], piste[0][1] -2, piste[0][0], piste[0][1] + 2), fill=128)
                im.save(main_folder + "/" + output_folder + "/" + image_file_name)

            # result = mockupalgorithm(im_numpy)
        else:
            result.append((0, 0))
            result.append((0, 0))
            result.append((0, 0))
            result.append((0, 0))
            im.save(main_folder + "/" + output_folder + "/" + image_file_name)

        del draw
        testitulokset.append(result)


    print testitulokset

    gt_data = []
    analysis_data = []
    analysis_data_printable = []
    
    with open(main_folder + "/" + groundtruthfile, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            gt_data.append(ast.literal_eval(line))

    c = 0

    for t in testitulokset:
        r, order, distances = smallest_result(t, gt_data[c])
        print r
        analysis_data.append(r)

        analysis_data_printable.append("********************************************")
        analysis_data_printable.append("Pircture name: " + imagenames[c])
        analysis_data_printable.append("Result RMSE: " + str(r))
        analysis_data_printable.append("Ground   : " + str(gt_data[c]))
        analysis_data_printable.append("Input    : " + str(t))
        analysis_data_printable.append("Ordered  : " + str(order))
        analysis_data_printable.append("Distances: " + str(distances))
        analysis_data_printable.append("Avg distance: " + str((sum(distances)/len(distances))))
        analysis_data_printable.append("Running time: " + str(testiajat[c]))
        analysis_data_printable.append("********************************************")


        if (r > 100):
            print t
            print gt_data[c]
            #tulos = rmse(t, gt_data[c])
            #analysis_data.append(tulos)

        c = c + 1

    with open(main_folder + "/" +output_file, 'w') as f:
        for tulos in analysis_data:
            f.write(str(tulos))
            f.write("\n")

    with open(main_folder + "/" +output_file_large, 'w') as f:
        for tulos in analysis_data_printable:
            f.write(str(tulos))
            f.write("\n")

    with open(main_folder + "/" +output_file_time, 'w') as f:
        for tulos in testiajat:
            f.write(str(tulos))
            f.write("\n")

    return


if __name__ == '__main__':
    app(sys.argv[1:])
