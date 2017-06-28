# coding=utf-8
# Importteina ainakin PIL, numpy filereadit ja writet
import numpy
from PIL import Image, ImageDraw
from os import listdir
from os.path import isfile, join
from algoritmimockup import mockupalgorithm
import sys
#from sop import segmentation, blur, contourfindrectangle
from sop_testing_shit import segmentation, blur, contourfindrectangle, contourThatHasCentroid
import cv2
import ast
import math
import itertools
from timeit import default_timer as timer
import timeit


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


def safe_div(x, y):
    if y == 0:
        return 0
    return x / y


def algorithm_time_test(image, blurnumber, segmentationnumber, centroidx, centroidy):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # muuta kuva hsv vareiksi ei tarvi
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_gray = blur(blurnumber, image_gray);

    image_bw = segmentation(segmentationnumber, image_gray);

    # cnt, origin = contourThatHasCentroid(image, image_bw)  # eka frame tai kokoajan

    cnt, origin, centroidx, centroidy, areafound = contourThatHasCentroid(image_bw, centroidx, centroidy,
                                                                          True)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def app(argv, th=False):
    # Tää on ajettava python 2.7 ja opencv 3.1 (myös 3.x pitäis kelvata

    # Kuvakansion tiedostonimet
    # Jos siellä sattuu olee muutakin roskaa niin joudut erottelemaan .png päätteiset

    thstring = ""
    if th:
        thstring = "_threshold"

    main_folder = "mallitesti"
    groundtruthfile = 'groundtruth.txt'

    input_folder = "input"
    output_folder = "output"

    output_file = "result" + thstring + ".txt"
    output_file_large = "result_large" + thstring + ".txt"

    output_file_time = "result_running_times" + thstring + ".txt"

    #width = 320
    #height = 240

    #Cannyasetukset
    blurselection = 3
    segmentationselection = 2

    if th:
        blurselection = 3
        segmentationselection = 1

    correct_detection_distance_limit = 4

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

        height = len(im_numpy)
        width = len(im_numpy[0])

        areafoundfirsttime = True
        centroidx = width / 2
        centroidy = height / 2
        areafound = True


        image = im_numpy

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # muuta kuva hsv vareiksi ei tarvi
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image_gray = blur(blurselection, image_gray);

        image_bw = segmentation(segmentationselection, image_gray);

        #cnt, origin = contourThatHasCentroid(image, image_bw)  # eka frame tai kokoajan

        cnt, origin, centroidx_, centroidy_, areafound_ = contourThatHasCentroid(image_bw, centroidx, centroidy,
                                                                              areafound)

        # Aikamittaus algoritmista tassa kohti
        wrapped = wrapper(algorithm_time_test, image, blurselection, segmentationselection, centroidx, centroidy)
        extime = timeit.timeit(wrapped, number=1000)
        extime = extime / 1000

        testiajat.append(extime)

        draw = ImageDraw.Draw(im)


        result = []

        if(not isinstance(cnt, (int, long ))):

            for piste in cnt:
                result.append((piste[0][0],piste[0][1]))
                draw.line((piste[0][0] - 2, piste[0][1], piste[0][0] + 2, piste[0][1]), fill=128)
                draw.line((piste[0][0], piste[0][1] -2, piste[0][0], piste[0][1] + 2), fill=128)
                im.save(main_folder + "/" + output_folder + "/" + image_file_name.replace(".", thstring + "."))

            # result = mockupalgorithm(im_numpy)
        else:
            # TODO: saatava jotenkin myos se milloin palauttaa tyhjaa arvoa
            result.append((0, 0))
            result.append((0, 0))
            result.append((0, 0))
            result.append((0, 0))
            im.save(main_folder + "/" + output_folder + "/" + image_file_name.replace(".", thstring + "."))

        del draw
        testitulokset.append(result)


    #print testitulokset

    gt_data = []
    analysis_data = []
    analysis_data_printable = []
    
    with open(main_folder + "/" + groundtruthfile, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            gt_data.append(ast.literal_eval(line))

    c = 0

    distances_all = []

    for t in testitulokset:
        r, order, distances = smallest_result(t, gt_data[c])
        #print r
        analysis_data.append(r)
        distances_all.append(distances)

        analysis_data_printable.append("********************************************")
        analysis_data_printable.append("Pircture name: " + imagenames[c])
        analysis_data_printable.append("Result RMSE: " + str(r))
        analysis_data_printable.append("Ground   : " + str(gt_data[c]))
        analysis_data_printable.append("Input    : " + str(t))
        analysis_data_printable.append("Ordered  : " + str(order))
        analysis_data_printable.append("Distances: " + str(distances))
        analysis_data_printable.append("Avg distance: " + str((sum(distances)/len(distances))))
        analysis_data_printable.append("Number of wrong: " + str(sum(i > correct_detection_distance_limit for i in distances)))
        analysis_data_printable.append("Number of right: " + str(sum(i <= correct_detection_distance_limit for i in distances)))

        wrong_distances = [i for i in distances if i > correct_detection_distance_limit]
        right_distances = [i for i in distances if i <= correct_detection_distance_limit]

        analysis_data_printable.append("Avg of wrong distances: " + str(safe_div(sum(wrong_distances), len(wrong_distances))))
        analysis_data_printable.append("Avg of right distances: " + str(safe_div(sum(right_distances), len(right_distances))))
        analysis_data_printable.append("Running time: " + str(testiajat[c]))
        analysis_data_printable.append("********************************************")

        c = c + 1


    # Kokonaisanalyysin tekoa kaikista tuloksista
    analysis_data_printable.append("********************************************")
    analysis_data_printable.append("Analysis from all distances")
    analysis_data_printable.append("Number of pictures: " + str(len(distances_all)))
    analysis_data_printable.append("Right/wrong dist limit: " + str(correct_detection_distance_limit))

    overall_wrong = 0
    overall_right = 0

    for d in distances_all:
        overall_wrong += sum(i > correct_detection_distance_limit for i in d)
        overall_right += sum(i <= correct_detection_distance_limit for i in d)

    analysis_data_printable.append(
        "Overall wrongly detected corners: " + str(overall_wrong))
    analysis_data_printable.append(
        "Overall rightly detected corners: " + str(overall_right))
    analysis_data_printable.append(
        "Avg runtime: " + str(safe_div(sum(testiajat), len(testiajat))))

    # Luokkia

    correct = []
    failed1 = []
    failed2 = []
    failed3 = []
    failed4 = []

    for d in distances_all:
        wrongs = sum(i > correct_detection_distance_limit for i in d)
        if wrongs == 0:
            correct.append(d)
        if wrongs == 1:
            failed1.append(d)
        if wrongs == 2:
            failed2.append(d)
        if wrongs == 3:
            failed3.append(d)
        if wrongs == 4:
            failed4.append(d)

    analysis_data_printable.append("Correct detected pictures: " + str(len(correct)))
    analysis_data_printable.append("Correct detected avg distance: " + str(numpy.mean(numpy.array(correct))))
    analysis_data_printable.append("Correct detected distance standard deviation: " + str(numpy.std(numpy.array(correct))))
    analysis_data_printable.append("1 corner failed pictures: " + str(len(failed1)))
    analysis_data_printable.append("2 corner failed pictures: " + str(len(failed2)))
    analysis_data_printable.append("3 corner failed pictures: " + str(len(failed3)))
    analysis_data_printable.append("4 corner failed pictures: " + str(len(failed4)))


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

    #app(sys.argv[1:])
    app(sys.argv[1:], th=True)

    app(sys.argv[1:])
    #app(sys.argv[1:], th=True)
