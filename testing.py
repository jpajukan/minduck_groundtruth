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
        # Tässä käyteään vaan muokattua mockup algoritmia 5 kuukautta vanhasta sop.py:stä
        # Oikeasti pitäis alustaa ohjelmaluokka ja kaikki
        # muuta kuva mustavalkoiseksi
        image = im_numpy

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # muuta kuva hsv vareiksi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image_gray = blur(3, image_gray);

        image_bw = segmentation(2, image_gray);

        cnt, origin = contourfindrectangle(image, image_bw)  # eka frame tai kokoajan

        result = cnt
        # result = mockupalgorithm(im_numpy)
        # Ota vastaan algoritmin output (täytyy sopia myöhemmin. Onko se reunapikselit, kulmapikselit vai alue?)

        # Syötä tulokset aikaisemmin alustettuu testituloslistaan
        testitulokset.append(result)


    # Tälläinen on suunnilleen lopullinen outputti, eli vaan joukko pikseleitä
    #print testitulokset

    # TODO:
    # Lue ground truth tiedosto
    # Joka rivillä siellä on yhden kuvan merkittävät pikselit koordinaattitupleina
    # Lue listaan seuraavasti
    # http://stackoverflow.com/questions/38712635/writing-list-of-tuples-to-a-textfile-and-reading-back-into-a-list

    gt_data = []

    with open(groundtruthfile, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines:
            gt_data.append(ast.literal_eval(line))
        #retreived_ds = ast.literal_eval(f.read())
    # En ole kokeillut toimiiko, ja joudut tehä varmaan loopilla jokaisen rivin lukemisen erikseen

    c = 0

    for t in testitulokset:
        print t
        print gt_data[c]
        c = c + 1

    # Yhdistä saamasi algoritmitulokset ja ground truth tiedosto matlabin datatiedostoksi (onko se .mat?)
    # Ei mitään hajua miten tämä tehdään. Voit joutua luomaan useitakin tiedostoja


    return


if __name__ == '__main__':
    app(sys.argv[1:])
