# coding=utf-8
# Importteina ainakin PIL, numpy filereadit ja writet
import numpy
from PIL import Image
from os import listdir
from os.path import isfile, join
from algoritmimockup import mockupalgorithm

def app():
    # Tää on ajettava python 2.7 ja opencv 3.1 (myös 3.x pitäis kelvata

    # Kuvakansion tiedostonimet
    # Jos siellä sattuu olee muutakin roskaa niin joudut erottelemaan .png päätteiset
    folder = "oikeakuvakansio"
    groundtruthfile = 'groundtruth.txt'
    imagenames = [f for f in listdir(folder) if isfile(join(folder, f))]


    # Alusta varsinainen leipäohjelma
    # Joudut jussin kanssa jutella miten teet tämän
    # Jos ohjelmassa olisi varsinainen algoritmi omana funktionaan niin sitten voisi tehdä tyyliin
    # From SOP import algolritmi
    # Tai jos leipäohjelma paketoidaan luokaksi niin sitten luo luokan alkuparametrein ja kutsuu sen funktioita
    # Tämä on varmaan epäselvin osuus tässä ohjelmassa

    # Alusta lista testituloksia varten
    testitulokset = []

    # Looppaa kuvatiedostot
    for image_file_name in imagenames:
        # Lataa kuva

        im = Image.open(folder + "/" + image_file_name)
        im = im.convert('RGB') # PNG RGBA muunto BGR #todo: INPUTIN ON OLTAVA BGR!!!!!
        im_numpy = numpy.asarray(im) # http://stackoverflow.com/questions/384759/pil-and-numpy

        # Syötä numpy array algorimille
        # Tässä käyteään vaan muokattua mockup algoritmia 5 kuukautta vanhasta sop.py:stä
        result = mockupalgorithm(im_numpy)
        # Ota vastaan algoritmin output (täytyy sopia myöhemmin. Onko se reunapikselit, kulmapikselit vai alue?)

        # Syötä tulokset aikaisemmin alustettuu testituloslistaan
        testitulokset.append(result)


    # Tälläinen on suunnilleen lopullinen outputti, eli vaan joukko pikseleitä
    print testitulokset

    # TODO:
    # Lue ground truth tiedosto
    # Joka rivillä siellä on yhden kuvan merkittävät pikselit koordinaattitupleina
    # Lue listaan seuraavasti
    # http://stackoverflow.com/questions/38712635/writing-list-of-tuples-to-a-textfile-and-reading-back-into-a-list
    #with open(fname, 'r') as f:
        #retreived_ds = ast.literal_eval(f.read())
    # En ole kokeillut toimiiko, ja joudut tehä varmaan loopilla jokaisen rivin lukemisen erikseen


    # Yhdistä saamasi algoritmitulokset ja ground truth tiedosto matlabin datatiedostoksi (onko se .mat?)
    # Ei mitään hajua miten tämä tehdään. Voit joutua luomaan useitakin tiedostoja


    return


if __name__ == '__main__':
    app()
