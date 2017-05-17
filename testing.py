# Importteina ainakin PIL, numpy filereadit ja writet

def app():
    # Alkuparemetrit, jotka on määriteltävä
        # Ground truth tekstitiedoston sijainti
        # Kuvakansion sijainti, kyseessä siis varsinainen testimateriaalikuvakansio, ei saa sekoittaa ground truth kuviin

    # Lue kuvakansion tiedostonimet
    # Jos siellä sattuu olee muutakin roskaa niin joudut erottelemaan .png päätteiset
    # testfolder = "testikuvakansio"
    # groundtruthfile = 'groundtruth.txt'
    # imagenames = [f for f in listdir(testfolder) if isfile(join(testfolder, f))]


    # Alusta varsinainen leipäohjelma
    # Joudut jussin kanssa jutella miten teet tämän
    # Jos ohjelmassa olisi varsinainen algoritmi omana funktionaan niin sitten voisi tehdä tyyliin
    # From SOP import algolritmi
    # Tai jos leipäohjelma paketoidaan luokaksi niin sitten luo luokan alkuparametrein ja kutsuu sen funktioita
    # Tämä on varmaan epäselvin osuus tässä ohjelmassa

    # Alusta lista testituloksia varten

    # Looppaa kuvatiedostot
    # for kuva in imagenames:
        # Lataa kuva
        # Mahdollisesti voit käyttää PIL im = Image.open(folder + "/" + image_file)
        # Muunna kuva muotoon jota algorimi syö, onko se RGB numpy array?
        # PNG kuvat ovat RGBA joten joutunet sellaisen luettua muuttamaan sen RGB:ksi im = im.convert('RGB')

        # Aloita ajanotto (jos on tarpeen)
        # Syötä numpy array algorimille
        # Ota vastaan algoritmin output (täytyy sopia myöhemmin. Onko se reunapikselit, kulmapikselit vai alue?)
        # Lopeta ajanotto (jos on tarpeen)

        # Syötä tulokset aikaisemmin alustettuu testituloslistaan

        # Uudelle kierrokselle loopissa


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
