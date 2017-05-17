from PIL import Image
from os import listdir
from os.path import isfile, join

def app():
    # Alkuparemetrit
    r = 0
    g = 0
    b = 255

    folder = "testikuvakansio"
    fname = 'groundtruth.txt'

    type = "kulma linja vai alue" # ei vielä käytetty

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    groundtruthlist = []

    for image_file in onlyfiles:
        im = Image.open(folder + "/" + image_file)  # Can be many different formats.

        im = im.convert('RGB')

        im_width, im_heigth = im.size;
        im_data = list(im.getdata())

        h = 0
        w = 0

        pixel_index = 0
        target_pixel = (r,g,b)

        pixel_coordinates = []

        while h < im_heigth:
            while w < im_width:
                # vertailu
                if im_data[pixel_index] == target_pixel:
                    pixel_coordinates.append((w, h))

                w += 1
                pixel_index += 1

            w = 0
            h += 1

        print(pixel_coordinates)
        groundtruthlist.append(pixel_coordinates)

    #http: // stackoverflow.com / questions / 38712635 / writing - list - of - tuples - to - a - textfile - and -reading - back - into - a - list

    with open(fname, 'w') as f:
        for one_picture in groundtruthlist:
            f.write(str(one_picture))
            f.write("\n")


if __name__ == '__main__':
    app()
