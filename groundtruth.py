from PIL import Image
from os import listdir
from os.path import isfile, join
import sys

def app(argv):
    # Alkuparemetrit
    r = 0
    g = 0
    b = 255

    main_folder = "mallitesti"
    groundtruth_folder = "groundtruth"

    fname = 'groundtruth.txt'

    try:
        main_folder = argv[0]
    except IndexError:
        pass

    onlyfiles = [f for f in listdir(main_folder + "/" + groundtruth_folder) if isfile(join(main_folder + "/" + groundtruth_folder, f))]

    onlyfiles.sort()

    groundtruthlist = []

    for image_file in onlyfiles:
        im = Image.open(main_folder + "/" + groundtruth_folder + "/" + image_file)  # Can be many different formats.

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

    #http://stackoverflow.com/questions/38712635/writing-list-of-tuples-to-a-textfile-and-reading-back-into-a-list
    with open(main_folder + "/" +fname, 'w') as f:
        for one_picture in groundtruthlist:
            f.write(str(one_picture))
            f.write("\n")


if __name__ == '__main__':
    app(sys.argv[1:])
