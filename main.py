from os.path import join, dirname

from vanishing_point.vanishing_point import (
    find_and_show_vanishing_points_in_image)


def main():
    images = [
        join(dirname(__file__), "files", "5D4KVN2Y_R.jpg"),
        join(dirname(__file__), "files", "5D4L1L1D_L.jpg")
    ]

    vanishing_points = [(image, find_and_show_vanishing_points_in_image(image))
                        for image in images]
    for image, vanishing_point in vanishing_points:
        print(f"the vanishing points for {image} are {vanishing_point}")


if __name__ == '__main__':
    main()
