import sys
from numpy import ndarray
import numpy as np
from typing import NamedTuple, List, Tuple
import math

import cv2


class Point(NamedTuple):
    x: int
    y: int


class Line(NamedTuple):
    start: Point
    end: Point


def read_image(path: str) -> ndarray:
    return cv2.imread(path)


def convert_to_gray_image(image: ndarray) -> ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_edges(gray_image: ndarray) -> ndarray:
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    v = np.median(blurred)
    sigma = 1/3
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(blurred, lower, upper)


def find_lines(image: ndarray) -> ndarray:
    return cv2.HoughLines(image, 1, np.pi / 180, 200)


def remove_parallel_lines(lines: ndarray) -> ndarray:
    return lines[np.unique(lines[..., 1], axis=0, return_index=True)[1]]


def define_line(line: ndarray, image_shape: Tuple[int, int]) -> Line:
    rho, theta = line[0]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x0 = cos_theta * rho
    y0 = sin_theta * rho
    x1 = int(x0 + 2 * image_shape[0] * -sin_theta)
    y1 = int(y0 + 2 * image_shape[1] * cos_theta)
    x2 = int(x0 - 2 * image_shape[0] * -sin_theta)
    y2 = int(y0 - 2 * image_shape[1] * cos_theta)
    return Line(Point(x1, y1), Point(x2, y2))


def define_lines(lines: ndarray, image_shape: Tuple[int, int]) -> List[Line]:
    return [define_line(line, image_shape) for line in lines]


def calculate_intersection(line1: Line, line2: Line) -> ndarray:
    try:
        slope1 = (line1.start.y - line1.end.y) / (line1.start.x - line1.end.x)

        intercept1 = line1.start.y - slope1 * line1.start.x

        slope2 = (line2.start.y - line2.end.y) / (line2.start.x - line2.end.x)
        intercept2 = line2.start.y - slope2 * line2.start.x
    except ZeroDivisionError:
        raise ValueError
    if abs(slope1 - slope2) < sys.float_info.epsilon:
        raise ValueError

    x = (intercept2 - intercept1) / (slope1 - slope2)
    y = slope1 * x + intercept1
    return np.array([x, y])


def calculate_intersections(lines: List[Line]) -> List[ndarray]:
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            if line1 == line2:
                continue
            try:
                intersections.append(calculate_intersection(line1, line2))
            except ValueError:
                continue
    return intersections


def create_gausian_kernel(row_size: int, column_size: int) -> ndarray:
    x, y = np.meshgrid(np.linspace(-1, 1, row_size),
                       np.linspace(-1, 1, column_size))
    sigma, mu = 1.0, 0.0
    return np.exp(-((np.sqrt(x * x + y * y) - mu)**2 / (2.0 * sigma**2)))


def draw_gausian_at_intersection(image: ndarray, intersection: ndarray,
                                 original_image_shape: Tuple[int, int]
                                 ) -> ndarray:
    row_size = int(original_image_shape[0] / 100)
    column_size = int(original_image_shape[1] / 100)
    kernel = create_gausian_kernel(row_size, column_size)
    image[int(intersection[1] + original_image_shape[0] -
              row_size / 2):int(intersection[1] + original_image_shape[0] +
                                row_size / 2),
          int(intersection[0] + original_image_shape[1] -
              column_size / 2):int(intersection[0] + original_image_shape[1] +
                                   column_size / 2)] += kernel

    return image


def draw_gausian_at_intersections(intersections: List[ndarray],
                                  image_shape: Tuple[int, int]) -> ndarray:
    gausian_mask = np.zeros((3 * image_shape[0], 3 * image_shape[1]))
    for intersection in intersections:
        try:
            gausian_mask = draw_gausian_at_intersection(
                gausian_mask, intersection, image_shape)
        except ValueError:
            continue
    return gausian_mask


def convert_mask_to_image(mask: ndarray) -> ndarray:
    return (255 * mask / np.max(mask)).astype("uint8")


def threshold_image(image: ndarray, threshold_percentage: float) -> ndarray:
    return cv2.threshold(image, threshold_percentage * np.max(image),
                         np.max(image), np.min(image))[1]


def find_contours_blobs_in_thresholded_mask(image: ndarray) -> ndarray:
    kernel = np.ones((7, 7), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=3)
    dialated_image = cv2.dilate(eroded_image, kernel, iterations=3)
    contours, _ = cv2.findContours(dialated_image, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_center_contour(contour: ndarray) -> Point:
    moments = cv2.moments(contour)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return Point(x, y)


def find_center_contours(contours: ndarray) -> List[Point]:
    return [get_center_contour(contour) for contour in contours]


def make_final_image(original_image: ndarray) -> ndarray:
    return np.pad(
        original_image,
        ((int(original_image.shape[0]), int(original_image.shape[0])),
         (int(original_image.shape[1]), int(original_image.shape[1])), (0, 0)),
        constant_values=225)


def distance(point1: Point, point2: Point) -> float:
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def point_on_line(point: Point, line: Line, img_width: int,
                  img_height: int) -> bool:
    line = Line(Point(line.start.x + img_width, line.start.y + img_height),
                Point(line.end.x + img_width, line.end.y + img_height))
    distance_change = (distance(line.start, point) + distance(line.end, point)) % distance(
        line.start, line.end)
    relative_length_change = distance_change / distance(line.start, line.end)
    return relative_length_change < 0.00001


def draw_lines_to_vanishing_point(image: ndarray, lines: List[Line],
                                  vanishing_points: List[Point]) -> ndarray:
    for vanishing_point in vanishing_points:

        for line in lines:

            if point_on_line(vanishing_point, line, image.shape[0] / 3,
                             image.shape[1] / 3):
                cv2.line(image, (int(line.start.x + image.shape[0] / 3),
                                 int(line.start.y + image.shape[1] / 3)),
                         (int(line.end.x + image.shape[0] / 3),
                          int(line.end.y + image.shape[1] / 3)), (0, 225, 0),
                         5)
        cv2.circle(image, (int(vanishing_point.x), int(vanishing_point.y)), 30,
                   (0, 0, 255), -1)

    return image


def show_image(image: ndarray):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_defined_lines_in_image(image: ndarray) -> List[Line]:
    gray_image = convert_to_gray_image(image)
    edges = find_edges(gray_image)
    lines = find_lines(edges)
    return define_lines(lines, gray_image.shape)


def get_vanishing_points(intersections: List[ndarray],
                         image_shape: Tuple[int, int]) -> List[Point]:
    gausian_mask = draw_gausian_at_intersections(intersections, image_shape)
    gausian_mask_image = convert_mask_to_image(gausian_mask)
    thresholded_mask = threshold_image(gausian_mask_image, 0.7)
    contours = find_contours_blobs_in_thresholded_mask(thresholded_mask)
    return find_center_contours(contours)


def make_annotated_image(defined_lines: List[Line], image: ndarray,
                         vanishing_points: List[Point]) -> ndarray:
    final_image = make_final_image(image)
    return draw_lines_to_vanishing_point(final_image, defined_lines,
                                         vanishing_points)


def find_and_show_vanishing_points_in_image(image_path: str) -> List[Point]:
    image = read_image(image_path)
    defined_lines = get_defined_lines_in_image(image)
    intersections = calculate_intersections(defined_lines)
    vanishing_points = get_vanishing_points(intersections,
                                            (image.shape[0], image.shape[1]))
    annotated_image = make_annotated_image(defined_lines, image,
                                           vanishing_points)
    show_image(annotated_image)
    return vanishing_points
