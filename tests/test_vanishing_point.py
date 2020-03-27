import math
import numpy as np
import pytest
from vanishing_point.vanishing_point import (
    remove_parallel_lines, define_line, Line, Point, define_lines,
    calculate_intersection, calculate_intersections, create_gausian_kernel,
    draw_gausian_at_intersection, draw_gausian_at_intersections, distance,
    point_on_line)


@pytest.fixture(name="line")
def fixture_line():
    yield np.array([[1, 10]])


@pytest.fixture(name="defined_line")
def fixture_defined_line():
    yield Line(Point(10, -17), Point(-11, 16))


@pytest.fixture(name="defined_line2")
def fixture_defined_line2():
    yield Line(Point(-17, 10), Point(16, -11))


@pytest.fixture(name="gausian_kernel")
def fixture_gausian_kernel_5_5():
    yield np.array(
        [[0.36787944, 0.53526143, 0.60653066, 0.53526143, 0.36787944],
         [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
         [0.60653066, 0.8824969, 1., 0.8824969, 0.60653066],
         [0.53526143, 0.77880078, 0.8824969, 0.77880078, 0.53526143],
         [0.36787944, 0.53526143, 0.60653066, 0.53526143, 0.36787944]])


def test_remove_parallel_lines():
    lines = np.array([(1, 10), (1, 5), (1, 10), (1, 5)])
    assert np.all(remove_parallel_lines(lines) == [[1, 5], [1, 10]])


def test_define_line(line, defined_line):
    assert define_line(line, (10, 10)) == defined_line


def test_define_lines(line, defined_line):
    lines = np.array([line, line, line])
    assert define_lines(
        lines, (10, 10)) == [defined_line, defined_line, defined_line]


def test_calclulate_intersection(defined_line, defined_line2):
    assert np.allclose(calculate_intersection(defined_line, defined_line2),
                       [-0.5, -0.5])


def test_calculate_intersections(defined_line, defined_line2):
    lines = [defined_line, defined_line2]
    assert np.allclose(calculate_intersections(lines), [[-0.5, -0.5]])


def test_create_gausian_kernel(gausian_kernel):
    assert np.allclose(create_gausian_kernel(5, 5), gausian_kernel)


def test_draw_gausian_at_intersection(gausian_kernel):
    image = np.zeros((600, 600))
    result = np.zeros((600, 600))
    result[300, 300] = create_gausian_kernel(1, 1)
    assert np.allclose(
        draw_gausian_at_intersection(image, np.array([201, 201]), (100, 100)),
        result)


def test_draw_gausian_at_intersections():
    intersections = [np.array([101, 101])]
    image_shape = (100, 100)
    result = np.zeros((300, 300))
    result[200, 200] = create_gausian_kernel(1, 1)
    assert np.allclose(
        draw_gausian_at_intersections(intersections, image_shape), result)


def test_distance():
    assert distance(Point(10, 10), Point(20, 20)) == math.sqrt(200)


def test_point_on_line():
    line = Line(Point(10, 10), Point(20, 20))
    point = Point(15, 15)
    assert point_on_line(point, line, 0, 0)


def test_fail_point_on_line():
    line = Line(Point(10, 10), Point(20, 20))
    point = Point(17, 16)
    assert not point_on_line(point, line, 0, 0)
