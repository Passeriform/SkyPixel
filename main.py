import argparse

import cv2
import numpy as np


def toSkeleton(image):
    skeleton = np.zeros(image.shape, np.uint8)
    size = np.size(image)

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # done = False

    skeleton = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=20)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=20)

    # while not done:
    #     eroded = cv2.erode(image, kernel)
    #     dilated = cv2.dilate(eroded, kernel)
    #     net = cv2.subtract(image, dilated)
    #     skeleton = cv2.bitwise_or(skeleton, net)
    #     image = eroded.copy()
    #
    #     zeroes = size - cv2.countNonZero(image)
    #     if zeroes == size:
    #         done = True

    if debug == 1:
        cv2.namedWindow("Skeleton image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Skeleton image", skeleton)
        x = cv2.waitKey(0)
        if x == 27:
            cv2.destroyWindow('Skeleton image')

    return skeleton


def toDefog(image):

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)

    if debug==1:
        cv2.namedWindow("Defogged image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Defogged image", sharpen)
        x = cv2.waitKey(0)
        if x == 27:
            cv2.destroyWindow('Defogged image')

    return sharpen


def toGrayscale(image):
    blurred = cv2.bilateralFilter(image, 20, 120, 1)
    # blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    if debug==1:
        cv2.namedWindow("Grayscale image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Grayscale image", gray)
        x = cv2.waitKey(0)
        if x == 27:
            cv2.destroyWindow('Grayscale image')

    return gray


def toCanny(image, sigma):
    median = np.median(image)

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    canny_edges = cv2.Canny(image, lower, upper)

    if debug==1:
        cv2.namedWindow("Canny Edge Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Canny Edge Detection", canny_edges)
        x = cv2.waitKey(0)
        if x == 27:
            cv2.destroyWindow('Canny Edge Detection')

    return canny_edges


def toHoughImage(image, edges):
    hough_lines = genHoughLines(edges)

    if debug==1:
        cv2.namedWindow("Hough Lines", cv2.WINDOW_NORMAL)

        # lines_image = np.zeros(image.shape, np.uint8)
        lines_image = image
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                # for rho, theta in line:
                #     a = np.cos(theta)
                #     b = np.sin(theta)
                #     x0 = a * rho
                #     y0 = b * rho
                #     x1 = int(x0 + 1000 * (-b))
                #     y1 = int(y0 + 1000 * (a))
                #     x2 = int(x0 - 1000 * (-b))
                #     y2 = int(y0 - 1000 * (a))
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("Hough Lines", lines_image)
        x = cv2.waitKey(0)
        if x == 27:
            cv2.destroyWindow('Hough Lines')

    return hough_lines


def genHoughLines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength=1, maxLineGap=10)

    if debug==1:
        print(lines)

    return lines


def createMask(image):
    grayscale_image = toGrayscale(img)
    skeleton_image = toSkeleton(grayscale_image)
    # defog_image = toDefog(grayscale_image)
    canny_edges = toCanny(skeleton_image, args.sigma)
    hough_lines = toHoughImage(img, canny_edges)

    return hough_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create sky mask for landscape')
    parser.add_argument('origin', help='Source of image file (JPG/PNG/TIFF/BMP supported)')
    parser.add_argument('-s', '--sigma', help='Sigma for Canny detection', type=float, default=0.33)
    parser.add_argument('-d', '--destination', help='Destination of image file (JPG/PNG/TIFF/BMP supported)')
    parser.add_argument('-v', '--debug', help='Debugging for intermediate verbosity', default=0, action='store_true')
    args = parser.parse_args()

    file = getattr(args, 'origin')
    if args.debug:
        debug = 1
    else:
        debug = 0

    img = cv2.imread(file)

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow("original", img)

    mask = createMask(img)

    if not args.destination:
        out_path = '{}_mask.png'.format(args.origin[args.origin.rfind('/')+1:args.origin.find('.')])
    else:
        out_path = args.destination

    cv2.imwrite(out_path, mask)
