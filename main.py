import argparse

import cv2
import numpy as np


def toSkeleton(image, erosion):
    # Params:
    thresh_val = 127
    max_val = 255
    open_iter = 10
    close_iter = 3


    size = np.size(image)

    _, image = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (erosion, erosion))
    # done = False

    skeleton = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    # skeleton = image

    # skeleton = cv2.erode(skeleton, kernel)
    # skeleton = cv2.dilate(skeleton, kernel)
    # skeleton = image

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
        #cv2.imwrite('/home/deepansh/Desktop/Mask.jpg',skeleton)
        x = cv2.waitKey(0)
        if x == 27:
            cv2.destroyWindow('Skeleton image')

    return skeleton


def toChannels(image):
    channels = cv2.split(image)
    for channel in channels:
        channel_img = cv2.bilateralFilter(channel, 20, 120, 1)
        # channel_img = cv2.GaussianBlur(channel, (3, 3), 0)

        if debug == 1:
            cv2.namedWindow("Channel image", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("Channel image", channel_img)
            x = cv2.waitKey(0)
            if x == 27:
                cv2.destroyWindow('Channel image')

    combined_channels = cv2.bitwise_and(*channels)

    return combined_channels


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


def toContour(image, edges):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if debug == 1:
        for i in range(len(contours)):
            cv2.namedWindow("Contour Lines", cv2.WINDOW_NORMAL)

            lines_image = np.zeros(image.shape, np.uint8)
            cv2.drawContours(lines_image, contours, i, (0, 255, 0), 2)

            cv2.imshow("Contour Lines", lines_image)
            x = cv2.waitKey(0)
            if x == 27:
                cv2.destroyWindow('Contour Lines')

    return contours


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
        # print(lines)
        pass

    return lines

def findBigContours(image,draw_contours):
	horizon_line=[]
	selected_contours = []
	height, width, _ = image.shape

	cntsSorted = sorted(draw_contours, key=lambda x: cv2.contourArea(x), reverse=True)


	for ind, c in enumerate(cntsSorted):

		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		#if extTop[1]<height/2:

		if extLeft[0]==0:
			if extRight[0]==width-1 or extTop[1]==0 or extBot[1]==height-1:
				horizon_line.append([extLeft,extRight,extTop,extBot])
				selected_contours.append(c)

		elif extBot[1]==height-1:
			if extRight[0]==width-1 or extTop[1]==0:
				horizon_line.append([extLeft,extRight,extTop,extBot])
				selected_contours.append(c)

		elif extTop[1]==0 and extRight[0]==width-1:
			horizon_line.append([extLeft,extRight,extTop,extBot])
			selected_contours.append(c)

	if debug == 1:
		for c in selected_contours:
			cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
			#cv2.imwrite('/home/deepansh/Desktop/Contour.jpg',image)
		
		cv2.imshow("Big C Image", image)
		cv2.waitKey(0)

	return selected_contours

def findElevation(image,contours):
	min=tuple(contours[0][contours[0][:, :, 1].argmax()][0])
	for c in contours:
		m=tuple(c[c[:, :, 1].argmax()][0])
		if m<min:
			min=m
	extBot=min
	height,width,_=image.shape
	elevation=(extBot[1]/height)*120
	#print(extBot)
	#print(height)
	print(elevation)
	if elevation <= 24:
		print("Good")
	elif elevation > 96:
		print("Bad")
	else:
		print("Mediocre")
	return elevation

def createMask(image, args):
	combined_image = toChannels(img)
	skeleton_image = toSkeleton(combined_image, args.erosion)
	canny_edges = toCanny(skeleton_image, args.sigma)
	draw_contours = toContour(img, canny_edges)
	contours=findBigContours(img,draw_contours)
	hough_lines = toHoughImage(img, canny_edges)
	elevation=findElevation(image,contours)
	return hough_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create sky mask for landscape')
    parser.add_argument('origin', help='Source of image file (JPG/PNG/TIFF/BMP supported)')
    parser.add_argument('-s', '--sigma', help='Sigma for Canny detection', type=float, default=0.33)
    parser.add_argument('-e', '--erosion', help='Erosion factor for Canny preparation', type=int, default=3)
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
    #cv2.imwrite('/home/deepansh/Desktop/Original.jpg',img)

    mask = createMask(img, args)

    if not args.destination:
        out_path = '{}_mask.png'.format(args.origin[args.origin.rfind('/')+1:args.origin.find('.')])
    else:
        out_path = args.destination

    #cv2.imwrite(out_path, mask)
