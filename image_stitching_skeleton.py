import cv2
import sys
import numpy as np

np.set_printoptions(threshold=np.inf)

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    print('Finding homography...')
    best_H = None

    mostInliers = []
    for i in range(0, max_num_trial):
    	randFP = []
    	for r in range(0, len(list_pairs_matched_keypoints)):
    		randFP.append(list_pairs_matched_keypoints[np.random.randint(0, high=len(list_pairs_matched_keypoints))])
    	matrixList = []

    	for match in range(0, len(randFP)):
    		
    		point_img1 = [randFP[match][0][0],randFP[match][0][1],1]
    		point_img2 = [randFP[match][1][0],randFP[match][1][1],1]

    		mata = [-point_img2[2] * point_img1[0], -point_img2[2] * point_img1[1], -point_img2[2] * point_img1[2],
    		0,0,0,
    		point_img2[0] * point_img1[0], point_img2[0] * point_img1[1], point_img2[0] * point_img1[2]]

    		matb = [0,0,0,
    		-point_img2[2] * point_img1[0], -point_img2[2] * point_img1[1], -point_img2[2] * point_img1[2],
    		point_img2[1] * point_img1[0], point_img2[1] * point_img1[1], point_img2[1] * point_img1[2]]

    		matrixList.append(mata)
    		matrixList.append(matb)


    	compareMatrix = np.matrix(matrixList)
    	u, s, v = np.linalg.svd(compareMatrix)
    	h = np.reshape(v[8], (3, 3))
    	h = (1 / h.item(8)) * h

    	inliers = []

    	for d in range(0, len(list_pairs_matched_keypoints)):
    		point1 = np.transpose(np.matrix([list_pairs_matched_keypoints[0][0][0],list_pairs_matched_keypoints[0][0][1],1]))
    		estpoint2 = np.dot(h, point1)
    		estpoint2 = (1/estpoint2.item(2))*estpoint2

    		point2 = np.transpose(np.matrix([list_pairs_matched_keypoints[0][1][0],list_pairs_matched_keypoints[0][1][1],1]))
    		error = point2 - estpoint2
    		dist = np.linalg.norm(error)
    		if dist < threshold_reprojtion_error:
    			inliers.append(list_pairs_matched_keypoints[d])

    	if len(inliers) > len(mostInliers):
    		mostInliers = inliers
    		best_H = h
    
    print('Found homography!')
    return best_H

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    print('Detecting Feature Points...')
    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    gray2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(gray2, None)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================

    sameFP = []
    eachMinDist = []
    # eachMinIndex = []
    for i in range(0, len(kp)):
    	minimumDist = []
    	minDistIndices = []
    	for j in range(0, len(kp2)):
    		minimumDist.append(np.linalg.norm(np.subtract(des[i], des2[j])))
    		minDistIndices.append([i,j])

    	minIndex = np.argmin(minimumDist)
    	tempMinDist = np.delete(minimumDist, minIndex)
    	secondMinIndex = np.argmin(tempMinDist)
    	
    	if minimumDist[minIndex] / minimumDist[secondMinIndex] < ratio_robustness:
    	    sameFP.append(minDistIndices[minIndex])
    	    eachMinDist.append(minimumDist[minIndex])

    lowestDistFP = []
    numberOfTotalFP = 25

    for i in range(0, numberOfTotalFP):
    	temp = np.argmin(eachMinDist)
    	lowestDistFP.append(sameFP[temp])
    	eachMinDist = np.delete(eachMinDist, temp)
    	sameFP = np.delete(sameFP, temp, 0)
    	

    list_pairs_matched_keypoints = []

    for i in range(0,len(lowestDistFP)):
	    list_pairs_matched_keypoints.append([[np.float(kp[lowestDistFP[i][0]].pt[0]), np.float(kp[lowestDistFP[i][0]].pt[1])], [np.float(kp2[lowestDistFP[i][1]].pt[0]), np.float(kp2[lowestDistFP[i][1]].pt[1])]])

    print('Found Feature Points!')

    return list_pairs_matched_keypoints

def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    print('Starting Transformation...')
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...
    invh = np.linalg.inv(H_1)
    height1, width1, channels1 = img_1.shape
    height2, width2, channels2 = img_2.shape

    #image 1 corner finding
    t = np.dot(H_1, [0,0,1])
    topLeftimg1 = []
    topLeftimg1.append(t.item(0) / t.item(2))
    topLeftimg1.append(t.item(1) / t.item(2))
    topLeftimg1[0] = round(topLeftimg1[0])
    topLeftimg1[1] = round(topLeftimg1[1])

    t = np.dot(H_1, [width1, 0, 1])
    topRightimg1 = []
    topRightimg1.append(t.item(0) / t.item(2))
    topRightimg1.append(t.item(1) / t.item(2))
    topRightimg1[0] = round(topRightimg1[0])
    topRightimg1[1] = round(topRightimg1[1])

    t = np.dot(H_1, [0, height1, 1])
    botLeftimg1 = []
    botLeftimg1.append(t.item(0) / t.item(2))
    botLeftimg1.append(t.item(1) / t.item(2))
    botLeftimg1[0] = round(botLeftimg1[0])
    botLeftimg1[1] = round(botLeftimg1[1])

    t = np.dot(H_1, [width1, height1, 1])
    botRightimg1 = []
    botRightimg1.append(t.item(0) / t.item(2))
    botRightimg1.append(t.item(1) / t.item(2))
    botRightimg1[0] = round(botRightimg1[0])
    botRightimg1[1] = round(botRightimg1[1])

    #image 2 corner finding
    topLeftimg2 = [0,0]
    topRightimg2 = [width2, 0]
    botLeftimg2 = [0, height2]
    botRightimg2 = [width2, height2]

    # find most left x value for cropping
    minx = 0
    if topLeftimg1[0] < botLeftimg1[0]:
    	minx = topLeftimg1[0]
    else:
    	minx = botLeftimg1[0]

    # find highest y value for cropping
    potentialminy = []
    potentialminy.extend((topLeftimg1[1], topRightimg1[1], topLeftimg2[1], topRightimg2[1]))
    miny = min(potentialminy)

    #find lowest y value for cropping
    potentialmaxy = []
    maxy = 0
    potentialmaxy.extend((botLeftimg1[1], botRightimg1[1], botLeftimg2[1], botRightimg2[1]))
    maxy = max(potentialmaxy)
    
    # find most right x value for cropping
    maxx = 0
    if topRightimg2[0] < botRightimg2[0]:
    	maxx = topRightimg2[0]
    else:
    	maxx = botRightimg2[0]

    # create canvas to place images on
    canvas = np.zeros((maxy-miny, maxx-minx, 3))
    heightcanvas = len(canvas)
    widthcanvas = len(canvas[0])
    channelscanvas = len(canvas[0][0])

    for x in range(minx, maxx):
    	for y in range(miny, maxy):
    		temppoint = [x, y, 1]
    		srcpoint = np.dot(invh, temppoint)
    		srcpointx = srcpoint.item(0) / srcpoint.item(2)
    		srcpointy = srcpoint.item(1) / srcpoint.item(2)
    		srcpointxf = srcpointx
    		srcpointyf = srcpointy
    		srcpointx = round(srcpointx)
    		srcpointy = round(srcpointy)
    		
    		if(srcpointx >= 0 and srcpointx < width1-1 and srcpointy >= 0 and srcpointy < height1-1):
    			for color in range(0, channels1):

    				leftxpercent = (np.ceil(srcpointxf)) - srcpointxf
    				rightxpercent = 1-leftxpercent
    				upperypercent = (np.ceil(srcpointyf)) - srcpointyf
    				lowerypercent = 1-upperypercent

    				topLeftPercent = leftxpercent * upperypercent
    				topRightPercent = rightxpercent * upperypercent
    				botLeftPercent = leftxpercent * lowerypercent
    				botRightPercent = rightxpercent * lowerypercent

	    			tempval = np.array(img_1[srcpointy][srcpointx])
	    			tempvalr = tempval.item(0) * topLeftPercent
	    			tempvalg = tempval.item(1) * topLeftPercent
	    			tempvalb = tempval.item(2) * topLeftPercent

	    			tempval = np.array(img_1[srcpointy][srcpointx+1])
	    			tempvalr += tempval.item(0) * topRightPercent
	    			tempvalg += tempval.item(1) * topRightPercent
	    			tempvalb += tempval.item(2) * topRightPercent

	    			tempval = np.array(img_1[srcpointy+1][srcpointx])
	    			tempvalr += tempval.item(0) * botLeftPercent
	    			tempvalg += tempval.item(1) * botLeftPercent
	    			tempvalb += tempval.item(2) * botLeftPercent

	    			tempval = np.array(img_1[srcpointy+1][srcpointx+1])
	    			tempvalr += tempval.item(0) * botRightPercent
	    			tempvalg += tempval.item(1) * botRightPercent
	    			tempvalb += tempval.item(2) * botRightPercent

	    			finalColor = [tempvalr, tempvalg, tempvalb]

	    			canvas[y+(-miny)][x+(-minx)] = finalColor

    		if(x >= 0 and x < width2 and y >= 0 and y < height2):
    			canvas[y+(-miny)][x+(-minx)] = img_2[y][x]

	    	if(srcpointx >= 0 and srcpointx < width1 and srcpointy >= 0 and srcpointy < height1):
	    		if(x >= 0 and x < width2 and y >= 0 and y < height2):
	    			temp1 = np.array(img_1[srcpointy][srcpointx])
	    			temp2 = np.array(img_2[y][x])
	    			tempr = (temp1.item(0) + temp2.item(0)) / 2
	    			tempg = (temp1.item(1) + temp2.item(1)) / 2
	    			tempb = (temp1.item(2) + temp2.item(2)) / 2
	    			avecolor = [tempr, tempg, tempb]

	    			canvas[y+(-miny)][x+(-minx)] = avecolor

    # ===== blend images: average blending
    # to be completed ...
    #did higher up

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...
    #did higher up
    print('Finished Transformation!')
    img_panorama = canvas
    return img_panorama

def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)
    print('All Done')

    return img_panorama

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))








