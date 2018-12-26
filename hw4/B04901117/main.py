import numpy as np
import cv2
import time, math


def computeDisp(Il, Ir, max_disp):
    # h, w, ch = Il.shape
    # labels = np.zeros((h, w), dtype=np.uint8)

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    MAX_DISPARITY = math.ceil(max_disp/16)*16
    window_size = 7
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=MAX_DISPARITY, # max_disp has to be dividable by 16
        blockSize=5,
        P1=7 * 3 * window_size ** 2,   # disparity smoothnesss terms
        P2=30 * 3 * window_size ** 2,  # disparity smoothnesss terms, 27*3 yields 18.03
        disp12MaxDiff=1, # Max allowed difference (in integer pixel units) in the left-right disparity check
        uniquenessRatio=4,
        speckleWindowSize=0,
        speckleRange=0,
        preFilterCap=40,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # opencv automatically scales disparity up by 16
    displ = left_matcher.compute(Il, Ir) / 16
    dispr = right_matcher.compute(Ir, Il) / 16

    toc = time.time()
    print('* Elapsed time (cost computation + cost aggregation + disparity optimization): %f sec.' % (toc - tic))

    # # >>> Cost aggregation
    # tic = time.time()
    # # TODO: Refine cost by aggregate nearby costs
    # toc = time.time()
    # print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # # >>> Disparity optimization
    # tic = time.time()
    # # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    # toc = time.time()
    # print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    occluded = np.zeros(displ.shape)
    for i in range(len(displ)):
        for j in range(len(displ[i])):
            left_disp = int(displ[i, j])
            if left_disp < 0 or (j-left_disp) < 0:
                occluded[i,j] = 1
            else:
                right_disp = -int(dispr[i, j-left_disp])
                if abs(right_disp - left_disp) > 0:
                    occluded[i,j] = 1

    right_offset = 50
    displ_fix_occlusion = displ.copy()

    for i in range(len(displ_fix_occlusion)):
        for j in range(len(displ_fix_occlusion[i]) - right_offset):
            if occluded[i,j] == 1:
                
                min_dist_defined = False
                nearest_disp = 0
                
                for x in range(j-1,-1,-1): # going left
                    if occluded[i,x] == 0:
                        min_dist = j-x # min_dist not defined yet
                        min_dist_defined = True
                        nearest_disp = displ_fix_occlusion[i, x]
                        break
                        
                for x in range(j+1, len(displ_fix_occlusion[i])): # going right
                    if occluded[i,x] == 0:
                        if not min_dist_defined or (x-j) < min_dist:
                            nearest_disp = displ_fix_occlusion[i, x]
                            break
                displ_fix_occlusion[i,j] = nearest_disp

    labels = cv2.medianBlur(np.float32(displ_fix_occlusion),5)

    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', (labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', (labels * scale_factor))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', (labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', (labels * scale_factor))


if __name__ == '__main__':
    main()
