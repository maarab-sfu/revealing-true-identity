import skimage
import scipy.spatial as spatial
import logging
import skimage.transform
import face_alignment
import sys
import numpy as np
import cv2
import torch

class Warp():       
    #https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/   
    def _shape_to_array(self, shape, w, h):
        """
        Function to convert shape object to array
        :param shape:
        :return:
        """
        # We need to bound the predicted facial landmarks to be inside the image dimensions. This is because the facial landmarks network can predict points outside the image dimensions.
        return np.asarray(list([max(0,min(p[0],w-1)), max(0,min(h-1,p[1]))] for p in shape[0]), dtype=np.int)

            

    ## 3D Transform
    def bilinear_interpolate(self, img, coords):
        """ Interpolates over every image channel
        http://en.wikipedia.org/wiki/Bilinear_interpolation
        :param img: max 3 channel image
        :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
        :returns: array of interpolated pixels with same shape as coords
        """
        int_coords = np.int32(coords)
        x0, y0 = int_coords
        dx, dy = coords - int_coords
        x0[x0>=img.shape[1]-1]=img.shape[1]-2
        y0[y0>=img.shape[0]-1]=img.shape[0]-2
        # 4 Neighour pixels
        q11 = img[y0, x0]
        q21 = img[y0, x0 + 1]
        q12 = img[y0 + 1, x0]
        q22 = img[y0 + 1, x0 + 1]

        btm = q21.T * dx + q11.T * (1 - dx)
        top = q22.T * dx + q12.T * (1 - dx)
        inter_pixel = top * dy + btm * (1 - dy)

        return inter_pixel.T

    def grid_coordinates(self, points):
        """ x,y grid coordinates within the ROI of supplied points
        :param points: points to generate grid coordinates
        :returns: array of (x, y) coordinates
        """
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0]) + 1
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1]) + 1

        return np.asarray([(x, y) for y in range(ymin, ymax)
                        for x in range(xmin, xmax)], np.uint32)


    def process_warp(self, src_img, result_img, tri_affines, dst_points, delaunay):
        """
        Warp each triangle from the src_image only within the
        ROI of the destination image (points in dst_points).
        """
        roi_coords = self.grid_coordinates(dst_points)
        # indices to vertices. -1 if pixel is not in any triangle
        roi_tri_indices = delaunay.find_simplex(roi_coords)

        for simplex_index in range(len(delaunay.simplices)):
            coords = roi_coords[roi_tri_indices == simplex_index]
            num_coords = len(coords)
            out_coords = np.dot(tri_affines[simplex_index],
                                np.vstack((coords.T, np.ones(num_coords))))
            
            x, y = coords.T
            result_img[y, x] = self.bilinear_interpolate(src_img, out_coords)

        return None


    def triangular_affine_matrices(self, vertices, src_points, dst_points):
        """
        Calculate the affine transformation matrix for each
        triangle (x,y) vertex from dst_points to src_points
        :param vertices: array of triplet indices to corners of triangle
        :param src_points: array of [x, y] points to landmarks for source image
        :param dst_points: array of [x, y] points to landmarks for destination image
        :returns: 2 x 3 affine matrix transformation for a triangle
        """
        ones = [1, 1, 1]
        for tri_indices in vertices:
            src_tri = np.vstack((src_points[tri_indices, :].T, ones))
            dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
            mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
            yield mat


    def warp_image_3d(self, src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
        rows, cols = dst_shape[:2]
        result_img = np.zeros((rows, cols, 3), dtype=dtype)

        delaunay = spatial.Delaunay(dst_points)
        tri_affines = np.asarray(list(self.triangular_affine_matrices(
            delaunay.simplices, src_points, dst_points)))

        self.process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

        return result_img


    ## 2D Transform
    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
        R = (np.dot(U, Vt)).T

        return np.vstack([np.hstack([s2 / s1 * R,
                                    (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                        np.array([[0., 0., 1.]])])


    def warp_image_2d(self, im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                    M[:2],
                    (dshape[1], dshape[0]),
                    dst=output_im,
                    borderMode=cv2.BORDER_TRANSPARENT,
                    flags=cv2.WARP_INVERSE_MAP)

        return output_im


    ## Generate Mask
    def mask_from_points(self, size, points,erode_flag=0):
        

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        if erode_flag:
            radius = 10  # kernel size
            kernel = np.ones((radius, radius), np.uint8)
            mask = cv2.erode(mask, kernel,iterations=1)

        return mask


    ## Color Correction
    def correct_colours(self, im1, im2, landmarks1):
        COLOUR_CORRECT_BLUR_FRAC = 0.6
        LEFT_EYE_POINTS = list(range(42, 48))
        RIGHT_EYE_POINTS = list(range(36, 42))

        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                                np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur = im2_blur.astype(int)
        im2_blur += 128*(im2_blur <= 1)

        result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


    ## Copy-and-paste
    def apply_mask(self, img, mask):
        """ Apply mask to supplied image
        :param img: max 3 channel image
        :param mask: [0-255] values in mask
        :returns: new image with mask applied
        """
        masked_img=cv2.bitwise_and(img,img,mask=mask)

        return masked_img


    ## Alpha blending
    def alpha_feathering(self, src_img, dest_img, img_mask, blur_radius=15):
        mask = cv2.blur(img_mask, (blur_radius, blur_radius))
        mask = mask / 255.0

        result_img = np.empty(src_img.shape, np.uint8)
        for i in range(3):
            result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

        return result_img


    def check_points(self, img,points):
        # Todo: I just consider one situation.
        if points[8,1]>img.shape[0]:
            logging.error("Jaw part out of image")
        else:
            return True
        return False
    def select_face(self, im, points, r=0):   
        points = np.asarray(points)
        LEFT_EYE_POINTS = list(range(42, 48))
        RIGHT_EYE_POINTS = list(range(36, 42))
        new=(np.mean(points[LEFT_EYE_POINTS], axis=0) + np.mean(points[RIGHT_EYE_POINTS], axis=0))//2        
        new[1]=max(0,new[1]-50)
        # Three extra points are added to have a bigger convex hull to include the forehead and eyebrows area because makeup attacks contain a lot of makeup in these areas
        points=np.concatenate((points,[new.astype(int)]))
        points=np.concatenate((points,[[points[0][0],max(0,points[0][1]-50)]]))
        points=np.concatenate((points,[[points[16][0],max(0,points[16][1]-50)]]))
        im_w, im_h = im.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)
        
        x, y = max(0, left-r), max(0, top-r)
        w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

        return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]
        
    def face_warp(self, src_face, dst_face, src_points, dst_points ):                  
        h, w = dst_face.shape[:2]
              
        src_mask = self.mask_from_points(src_face.shape[:2], src_points)
        src_face = self.apply_mask(src_face, src_mask)
        # Correct Color for 2d warp
        # warped_dst_img = self.warp_image_3d(dst_face, dst_points, src_points, src_face.shape[:2])
        # src_face = self.correct_colours(warped_dst_img, src_face, src_points)
        # Warp
        warped_src_face = self.warp_image_2d(src_face, self.transformation_from_points(dst_points, src_points), (h, w, 3))

        ## Mask for blending
        mask = self.mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask*mask_src, dtype=np.uint8)
        r = cv2.boundingRect(mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

        return output


    # # Apply affine transform calculated using srcTri and dstTri to src and
    # # output an image of size.
    # def applyAffineTransform(self, src, srcTri, dstTri, size) :
        
    #     # Given a pair of triangles, find the affine transform.
    #     warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
        
    #     # Apply the Affine Transform just found to the src image
    #     dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    #     return dst


    # # Check if a point is inside a rectangle
    # def rectContains(self, rect, point) :
    #     if point[0] < rect[0] :
    #         return False
    #     elif point[1] < rect[1] :
    #         return False
    #     elif point[0] > rect[0] + rect[2] :
    #         return False
    #     elif point[1] > rect[1] + rect[3] :
    #         return False
    #     return True


    # #calculate delanauy triangle
    # def calculateDelaunayTriangles(self, rect, points):
    #     #create subdiv
    #     subdiv = cv2.Subdiv2D(rect)
    #     # Insert points into subdiv
    #     for p in points:
    #         subdiv.insert(p) 
        
    #     triangleList = subdiv.getTriangleList()
        
    #     delaunayTri = []
        
    #     pt = []    
            
    #     for t in triangleList:        
    #         pt.append((t[0], t[1]))
    #         pt.append((t[2], t[3]))
    #         pt.append((t[4], t[5]))
            
    #         pt1 = (t[0], t[1])
    #         pt2 = (t[2], t[3])
    #         pt3 = (t[4], t[5])        
            
    #         if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
    #             ind = []
    #             #Get face-points (from 68 face detector) by coordinates
    #             for j in range(0, 3):
    #                 for k in range(0, len(points)):                    
    #                     if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
    #                         ind.append(k)    
    #             # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
    #             if len(ind) == 3:                                                
    #                 delaunayTri.append((ind[0], ind[1], ind[2]))
            
    #         pt = []        
                
        
    #     return delaunayTri
            

    # # Warps and alpha blends triangular regions from img1 and img2 to img
    # def warpTriangle(self, img1, img2, t1, t2) :

    #     # Find bounding rectangle for each triangle
    #     r1 = cv2.boundingRect(np.float32([t1]))
    #     r2 = cv2.boundingRect(np.float32([t2]))

    #     # Offset points by left top corner of the respective rectangles
    #     t1Rect = [] 
    #     t2Rect = []
    #     t2RectInt = []

    #     for i in range(0, 3):
    #         t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
    #         t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
    #         t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    #     # Get mask by filling triangle
    #     mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    #     cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    #     # Apply warpImage to small rectangular patches
    #     img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #     #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
        
    #     size = (r2[2], r2[3])

    #     img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        
    #     img2Rect = img2Rect * mask

    #     # Copy triangular region of the rectangular patch to the output image
    #     img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        
    #     img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
        

    # def face_warp(self, img1, img2, points1, points2 ):           
    #     img1Warped = np.copy(img2)        
    #     # Find convex hull
    #     hull1 = []
    #     hull2 = []
    #     hullIndex = cv2.convexHull(np.array(points2,dtype=np.float32), returnPoints = False)            
    #     for i in range(0, len(hullIndex)):
    #         hull1.append(points1[int(hullIndex[i])])
    #         hull2.append(points2[int(hullIndex[i])])
        
        
    #     # Find delanauy traingulation for convex hull points
    #     sizeImg2 = img2.shape    
    #     rect = (0, 0, sizeImg2[1], sizeImg2[0])
        
    #     dt = self.calculateDelaunayTriangles(rect, hull2)
        
    #     if len(dt) == 0:
    #         quit()
        
    #     # Apply affine transformation to Delaunay triangles
    #     for i in range(0, len(dt)):
    #         t1 = []
    #         t2 = []
            
    #         #get points for img1, img2 corresponding to the triangles
    #         for j in range(0, 3):
    #             t1.append(hull1[dt[i][j]])
    #             t2.append(hull2[dt[i][j]])
            
    #         self.warpTriangle(img1, img1Warped, t1, t2)
        
                
    #     # Calculate Mask
    #     hull8U = []
    #     for i in range(0, len(hull2)):
    #         hull8U.append((hull2[i][0], hull2[i][1]))
        
    #     mask = np.zeros(img2.shape, dtype = img2.dtype)  
        
    #     cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
        
    #     r = cv2.boundingRect(np.float32([hull2]))    
        
    #     center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
            
        
    #     # Clone seamlessly.
    #     output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
        
    #     return output 
    