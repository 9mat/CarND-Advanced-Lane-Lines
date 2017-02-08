##Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_chessboards]: ./output_images/undistorted_chessboards.jpg "Undistorted"
[undistorted_roads]: ./output_images/undistorted_imgs.jpg "Road Transformed"
[pipeline]: ./output_images/pipeline.jpg "Binary Example"
[warp]: ./output_images/warped_img.jpg "Warp Example"
[pipeline_warp]: ./output_images/pipline_and_warp.jpg "Fit Visual"
[slidding_windows]: ./output_images/slidding_windows.jpg "Sliding window"
[fitted_lanes]: ./output_images/fitted_lanes.jpg "Fitted lanes"
[output]: ./output_images/output.jpg "Fit Visual"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the ***second code cell*** of the IPython notebook named ***lane-tracking.ipynb***.  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
```python
# Find chessboard corners and append to the list when possible
for i in tqdm(range(len(img_list))):
    img = cv2.imread(img_list[i])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)            
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # if found the corners, add them to the list 
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
    
# Calcualte the undistortion matrix 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None,None)
```
I defined a function to correct for distortion using `cv2.undistort()` with the above calibration results. 
```python
# define a function that will correct for distortion an arbitrary image
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)
```
I applied this distortion correction to the chessboard images and obtained this result: 

![Chessboards][undistorted_chessboards]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
I appled the same distortion correction as above on the road images from the test folder and obtained the results in the following figure.

![Distortion correction roads][undistorted_roads]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate the binary images. The implementation is coded in the function `pipeline()` in the ***7<sup>th</sup> code cell*** of the Jupyter notebook ***lane-tracking.ipynb***.

```python
def pipeline(img, sx_thresh=30, sl_thresh=200, suv_thresh = 100):
    # Apply gaussian blur to reduce noise
    img = gaussian_blur(img)

    # Convert to YUV and use U and V to detect yellow
    y,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    uv = np.abs(u.astype(np.int32) - v.astype(np.int32))*3
    
    # Convert to HSL and use L to detect white
    h,l,s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    l = clahe.apply(l)
    
    # Gradient
    sobelx = np.absolute(cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)) # Take the derivative in x
    
    mask_sx = scaled(sobelx) > sx_thresh
    mask_white = (l>sl_thresh)
    mask_yellow = (uv>suv_thresh)
    
    # Stack all 3 thresholds together
    color_binary = np.dstack((mask_yellow, mask_white, mask_sx))
    return color_binary
```
I used the UV channel from YUV color space, the L channel from HLS color space and the gradient in lightness along the x-direction to create the thresholded binary image.

The above choice of the color space is the result of a long process of trials and errors. The UV channel performs very well at detecting yellow. The L channel is very robust in detecting white. I use these two channels instead of the saturation (S from HLS) suggested in the lecture because, even though S is very good in detecting both white and yellow, it underperforms in detecting yellow compared to UV, and in detecting white compared to L, especially when the road is overexposed to sunlight. The Sobel X transform is used to make sure the pipeline able to detect edges even when there are shadows that may change the shades of yellow and white of the lane marks.

I also apply Gaussian blur to the bottom half of the image to reduce noise, and apply local brightness historam equalization to the lightness channel to enhance constrast. 

The following figure shows the result of the pipeline, together with the output from intermediate steps (color/gradient transformation and thresholding)
![Pipeline][pipeline]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform includes a function called `birdeye()`, which appears in the ***9<sup>th</sup> code cell*** in the file ***lane-tracking.ipynb***.  To define the function `birdeye()`, I precomputed the perspective transform matrix using the  source (`src`) and destination (`dst`) points.  For the project video, I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 32, img_size[1] / 2 + 80],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 32), img_size[1] / 2 + 80]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source         | Destination   | 
|:--------------:|:-------------:| 
|  605, 440      | 320, 0        | 
|  203, 720      | 320, 720      |
| 1127, 720      | 960, 720      |
|  675, 440      | 960, 0        |

I then precomputed the transformation matrices (the computation is in the same code cell) using `cv2.getPerspectiveTransform()` function

```python
M = cv2.getPerspectiveTransform(source_points, warped_points)
Minv = cv2.getPerspectiveTransform(warped_points, source_points)
```
Precomputation avoids me of the need to repeatedly call `cv2.getPerspectiveTransform()` for transformation matrices, as I will assume that the road will be on the same plane throughout the video and hence the perspective transform will not change over time. Using this precomputed transformation, I defined the `birdeye()` function (in the same code cell) that took in a road image and outputed the corresponding bird eye view of the road.
```python
def birdeye(img):
    return cv2.warpPerspective(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][warp]

The following figures show the perspective tranform on the road image togehter with the binary image from the thresholding pipeline.
![alt text][pipeline_warp]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I first identified the potential lane pixels from the thresholded binary image by one of the two following methods. If no prior lane has been detected for the last 20 frames, I will use a series of slidding windows to find the lane pixels, otherwise I will use the pixels within a certain margin from the previous best fitted lanes as the potential lane pixels.

The sliding window method is implemented in the function `find_lane_pixels_with_window_centroids()` the ***14<sup>th</sup> code cell*** in the Jupyter notebook ***lane-tracking.ipynb***. I first looked for the starting point of the lanes by applying convolution over the histogram of the bottom third of the binary image to identify the two peaks on the left and the right sides of the image.

```python
    l_sum = np.sum(binary_warped[top:,:mid], axis=0)
    r_sum = np.sum(binary_warped[top:,mid:], axis=0)

    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+mid
```
I would then iteratively look for the centroids of all points within a window of certain size centering around the previous centroid, and update the new centroids accordingly.

```python
     for level in range(1,img_height//window_height):
       # convolve the window into the vertical slice of the image
        image_layer = np.sum(binary_warped[img_height-(level+1)*window_height:img_height-level*window_height], axis=0)
        conv_signal = np.convolve(window, image_layer)
        
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img_width))
        if np.max(conv_signal[l_min_index:l_max_index]) > 30:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,img_width))
        if np.max(conv_signal[r_min_index:r_max_index]) > 30:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
```

If lanes had been detected in the previous frame, I would use them as reference to simplify the search for lane pixels. This method is implemented in the function `find_lane_pixels_with_previous_fit()`. This function basically looked for all the non-zero pixel in the binary image within a certain margin from the given fitted lines.

After obtaining the potential lane pixels, I used regression to fit a quadratic curve to the pixels. Instead of fitting 2 curves separately, I made use of the prior knowledge that the lanes are almost parallel to each other and estimate them together, constraining the quadratic and the linear coefficients to be them same, only allowing for the intercept to changes between the two curve.

The fitting is implemented in the function `fit_lane()` in the ***15<sup>th</sup> code cell*** of the Jupyter notebook ***lane-tracking.ipynb***.
```python
def fit_lane(lefty, leftx, righty, rightx, leftw, rightw, img_height, img_width):

    def make_feature(y,right):
        yy = y - img_height # recenter y
        polynomial = yy**np.arange(1,3).reshape(-1,1)
        dv_right = right*np.ones_like(yy)
        interaction = dv_right*(yy**2)
        return np.vstack([polynomial, dv_right]).transpose()
    
    features_left = make_feature(lefty, False)
    features_right = make_feature(righty, True)
    
    features = np.vstack([features_left, features_right])
    output = np.hstack([leftx, rightx])
    weight = np.hstack([leftw*lefty, rightw*righty])
        
    if len(features) < 20:
        return None, None, (None, None, None)
    
    regr=linear_model.LinearRegression()
    regr.fit(features, output, weight)
        
    fity = np.arange(img_height)
    
    fit_features_left = make_feature(fity, False)
    fit_features_right = make_feature(fity, True)

    fit_leftx = regr.predict(fit_features_left)
    fit_rightx = regr.predict(fit_features_right)
    
    return regr, make_feature, (fity, fit_leftx, fit_rightx)
```

The result from the slidding window method and from the using previous fitted lines method are shown below
![slidding windows][slidding_windows]
![fit from previous lines][fitted_lanes]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

First, I mannually measured the width of the lanes in pixels, and the number of dash lines and gaps between them in the birdeye view of the road. Base on these numbers I calculated the scaling factor `ym_per_pix = 3*17/720` and `xm_per_pix = 3.7/(960-320)` to convert all coordinates from pixels to meters.

The calculation for the radius of curvation is done in the ***21<sup>st</sup> code cell*** of the Jupyter notebook ***lane-tracking.ipynb***. As the pixels obtained from the top half are often noisy, I only used the bottom part of the image for this calculation.

```python
def radius_of_curvature(fity, fitx):
    # Fit new polynomials to x,y in world space
    y = fity - len(fity)
    x = fitx - fitx.mean()
    a,b,c = np.polyfit(y[-100:]*ym_per_pix, x[-100:]*xm_per_pix, 2)
    return ((1 + b**2)**1.5)/(2*a)
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the ***19<sup>th</sup>*** code cell in the Jupyter notebook "lane-tracking.ipynb" in the function `stitch()`. This function draw the fitted lanes and the lane pixels in the birdeye onto a blank image `color_warp`, then warp it back to the normal view, and finally, overlay it onto the orignal image. The main logic of the function is implemented as follows:

```python
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    result = img.copy()
    
    # draw an arrow to indicate the center of the car
    cv2.arrowedLine(result, (img.shape[1]//2,img.shape[0]), (img.shape[1]//2,img.shape[0]*2//3), [0,200,200], 7,tipLength=0.1)
    result = cv2.addWeighted(result, 0.9, newwarp, 0.6, 0)
    highlighted_birdeye_img = cv2.addWeighted(birdeye_img, 1, color_warp, 0.3, 0)
```
This function also draws the output from the intermediate steps of the pipeline side by side to the final output image. The following figure is an example of the output of this function.
![Output][output]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](http://www.youtube.com/watch?v=F8pqSywS6Iw)

[![Video](http://img.youtube.com/vi/F8pqSywS6Iw/0.jpg)](http://www.youtube.com/watch?v=F8pqSywS6Iw "Video Title")

[Here](http://www.youtube.com/watch?v=vBnTaBC2iJU) is the result on the challenge video

[![Video](http://img.youtube.com/vi/vBnTaBC2iJU/0.jpg)](http://www.youtube.com/watch?v=vBnTaBC2iJU "Video Title")

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are still many cases when the pipeline has difficuties:
* Overexposure to sunlinght make it very difficult for the L-channel to identify white lane
* Grass or leaves on the side with similar shades of yellow can be falsely detected by the UV channel
* Uneven color of the surface can create noises in the gradient, resulting in a lot of false positive detected by Sobel X
* Flares from the front glasses and from other objects can also create false positives in detecting lane pixels
* Upward and downward slopes can change the plane on which the road lie, impairing the accuracy of the perspective transformation
* Other vehicle and objects on the road can obstruct the view to the lanes, as well as creating false positive in terms of edge and color detection

More could be done:
* Adaptive thresholds could be employed to response to different condition in lighting, i.e. threshold to detect white in shadows and/or low-light condition could be set lower than under abundance of sunlight. Histogram equalization could help as well
* More sophisticated logic to combine different thresholds, e.g. under shadows, edge detection seems to work better then color detection
* Detect and eliminate other lines: very often we have other lines that run parallel to the lanes (e.g. other lanes, the edge of the road, a mark of the road), sliding windows some can mistake these lines as the main lanes and bias the curve fitting. tracing these lines can help eliminate them and improve the robustness of the pipeline

