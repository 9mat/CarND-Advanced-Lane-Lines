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
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "lane-tracking.ipynb".  
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
I define a function to correct for distortion using the above calibration results and the function `cv2.undistort()`
```python
# define a function that will correct for distortion an arbitrary image
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)
```
I apply this distortion correction to the test image obtained this result: 

![Chessboards][undistorted_chessboards]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
I apply the same distortion correction above on the road images from the test folder and obtain results in the following figure.
![Distortion correction roads][undistorted_roads]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, The implementation is coded in the function `pipeline()` in the 7th code cell in the Jupyter notebook "lane-tracking.ipynb".

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
I used UV channel from YUV color space, the L channel from HLS color space and the gradient in the x-direction to create the thresholded binary image.

The choice of the color space is the result of a long process of trials and errors. The UV channel performs very well in detecting yellow. The L channel is very robust in detecting white. I use these two channels instead of the saturation (S from HLS) because, even though S is very good in detecting both white and yellow, it underperforms in each of the two colors compared to each of the aforementioned color channel, especially when the road is overexposed to sunlight. The Sobel X transform is used to make sure the pipeline detecting edges even when there are shadows that may change the shades of yellow and white of the lane marks.

I also apply Gaussian blur to the bottom half of the image to reduce noise, and apply local brightness historam equalization to the lightness channel to enhance constrast. 

The following figure show the result of the pipeline, together with the immediate steps (color/gradient transformation and thresholding)
![Pipeline][pipeline]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birdeye()`, which appears in the 9th code ceell in the file `lane-tracking.ipynb`.  To define the function `birdeye`, I precomputed the perstective transformation matrix using the  source (`src`) and destination (`dst`) points.  For the project video, I chose the hardcode the source and destination points in the following manner:

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
Precomputation avaoid repeated call to the `cv2.getPerspectiveTransform()`, as I will assume that the road is will be the same plane throughout the video and hence the perspective transformation will not change over time. Using this precomputed transformation, I define the `birdeye()` function (in the same code cell) that take in a road image and output the corresponding bird eye view of the road.
```python
def birdeye(img):
    return cv2.warpPerspective(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][warp]

![alt text][pipeline_warp]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

