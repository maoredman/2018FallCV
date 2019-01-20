# 2018 Fall Computer Vision course

## HW1: improving grayscale conversion

First, I use the original RGB image to calculate its [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter), then run the image through the filter to get a *"bilateral-filtered image"*.

I also convert the original RGB image to grayscale using a random set of coefficients (for example, the conventional set of coefficients are: 0.299 * R + 0.587 * G + 0.114 * B), getting a *"grayscale candidate"*. This candidate can then be used to calculate its own bilateral filter, but instead of running the filter on itself, I run it on the original RGB image, getting a *"joint bilateral-filtered image"*.

Because I can generate many *"grayscale candidates"*, I can also generate many different bilateral filters from them, hence I can get many *"joint bilateral-filtered images"*. I then see which *"joint bilateral-filtered image"* has the lowest difference. The *"grayscale candidate"* used to generate it is presumed to be the best grayscale converson of the original RGB image.

**Everything was implemented using only `numpy`; no `opencv` filtering functions were used**

Below are some results:

![](https://i.imgur.com/XZ3AboI.png)
![](https://i.imgur.com/cKg1B98.png)

## HW2: image dimension reduction

I implemented PCA (principal component analysis) and LDA (linear discriminant analysis) using `numpy` to perform dimension reduction on images of faces, then used KNN to build a facial recognition classifier.

PCA and LDA were both able to group together faces of the same person very well, probably because of the small dataset size. Here is the dimension reduction result of LDA (reduced to 30 dimensionsby LDA, then further reduced to 2 dimensions by t-SNE for visualization):

![](https://i.imgur.com/GdWdEGl.png)

The KNN classifier was very accurate, again probably because of the small dataset size:

![](https://i.imgur.com/hn9DFJH.png)

## HW3: homography

I used homography transformations (calculated using `numpy`) to warp images. For example, I warped an unreadable QR code like this...

![](https://i.imgur.com/J14t9IU.jpg)

into something readable like this!

![](https://i.imgur.com/OkNg6Ob.png)

## HW4: stereo matching

I used the `StereoSGBM` algorithm of `opencv` along with post-processing techniques such as left-right consistency check and median filtering to generate a depth map from two camera images

![](https://i.imgur.com/9oD6FTQ.png)

![](https://i.imgur.com/JiQcqIz.png)
![](https://i.imgur.com/hJWZsxn.png)
