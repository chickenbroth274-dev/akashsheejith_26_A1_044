# akashsheejith_26_A1_044
---27/08/26 THURSDAY---
- started w/ task 2
- installed opencv, numpy, matplotlib
- tried to understand the image processing pipeline
- looked up the math behind the processes(grayscale, canny,mask, hough transform) on youtube
- bitwise_and, bitwise_or,blending and how mask is applied

---28/08/26 FRIDAY---
- straight lines formed but multiple short lines cover the lanes
- use average slope and intercept on left and right side to plot 2 straight lines(left and right)
- switch from average to median because better results
- need to learn how to implement a dynamic mask because a fixed mask sometimes yield unwanted results

---29/08/26 SATURDAY---
- started w/ task 3
- image processing pipeline is somewhat the same
- hsv conversion also done this time to separate coloured objects
- white mask used for lanes and potholes
- potholes are separated from lanes using aspect ratio of drawn ellipses
