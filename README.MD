# Calculate vanishing point
## Assignment explanation
For this assignment I had to find the dominant straight lines in the image. 
I did this using the cv2 method _HoughLines_ which was fed with edges found 
using canny edge detection. 

Based on these lines I calculated intersections between all lines. 
On these intersections I placed a gausian kernel to deal with the noise 
of not all lines going exactly to the same point. 

From this gausian map I 
found values with a value in the top 70 percent of values which in my opinion 
will probably include the vanishing points as they should be an overlap of many 
gausian (near) centers. 

On this thresholded mask I performed an erosion to get rid of small peaks which 
are probably not vanishing points but just noise and dilation to close the gabs 
in the blobs created by erosion. 

The vanishing points are the centers of the blobs that are left. And the lines 
that go through it are filtered and shown on the image together with the 
vanishing point.

## Is it perfect?
The code I present here is far from perfect and there are numerous things I 
would want to do differently. However I don't have a lot of time left in the 
coming week and think that this gives a good representation of my knowledge and
abilities in this subfield of computer vision, opencv numpy and python. 

## What could I have done better
- more tests, I wanted to test everything but simply did not have enough time. 
I tested the in my opinion most important or error prone functions.
- I don't like the way in which a Line/intersection is represented in image space. 
This makes it hard to compare with vanishing points that are represented in the 
extended (3x width of image and 3x height of image) image space. It also makes 
it less readable and more error prone because I had to add the original image 
dimension to a point on a line or intersection. 
- I don't like the extended image. There should be a better way to draw outside
of the original image and extend with white pixels up to that point, I could 
however not think of one. 
- Comparing all lines with all other lines to find intersections is a really 
compute heavy way of finding the required intersections. It can be improved a 
lot by only comparing lines in the same type with each other. So first cluster 
lines that have similar orientation and are convergent. 

## What do I like
- I like the gausian at every intersection idea, it gives a modular way to deal
with uncertainty and gives an interesting mask image that also holds some 
interesting information. It also should automatically find multiple vanishing 
points.
- Using the distance between the point and the ends of the line to see if 
a line goes through a point
- reducing the number of lines drastically by removing all non unique 
orientations. This can of course be a problem when you have two identically 
oriented lines that converge towards a different vanishing point. But the 
chance of having exactly the same orientation seems to be small when you 
compare it with the large reduction of lines you have to check. 


## Running the code
Code can be run using the main method or by running main.py in a pipenv environment. 
It will after a while display the first image with found vanishing point and 
lines passing through it and after pressing a button do the same for the second 
image. After pressing a button again it will print the vanishing points.