## The algorithm is located in src/my_package/my_package/ named 'gap_finder_base.py'

# Algorithm Explanation
#### The algorithm takes in a list of ranges from a LiDAR scan and returns a twist dictionary (containing speed and steering value) that will move the car to the desired gap in the scan after drawing safety bubbles.

#### Instead of always aiming for the deepest gap, the car will find the gap start from 65% of the maximum distance so that it doesn't clip the wall. The threshold ensures the car not makes decision solely based on the max distance, as smaller distance might be more central and reachable.

#### The algorithm ensures efficiency in identifying the obstacles by capping the range value to limit how far the car can look and by limiting the field of view.

#### The algorithm calculates the front clearance distance (the distance of obstacles directly ahead of the vehicle) measured by the middle lidar scan to determine the required speed. The centre priority mask prioritises the centre of the field of view by assigning higher weightage. Thus, the car will give more importance to obstacles or gaps directly in front when determining speed and steering value.

#### It also identifies the disparity and the closest obstacle around to avoid collision. An unsafety region around them will be set and the modified ranges detected by the lidar points within these regions are set to the minimum distance to the obstacle. By this, it can focus on the most critical part and maintain a safety buffer against obstacles.

#### The car might become unstable due to the frequent small changes in speed.  The algorithm ensures the car speed is one of the 7 bin speeds (Evenly divided speed within max and min speed interval).
