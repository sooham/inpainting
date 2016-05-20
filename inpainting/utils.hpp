#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

struct gradients {
    cv::Mat dx;
    cv::Mat dy;
};

// the radius of a patch
// results in 9x9 patches
int radius = 4;

// the radius of pixels around specified point in border
// TODO: should be MACROS?
int border_radius = 2;


