# ShadowDetection
Implementation of paper **"Illumination-Aware Image Segmentation for Real-Time Moving Cast Shadow Suppression"**.
2022 IEEE International Conference on Imaging Systems and Techniques (IST)

### Introduction
One of the main challenges facing foreground detection methods is the performance deterioration due to shadows cast by moving objects. In this paper, a new real-time method is proposed that integrates various cues for region-wise classification to deal with achromaticity and camouflage issues in suppressing cast shadows. Specifically, after background subtraction, a locally near-invariant illumination feature is used as input for watershed segmentation approach to extract a number of superpixels. The superpixels are further merged according to three illumination criteria with the purpose of constructing segments that are locally homogeneous in terms of illumination variations. These segments are then classified according to the number of potential shadow candidates, gradient direction correlation, and the number of external boundary points. The potential shadow candidates are extracted by establishing a set of chromatic criteria in the HSV color-space. The gradient correlation is considered due to the fact that shadows do not impose considerable variations in the gradient directions. On the other hand, shadow segments contain a notable number of extrinsic boundary points which is used as an additional cue. Final shadow detection is achieved by integrating the outputs of the previous steps. The experimental results using publicly available videos from ATON dataset show the feasibility of our proposed method for real-time applications.


Pixel-wise approaches fail to differentiate between shadows and dark objects that have similar color value as they are limited only to the variations in the RGB values and do not take the spatial relations among each pixel and its neighborhood into account.
Therefore, a combination of pixel-based and region-based techniques can help with locating the dark objects and reduce the misclassification errors.

<a href="url"><img src="https://user-images.githubusercontent.com/24352869/188034875-2e4bfa27-616f-4c63-a737-d6f0a2b1259b.png" align="left" width=50% height=50% ></a>

### Segmentation

We apply the watershed segmentation approach on the spectral ratios of each region in $\mathbb{R}$ to obtain the superpixels.
Afterward, correlated superpixels are merged by applying the union-find algorithm.
Due to the ratio-invariance property of shadows, two neighboring superpixels are merged if their spectral ratio differences are less than a small threshold across all three color channels.
In addition, the edge between two superpixel may have been caused by intersecting shadows, which are difference-invariant.
Therefore, two neighboring segments are merged if the difference between the foreground values is close to the difference between their background values.
Another possible scenario is if the moving shadow is cast over an existing stationary shadow.
In this case, the background values are different, but the foreground values are similar and close to the background value of the darker segment.

<a href="url"><img src="https://user-images.githubusercontent.com/24352869/188035043-20a59946-d784-48aa-9e18-2e3ea22deed8.png" align="left" width=50% height=50% ></a>

### Candidate shadows
Since the HSV color-space separates the chromaticity from the intensity to a good level and is useful to distinguish the variations in illumination from the changes in material.
Below figure illustrates the potential shadow zone in the RGB color-space which is a portion of the conic region in the RGB space.
Since shadows have little to no effect on the H(hue) component of the HSV color-space, we choose the S(saturation) and V(value) components to set the criteria.
The value ratio can roughly specify the attenuation which is represented by the vector magnitudes and the saturation component can determine the apex angle of the cone which depends on the ambient illumination.

<a href="url"><img src="https://user-images.githubusercontent.com/24352869/188035082-960280cd-0ead-49be-8100-0cdf63d1582d.png" align="left" width=50% height=50% ></a>

### Method overview
In this paper, a real-time method is proposed to detect and suppress moving shadows with minimal manual involution.
First, the global foreground modeling (GFM) method is applied for foreground segmentation due to its efficiency and robustness.
Therefore, we employ a region-based classification method, which is capable of dealing with achromaticity and camouflage issues.
The watershed segmentation approach is applied in order to extract superpixels.
A locally near-invariant illumination feature is applied to merge correlated superpixels and segment the foreground into a number of regions.
These regions are then classified based on the number of candidate shadow samples, foreground-background gradient direction correlation, and the number of external terminal points.
At the end, the results of all the three steps are integrated for final shadow detection.
This integration results in an accurate and robust shadow detection method for real-time video analytics applications.

### System architecture of the shadow detection method
![flowchart](https://user-images.githubusercontent.com/24352869/184900962-b08a6e41-edf5-402c-81e9-f309e24844c0.png)


### Comparisons
![WeChat Screenshot_20220816101306](https://user-images.githubusercontent.com/24352869/184901545-06cce1b6-e0d9-4676-bd40-4c3f0d7b8e52.png)

### Run time
![image](https://user-images.githubusercontent.com/24352869/188035172-60a7c4c0-8f73-4f01-a393-4b35b93732a6.png)


### Environment
- OpenCV 3.4.1
- Visual Studio 19.0


### Citation
```
@inproceedings{ghahremannezhad2022illumination,
  title={Illumination-Aware Image Segmentation for Real-Time Moving Cast Shadow Suppression},
  author={Ghahremannezhad, Hadi and Shi, Hang and Liu, Chengjun},
  booktitle={2022 IEEE International Conference on Imaging Systems and Techniques (IST)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}
```

