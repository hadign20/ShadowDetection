#pragma once
#ifndef _MOVINGSHADREM_
#define _MOVINGSHADREM_

#include <iostream>
#include <opencv2/opencv.hpp>


//-- parameters
#define  SEG_THRESH			5					//-- diff of two pixels to merge (smaller means less merging)
#define  REGION_SIZE		2					//-- block size
#define  minSegArea			2					//-- min segment area
#define  minSetArea			30					//-- min set area
#define  OnelumRatioLow		30					//-- min of fg/bg for shadow
#define  OnelumRatioHigh	140					//-- max of fg/bg for shadow
#define  gradThresh			.001				//-- gradient threshold (larger means more shadow)
#define  tpThresh			.0001				//-- terminal point threshold (smaller means more shadow)
#define	 gmmDist			3.0					//-- number of standard deviation distance in global modeling (larger means more shadow)

#define  dilationSize		2
#define  winSize			3					//-- window size for noise correction
#define  stepSize			REGION_SIZE / 2		//-- step for merge
#define  FRINGE				true				//-- shadow outline
#define  COMPACT			false				//-- regular vs compact watershed
#define  SIMPLE				true				//-- regular vs compact watershed


class MovingShadRem
{
public:
	MovingShadRem(cv::Mat& frame);
	~MovingShadRem();

	void removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& noShadowMask);

	//-- debug
	cv::Mat l1, l2, l3, l4, finalShadMask, ratioRep, edgeHeatMap, tpHeatMap, watershedImage, watershedMerged, ratioMean;
	double avgTime = 0.0, frameTime = 0.0, preTime = 0.0, segTime = 0.0, gradTime = 0.0, termTime = 0.0, gmmTime = 0.0, postTime = 0.0;

private:
	int frameNum = 0;
	long numSeg = 0;
	int numSet = 0;

	struct Set {
		int parent = 0;
		int rank = 0;
	};


	struct Gaussian {
		float sigma[9];
		float mu[9];
		float weight;
		int n;
	};

	Gaussian* sh;
	int cov0 = 36, cov_low = 4, cov_hi = 2000;
	float alphaM = 0.001;


	void divideMats(const cv::Mat& first, const cv::Mat& second, const cv::Mat& mask, int scale, cv::Mat& result);
	void shadowsDontCauseEdge(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, const cv::Mat& segments, cv::Mat& edgeMask);
	void terminalPointCal(const cv::Mat& fgMask, const cv::Mat& segments, cv::Mat& tpMask);
	void shadowModelGlobal(cv::Mat& l1, cv::Mat& l2, cv::Mat& l3, const cv::Mat& Frame, const cv::Mat& bg, const cv::Mat& fgMask, cv::Mat& l4);
	void whiteAndGray(const cv::Mat& fgMask, const cv::Mat& whiteMask, cv::Mat& whiteAndGrayMask);
	void watershedSeg(const cv::Mat& frame, const cv::Mat& bg, const cv::Mat& fgMask, cv::Mat& lumRatio, int region_size, cv::Mat& marks);
	void mergeSegments(const cv::Mat& frame, const cv::Mat& bg, const cv::Mat& fgMask, const cv::Mat& lumRatio, int region_size, const cv::Mat& marks, cv::Mat& segments);
	void noiseCorrection(const cv::Mat& fgMask, cv::Mat& shadowMask, int windowSize, cv::Mat& wAndG);
	int find(std::vector<Set>&, int i);
	void Union(std::vector<Set>&, int xroot, int yroot);
};

#endif //_MOVINGSHADREM_

