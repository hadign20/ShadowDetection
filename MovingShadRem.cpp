#include "MovingShadRem.h"
#include <windows.h>
#include <iomanip>
#include <vector>
#include <math.h>
#include <unordered_map>

#define DEBUG		1
#define TIME_DEBUG	1

void watershedEx(cv::Mat& src, cv::Mat& dst);
void compact_watershed(cv::Mat& img, cv::Mat& B, float dy, float dx, float compValStep, cv::Mat& seeds);
void computeLabelsFromBoundaries(const cv::Mat& image, const cv::Mat& boundaries, cv::Mat& labels, int BOUNDARY_VALUE, int INNER_VALUE);
int relabelConnectedSuperpixels(cv::Mat& labels);
void drawContours1(const cv::Mat& image, const cv::Mat& labels, cv::Mat& contours, bool eight_connected);

//===========================================================
//	timer
//===========================================================
double PCFreq = 0.0;
__int64 CounterStart = 0;
double duration = 0.0;

void startTimer() {
	duration = 0.0;

	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";
	//PCFreq = double(li.QuadPart); //-- seconds
	PCFreq = double(li.QuadPart) / 1000.0; //-- milliseconds
	//PCFreq = double(li.QuadPart) / 1000000.0; //-- microseconds
	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

void stopTimer() {
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	duration = (li.QuadPart - CounterStart) / PCFreq;
	//std::cout << "\nduration time (ms): " << duration << "\r";
}


//===========================================================
//	constructor
//===========================================================
MovingShadRem::MovingShadRem(cv::Mat& frame) {
	//-- debug
	l1 = cv::Mat::zeros(frame.size(), CV_8UC1);
	l2 = cv::Mat::zeros(frame.size(), CV_8UC1);
	l3 = cv::Mat::zeros(frame.size(), CV_8UC1);
	l4 = cv::Mat::zeros(frame.size(), CV_8UC1);
	edgeHeatMap = cv::Mat::zeros(frame.size(), CV_8UC3);
	tpHeatMap = cv::Mat::zeros(frame.size(), CV_8UC3);
	watershedImage = cv::Mat::zeros(frame.size(), CV_8UC3);
	watershedMerged = cv::Mat::zeros(frame.size(), CV_8UC3);
	ratioMean = cv::Mat::zeros(frame.size(), CV_8UC3);

	//-- global shadow model
	sh = new Gaussian;
	for (int c = 0; c < 3; c++) {
		sh->mu[c] = 0;
		sh->sigma[c] = 0;
	}
}

//===========================================================
//	destructor
//===========================================================
MovingShadRem::~MovingShadRem() {}


//===========================================================
//	function to remove moving cast shadows
//===========================================================
/**
 * @brief removeShadows: remove moving cast shadows from video
 *
 * @param frame: the current frame
 * @param fgMask: foreground mask image from any BS model
 * @param bg: the background image
 * @param noShadowMask: the final result mask image
 */
void MovingShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& msrMask) {
	cv::Mat marks, segments, colorCriteriaMask, edgeMask, tpMask, gmmMask;
	
#if TIME_DEBUG
	frameNum++;
	frameTime = 0.0;
#endif

	//-------------------------------------------------
	//-- preprocessing
	//-------------------------------------------------
#if TIME_DEBUG
	startTimer();
#endif

	//cv::medianBlur(frame, frame, 5);

#if FRINGE
	//-- remove fg blob contours
	cv::Mat fgMaskEdges, fgMaskEdgesDilated;
	Canny(fgMask, fgMaskEdges, 0, 128);
	dilate(fgMaskEdges, fgMaskEdgesDilated, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
	subtract(fgMask, fgMaskEdgesDilated, fgMask);
#endif

#if TIME_DEBUG
	stopTimer();
	preTime = preTime + (duration - preTime) / (double)(frameNum + 1);
	frameTime += duration;
#endif

	//-------------------------------------------------
	//-- calculate luminance ratio
	//-------------------------------------------------
	cv::Mat lumRatio(frame.size(), frame.type(), 0.0); //-- fg/bg
	divideMats(frame, bg, fgMask, 255, lumRatio);


	//-------------------------------------------------
	//-- segmentation
	//-------------------------------------------------
#if TIME_DEBUG
	startTimer();
#endif

	watershedSeg(frame, bg, fgMask, lumRatio, REGION_SIZE, marks);
	mergeSegments(frame, bg, fgMask, lumRatio, REGION_SIZE, marks, segments);

#if TIME_DEBUG
	stopTimer();
	segTime = segTime + (duration - segTime) / (double)(frameNum + 1);
	frameTime += duration;
#endif

	//-------------------------------------------------
	//-- candidate shadows
	//-------------------------------------------------
	cv::inRange(lumRatio, cv::Scalar(OnelumRatioLow, OnelumRatioLow, OnelumRatioLow), cv::Scalar(OnelumRatioHigh, OnelumRatioHigh, OnelumRatioHigh), colorCriteriaMask);
	//cv::inRange(lumRatio, cv::Scalar(BlumRatioLow, GlumRatioLow, RlumRatioLow), cv::Scalar(BlumRatioHigh, GlumRatioHigh, RlumRatioHigh), colorCriteriaMask);

	//-------------------------------------------------
	//-- gradient
	//-------------------------------------------------
#if TIME_DEBUG
	startTimer();
#endif

	shadowsDontCauseEdge(frame, fgMask, bg, segments, edgeMask);

#if TIME_DEBUG
	stopTimer();
	gradTime = gradTime + (duration - gradTime) / (double)(frameNum + 1);
	frameTime += duration;
#endif

	//-------------------------------------------------
	//-- terminal points
	//-------------------------------------------------
#if TIME_DEBUG
	startTimer();
#endif

	terminalPointCal(fgMask, segments, tpMask);

#if TIME_DEBUG
	stopTimer();
	termTime = termTime + (duration - termTime) / (double)(frameNum + 1);
	frameTime += duration;
#endif

	//-------------------------------------------------
	//-- statistical modeling
	//-------------------------------------------------
#if TIME_DEBUG
	startTimer();
#endif

	shadowModelGlobal(colorCriteriaMask, edgeMask, colorCriteriaMask, lumRatio, bg, fgMask, gmmMask);

#if TIME_DEBUG
	stopTimer();
	gmmTime = gmmTime + (duration - gmmTime) / (double)(frameNum + 1);
	frameTime += duration;
#endif
	

	//-------------------------------------------------
	//-- final shadow detection
	//-------------------------------------------------
	//cv::Mat shadowMask = colorCriteriaMask;
	//cv::Mat shadowMask = colorCriteriaMask & edgeMask;
	//cv::Mat shadowMask = colorCriteriaMask & gmmMask;
	cv::Mat shadowMask = colorCriteriaMask & tpMask;
	//cv::Mat shadowMask = colorCriteriaMask & gmmMask & tpMask;
	//cv::Mat shadowMask = colorCriteriaMask & edgeMask & gmmMask;
	//cv::Mat shadowMask = colorCriteriaMask & edgeMask & tpMask & gmmMask;
	
	msrMask = fgMask - shadowMask;

	//-------------------------------------------------
	//-- post processing
	//-------------------------------------------------


#if TIME_DEBUG
	startTimer();
#endif
	
	noiseCorrection(fgMask, shadowMask, winSize, finalShadMask);

#if TIME_DEBUG
	stopTimer();
	postTime = postTime + (duration - postTime) / (double)(frameNum + 1);
	frameTime += duration;
#endif
	



	
	//-------------------------------------------------
	//-- imshow
	//-------------------------------------------------
#if DEBUG
	cv::imshow("lumRatio", lumRatio);
	lumRatio.copyTo(ratioRep);

	whiteAndGray(fgMask, colorCriteriaMask, l1);
	cv::imshow("colorCriteriaMask", l1);

	whiteAndGray(fgMask, edgeMask, l2);
	cv::imshow("edgeMask", l2);

	whiteAndGray(fgMask, tpMask, l3);
	cv::imshow("tpMask", l3);

	whiteAndGray(fgMask, gmmMask, l4);
	cv::imshow("gmmMask", l4);

	cv::imshow("finalShadMask", finalShadMask);
#endif

#if TIME_DEBUG
	std::cout << "frame time (ms): " << frameTime << "\r";
	avgTime = avgTime + (frameTime - avgTime) / (double)(frameNum + 1);
#endif
}


//===========================================================
//	core functions
//===========================================================
void MovingShadRem::divideMats(const cv::Mat& first, const cv::Mat& second, const cv::Mat& mask, int scale, cv::Mat& result) {
	int ch = first.channels();
	result = cv::Mat(first.size(), first.type(), 0.0);
	for (int y = 0; y < first.rows; y++) {
		uchar const* first_ptr = first.ptr<uchar>(y);
		uchar const* second_ptr = second.ptr<uchar>(y);
		uchar const* mask_ptr = mask.ptr<uchar>(y);
		uchar* result_ptr = result.ptr<uchar>(y);
		for (int x = 0; x < first.cols; x++) {
			if (mask_ptr[x]) {
				for (int c = 0; c < ch; c++) {
					*(result_ptr + ch * x + c) = *(second_ptr + ch * x + c) == 0 ? 0 : *(first_ptr + ch * x + c) * scale / (float) * (second_ptr + ch * x + c);
				}
			}
		}
	}
}


void MovingShadRem::shadowsDontCauseEdge(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, const cv::Mat& segments, cv::Mat& corrMask) {
	
	corrMask = cv::Mat(frame.size(), CV_8UC1, 0.0);
	if (numSet < 1) return;

	cv::Mat bgGray, bgEdges, MO, MOGray, MOEdges, fgMaskEdges, fgMaskEdgesDilated;
	double low_th = 0.0, high_th = 0.0;
	cv::Mat edgeMask(frame.size(), CV_8UC1, 0.0);

	//-- bg edges
	cvtColor(bg, bgGray, CV_RGB2GRAY);
	Canny(bgGray, bgEdges, 200, 255);
	dilate(bgEdges, bgEdges, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

	//-- fg edges
	frame.copyTo(MO, fgMask);
	cvtColor(MO, MOGray, CV_RGB2GRAY);
	Canny(MOGray, MOEdges, 100, 150);

	//-- fg mask edges (contour)
	Canny(fgMask, fgMaskEdges, 0, 128);
	dilate(fgMaskEdges, fgMaskEdgesDilated, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

	//-- Es
	edgeMask = MOEdges - (fgMaskEdgesDilated + bgEdges);
	dilate(edgeMask, edgeMask, cv::Mat(), cv::Point(-1, -1), 6, 1, 1);

//
//#if SIMPLE
//	corrMask = fgMask - edgeMask;
//	return;
//#endif

	//-------------------------------------------------
	std::vector<int> nGrad(numSet + 1, 0);
	std::vector<int> areaVec(numSet + 1, 0);
	std::vector<float> probVec(numSet + 1, 0);

	for (int y = 0; y < frame.rows; y++) {
		int const* seg_ptr = segments.ptr<int>(y);
		uchar const* ed_ptr = edgeMask.ptr<uchar>(y);

		for (int x = 0; x < frame.cols; x++) {
			if (*(seg_ptr + x)) {
				if ((int)*(ed_ptr + x) == 255) {
					++nGrad[*(seg_ptr + x)];
				}
				++areaVec[*(seg_ptr + x)];
			}
		}
	}

	for (int i = 1; i <= numSet; i++) 
		//if (areaVec[i] > minSetArea)
			probVec[i] = (float)nGrad[i] / (1 + areaVec[i]);


	
	for (int y = 0; y < frame.rows; y++) {
		uchar const* mask_ptr = fgMask.ptr<uchar>(y);
		int const* seg_ptr = segments.ptr<int>(y);
		uchar* c_ptr = corrMask.ptr<uchar>(y);
		for (int x = 0; x < frame.cols; x++) {
			if(*(mask_ptr + x) != 0 && probVec[*(seg_ptr + x)] < gradThresh)
				*(c_ptr + x) = 255;
		}
	}

	
#if DEBUG
	cv::Mat probMap(frame.rows, frame.cols, CV_8UC1, 0.0);
	for (int y = 0; y < frame.rows; y++) {
		int const* seg_ptr = segments.ptr<int>(y);
		uchar* p_ptr = probMap.ptr<uchar>(y);
		for (int x = 0; x < frame.cols; x++) {
			*(p_ptr + x) = (int)(probVec[*(seg_ptr + x)] * 255);
		}
	}

	cv::applyColorMap(probMap, edgeHeatMap, cv::COLORMAP_JET);
	edgeHeatMap.setTo(cv::Scalar(0, 0, 0), cv::Mat(fgMask == 0));
	cv::imshow("edgeHeatMap", edgeHeatMap);
	//cv::imshow("edgeProbMap", probMap);
#endif
	
}




/*
* calculate the number of external terminal pixels and all terminal pixels
* */
void MovingShadRem::terminalPointCal(const cv::Mat& fgMask, const cv::Mat& segments, cv::Mat& tpMask) {
	tpMask = cv::Mat(fgMask.size(), CV_8UC1, 0.0);
	std::vector<int> nAllBorders(numSet + 1, 0);
	std::vector<int> nExtBorders(numSet + 1, 0);
	std::vector<float> probVec(numSet + 1, 0);

	for (int y = 1; y < fgMask.rows - 1; y++) {
		int const* seg_ptr = segments.ptr<int>(y);
		int const* seg_ptr_pre = segments.ptr<int>(y - 1);
		int const* seg_ptr_post = segments.ptr<int>(y + 1);
		uchar const* mask_ptr = fgMask.ptr<uchar>(y);
		uchar const* mask_ptr_pre = fgMask.ptr<uchar>(y - 1);
		uchar const* mask_ptr_post = fgMask.ptr<uchar>(y + 1);

		for (int x = 1; x < fgMask.cols - 1; x++) {
			if ((int) * (mask_ptr + x) == 0)
				continue;
			if (*(seg_ptr + x) != *(seg_ptr + x - 1) ||
				*(seg_ptr + x) != *(seg_ptr + x + 1) ||
				*(seg_ptr + x) != *(seg_ptr_pre + x) ||
				*(seg_ptr + x) != *(seg_ptr_post + x) ||
				*(seg_ptr + x) != *(seg_ptr_pre + x - 1) ||
				*(seg_ptr + x) != *(seg_ptr_pre + x + 1) ||
				*(seg_ptr + x) != *(seg_ptr_post + x - 1) ||
				*(seg_ptr + x) != *(seg_ptr_post + x + 1)) {
				nAllBorders[*(seg_ptr + x)]++;
				if (*(mask_ptr + x) != *(mask_ptr + x - 1) ||
					*(mask_ptr + x) != *(mask_ptr + x + 1) ||
					*(mask_ptr + x) != *(mask_ptr_pre + x) ||
					*(mask_ptr + x) != *(mask_ptr_post + x) ||
					*(mask_ptr + x) != *(mask_ptr_pre + x - 1) ||
					*(mask_ptr + x) != *(mask_ptr_pre + x + 1) ||
					*(mask_ptr + x) != *(mask_ptr_post + x - 1) ||
					*(mask_ptr + x) != *(mask_ptr_post + x + 1)) {
					nExtBorders[*(seg_ptr + x)]++;
				}
			}
		}
	}

	for (int i = 1; i <= numSet; i++)
		probVec[i] = (float)nExtBorders[i] / (1 + nAllBorders[i]);

	for (int y = 0; y < fgMask.rows; y++) {
		uchar const* mask_ptr = fgMask.ptr<uchar>(y);
		int const* seg_ptr = segments.ptr<int>(y);
		uchar* tp_ptr = tpMask.ptr<uchar>(y);
		for (int x = 0; x < fgMask.cols; x++) {
			if (*(mask_ptr + x) != 0 && nAllBorders[*(seg_ptr + x)] != 0 && nExtBorders[*(seg_ptr + x)] != 0) {
				if (probVec[*(seg_ptr + x)] > tpThresh)
					*(tp_ptr + x) = 255;
			}
		}
	}

#if DEBUG
	cv::Mat probMap(fgMask.rows, fgMask.cols, CV_8UC1, 0.0);
	for (int y = 0; y < fgMask.rows; y++) {
		int const* seg_ptr = segments.ptr<int>(y);
		uchar* p_ptr = probMap.ptr<uchar>(y);
		uchar const* mask_ptr = fgMask.ptr<uchar>(y);
		for (int x = 0; x < fgMask.cols; x++) {
			if(*(mask_ptr + x) != 0)
				*(p_ptr + x) = (int)((0.7 - probVec[*(seg_ptr + x)]) * 255);
		}
	}

	cv::applyColorMap(probMap, tpHeatMap, cv::COLORMAP_JET);
	tpHeatMap.setTo(cv::Scalar(0, 0, 0), cv::Mat(fgMask == 0));
	cv::imshow("tpHeatMap", tpHeatMap);
	//cv::imshow("tpprobMap", probMap);
#endif
}


void MovingShadRem::whiteAndGray(const cv::Mat& fgMask, const cv::Mat& shadowMask, cv::Mat& whiteAndGrayMask) {
	whiteAndGrayMask = cv::Mat(fgMask.size(), CV_8UC1, 0.0);
	whiteAndGrayMask.setTo(128, shadowMask);
	whiteAndGrayMask.setTo(255, fgMask - shadowMask);
}



//============================
//	Gaussian shadow model
//============================
void MovingShadRem::shadowModelGlobal(cv::Mat& l1, cv::Mat& l2, cv::Mat& l3, const cv::Mat& Frame, const cv::Mat& bg, const cv::Mat& fgMask, cv::Mat& gmmMask) {
	cv::Mat mask(Frame.size(), CV_8U, cv::Scalar(0));
	cv::Mat bgMask(Frame.size(), CV_8U, cv::Scalar(0));
	for (int y = 1; y < l1.rows - 1; ++y) {
		const uchar* l1Ptr = l1.ptr(y);
		const uchar* l2Ptr = l2.ptr(y);
		const uchar* l3Ptr = l3.ptr(y);
		const uchar* framePtr = Frame.ptr(y);
		const uchar* pre_Ptr = Frame.ptr(y - 1);
		const uchar* post_Ptr = Frame.ptr(y + 1);
		for (int x = 1; x < l1.cols - 1; ++x) {
			if (*(l1Ptr + x) && *(l2Ptr + x) && *(l3Ptr + x)) {
				for (int c = 0; c < 3; c++) {
					float haar_x = (float) * (framePtr + 3 * (x - 1) + c) + (float) * (framePtr + 3 * (x + 1) + c) - 2 * (float) * (framePtr + 3 * x + c);
					float haar_y = (float) * (pre_Ptr + 3 * x + c) + (float) * (post_Ptr + 3 * x + c) - 2 * (float) * (framePtr + 3 * x + c);

					if (sh->sigma[c] == 0) {
						sh->mu[c] = (float) * (framePtr + 3 * x + c);
						sh->sigma[c] = cov0;
						sh->mu[c + 3] = haar_x;
						sh->sigma[c + 3] = cov0;
						sh->mu[c + 6] = haar_y;
						sh->sigma[c + 6] = cov0;

					}
					else {
						sh->mu[c] = sh->mu[c] - alphaM * (sh->mu[c] - (float) * (framePtr + 3 * x + c));
						float var_temp = sh->sigma[c] + alphaM * ((sh->mu[c] - (float) * (framePtr + 3 * x + c)) * (sh->mu[c] - (float) * (framePtr + 3 * x + c)) - sh->sigma[c]);
						sh->sigma[c] = var_temp < cov_low ? cov_low : (var_temp > cov_hi ? cov_hi : var_temp);
						sh->mu[c + 3] = sh->mu[c + 3] - alphaM * (sh->mu[c + 3] - haar_x);
						var_temp = sh->sigma[c + 3] + alphaM * ((sh->mu[c + 3] - haar_x) * (sh->mu[c + 3] - haar_x) - sh->sigma[c + 3]);
						sh->sigma[c + 3] = var_temp < cov_low ? cov_low : (var_temp > cov_hi ? cov_hi : var_temp);

						sh->mu[c + 6] = sh->mu[c + 6] - alphaM * (sh->mu[c + 6] - haar_y);
						var_temp = sh->sigma[c + 6] + alphaM * ((sh->mu[c + 6] - haar_y) * (sh->mu[c + 6] - haar_y) - sh->sigma[c + 6]);
						sh->sigma[c + 6] = var_temp < cov_low ? cov_low : (var_temp > cov_hi ? cov_hi : var_temp);
					}
				}
			}
		}
	}

	//-- classification
	for (int y = 0; y < mask.rows; ++y) {
		const uchar* framePtr = Frame.ptr(y);
		const uchar* bgPtr = bg.ptr(y);
		const uchar* maskPtr = fgMask.ptr(y);
		uchar* Ptr = mask.ptr(y);
		//uchar* bgShadPtr = bgMask.ptr(y);
		for (int x = 0; x < mask.cols; ++x) {
			if (*(maskPtr + x) == 255) {
				bool isShadow = true;
				for (int c = 0; c < 3; c++) {
					float dis = (sh->mu[c] - (float) * (framePtr + 3 * x + c)) * (sh->mu[c] - (float) * (framePtr + 3 * x + c));
					if (dis > gmmDist * sh->sigma[c])
						isShadow = false;
				}
				if (isShadow)
					*(Ptr + x) = 255;
				else
					*(Ptr + x) = 0;
			}
		}
	}

	mask.copyTo(gmmMask);
	//std::cout << "mu: " << sh->mu << " , cov: " << sh->sigma << std::endl;
}



//--------------------------------------------------------------------
// A utility function to find set of an element i (uses path compression technique)
int MovingShadRem::find(std::vector<Set>& sets, int i) {
	// find root and make root as parent of i (path compression)
	if (sets[i].parent != i)
		sets[i].parent = find(sets, sets[i].parent);

	return sets[i].parent;
}

// A function that does union of two sets of x and y (uses union by rank)
void MovingShadRem::Union(std::vector<Set>& sets, int xroot, int yroot) {

	// Attach smaller rank tree under root of high rank tree (Union by Rank)
	if (sets[xroot].rank < sets[yroot].rank)
		sets[xroot].parent = yroot;
	else if (sets[xroot].rank > sets[yroot].rank)
		sets[yroot].parent = xroot;

	// If ranks are same, then make one as root and increment its rank by one
	else {
		sets[yroot].parent = xroot;
		sets[xroot].rank++;
	}
}

/**
 * @brief watershedSeg: watershed segmentation
 *
 * @param frame: the current frame
 * @param fgMask: foreground mask image from any BS model
 * @param bg: the background image
 * @param lumRatio: fg/bg
 * @param region_size: size of the blocks
 * @param marks: the final result label image
 */
void MovingShadRem::watershedSeg(const cv::Mat& frame, const cv::Mat& bg, const cv::Mat& fgMask, cv::Mat& lumRatio, int region_size, cv::Mat& marks)
{
	marks = cv::Mat(frame.size(), CV_32S, 0.0);

#if COMPACT
	//-- compact watershed
	cv::Mat boundaries, seeds;
	compact_watershed(lumRatio, boundaries, region_size, region_size, 5.0, seeds);
	boundaries.convertTo(boundaries, CV_32S);
	computeLabelsFromBoundaries(lumRatio, boundaries, marks, -1, -2);
#else
	//-- regular watershed
	long label = 1;
	for (int i = region_size / 2; i < marks.rows; i += region_size) {
		uchar const* mask_ptr = fgMask.ptr<uchar>(i);
		int* marks_ptr = marks.ptr<int>(i);
		for (int j = region_size / 2; j < marks.cols; j += region_size) {
			if ((int) * (mask_ptr + j) == 0)
				*(marks_ptr + j) = 0;
			else {
				*(marks_ptr + j) = label;
				label++;
			}
		}
	}
	numSeg = label;
	watershedEx(lumRatio, marks);
#endif

	marks.setTo(0, ~fgMask);
}


/**
 * @brief mergeSegments: merge segmentation
 *
 * @param frame: the current frame
 * @param fgMask: foreground mask image from any BS model
 * @param bg: the background image
 * @param lumRatio: fg/bg
 * @param region_size: size of the blocks
 * @param marks: initial label image
 * @param segments: merged label image
 */
void MovingShadRem::mergeSegments(const cv::Mat& frame, const cv::Mat& bg, const cv::Mat& fgMask, const cv::Mat& lumRatio, int region_size, const cv::Mat& marks, cv::Mat& segments) {
	//-- calculate means and counts of segments
	std::vector<std::vector<int>> meanVec(numSeg, std::vector<int>(3, 0));
	std::vector<int> areaVec(numSeg, 0);
	for (int y = 0; y < lumRatio.rows; y++) {
		const int* marks_ptr = marks.ptr<int>(y);
		const uchar* lum_ptr = lumRatio.ptr<uchar>(y);
		for (int x = 0; x < lumRatio.cols; x++) {
			if (*(marks_ptr + x) > 0) {
				for (int c = 0; c < 3; c++) 
					meanVec[*(marks_ptr + x )][c] += (int)*(lum_ptr + 3 * x + c);
				areaVec[*(marks_ptr + x)] ++;
			}
		}
	}

	for (int i = 1; i < numSeg; i++) 
		for (int c = 0; c < 3; c++)
			meanVec[i][c] = areaVec[i] == 0 ? 0 : meanVec[i][c] / areaVec[i];


	//-------------------------------------------
	//-- merge
	std::vector<Set> sets(numSeg);
	for (int v = 0; v < numSeg; ++v) 
		sets[v].parent = v;

	for (int y = stepSize; y < marks.rows; y += stepSize) {
		int const* marks_ptr = marks.ptr<int>(y);
		int const* marks_ptr_pre = marks.ptr<int>(y - stepSize);
		uchar const* mask_ptr = fgMask.ptr<uchar>(y);
		uchar const* mask_ptr_pre = fgMask.ptr(y - stepSize);
		uchar const* lum_ptr = lumRatio.ptr<uchar>(y);
		uchar const* lum_ptr_pre = lumRatio.ptr(y - stepSize);
		uchar const* fr_ptr = frame.ptr<uchar>(y);
		uchar const* fr_ptr_pre = frame.ptr(y - stepSize);
		uchar const* bg_ptr = bg.ptr<uchar>(y);
		uchar const* bg_ptr_pre = bg.ptr(y - stepSize);

		for (int x = stepSize; x < marks.cols; x += stepSize) {
			if ((int) *(mask_ptr + x) == 255) {
				//-- left
				if ((int)*(mask_ptr + x - stepSize) == 255 && (int) *(marks_ptr + x - stepSize) != (int) *(marks_ptr + x)) {
					int count = 0;
					for (int c = 0; c < 3; c++) {
						if (
							(abs((int) * (lum_ptr + 3 * x + c) - (int) * (lum_ptr + 3 * (x - stepSize) + c)) < SEG_THRESH)
							//|| (abs(abs((int)*(fr_ptr + 3 * x + c)- (int) * (fr_ptr + 3 * (x - stepSize) + c)) - abs((int) * (bg_ptr + 3 * x + c) - (int) * (bg_ptr + 3 * (x - stepSize) + c))) < SEG_THRESH)
							//|| (abs((int)meanVec[*(marks_ptr + x)][c] - (int)meanVec[*(marks_ptr + x - stepSize)][c]) < SEG_THRESH)
							)
							count++;
					}
					if (count == 3) {
						int current = find(sets, *(marks_ptr + x));
						int left = find(sets, *(marks_ptr + x - stepSize));
						Union(sets, current, left);
					}
				}

				//-- up
				if ((int) *(mask_ptr_pre + x) == 255 && *(marks_ptr_pre + x) != *(marks_ptr + x)) {
					int count = 0;
					for (int c = 0; c < 3; c++)
						if (
							(abs((int) * (lum_ptr + 3 * x + c) - (int) * (lum_ptr_pre + 3 * x + c)) < SEG_THRESH)
							//|| (abs(abs((int) * (fr_ptr + 3 * x + c) - (int) * (fr_ptr_pre + 3 * x + c)) - abs((int) * (bg_ptr + 3 * x + c) - (int) * (bg_ptr_pre + 3 * x + c))) < SEG_THRESH)
							//|| (abs(meanVec[*(marks_ptr + x)][c] - meanVec[*(marks_ptr_pre + x)][c]) < SEG_THRESH)
							)
							count++;
					if (count == 3) {
						int first = find(sets, *(marks_ptr + x));
						int second = find(sets, *(marks_ptr_pre + x));
						Union(sets, first, second);
					}
				}
			}
		}
	}

	//-- construct segments
	segments = cv::Mat::zeros(marks.size(), CV_32SC1);
	std::unordered_map<int, int> labels;
	int label = 1;
	for (int y = 0; y < marks.rows; y++) {
		int* segments_ptr = segments.ptr<int>(y);
		int const* marks_ptr = marks.ptr<int>(y);
		uchar const* mask_ptr = fgMask.ptr<uchar>(y);
		for (int x = 0; x < marks.cols; x++) {
			if (*(mask_ptr + x) && areaVec[*(marks_ptr + x)] > minSegArea) {
				int root = find(sets, *(marks_ptr + x));
				if (labels.find(root) == labels.end()) {
					labels.insert({ root, label });
					label++;
				}
				*(segments_ptr + x) = labels[root];
			}	
		}
	}
	numSet = labels.size();

	

#if DEBUG
	cv::Mat means(fgMask.size(), CV_8UC3, 0.0);
	for (int y = 0; y < lumRatio.rows; y++) {
		uchar* means_ptr = means.ptr<uchar>(y);
		const int* marks_ptr = marks.ptr<int>(y);
		for (int x = 0; x < lumRatio.cols; x++)
			if (*(marks_ptr + x) > 0)
				for (int c = 0; c < 3; c++)
					*(means_ptr + 3 * x + c) = meanVec[*(marks_ptr + x)][c];
	}

	cv::imshow("means", means);
	means.copyTo(ratioMean);

	std::vector<cv::Vec3b> colorTab;
	for (int i = 0; i <= 600; i++) {
		int b = cv::theRNG().uniform(0, 255);
		int g = cv::theRNG().uniform(0, 255);
		int r = cv::theRNG().uniform(0, 255);
		colorTab.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}


	for (int i = 0; i < marks.rows; i++) {
		for (int j = 0; j < marks.cols; j++) {
			int index = marks.at<int>(i, j);
			if (index == -1)
				watershedImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);//watershed line
			else if(index == 0)
				watershedImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);//bg
			else
				watershedImage.at<cv::Vec3b>(i, j) = colorTab[index % 600];//masks with different colors for single region
		}
	}

	//for (int i = region_size / 2; i < marks.rows; i += region_size) {
	//	for (int j = region_size / 2; j < marks.cols; j += region_size) {
	//		watershedImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
	//	}
	//}

	imshow("watershed", watershedImage);

	for (int i = 0; i < segments.rows; i++) {
		for (int j = 0; j < segments.cols; j++) {
			int index = segments.at<int>(i, j);
			if (index == -1)
				watershedMerged.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);//watershed line
			else if (index == 0)
				watershedMerged.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);//bg
			else
				watershedMerged.at<cv::Vec3b>(i, j) = colorTab[index % 600];//masks with different colors for single region
		}
	}

	//for (int i = region_size / 2; i < marks.rows; i += region_size) {
	//	for (int j = region_size / 2; j < marks.cols; j += region_size) {
	//		watershedMerged.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
	//	}
	//}

	imshow("watershedMerged", watershedMerged);

	//cv::Mat convert_img, wshed;
	//frame.convertTo(convert_img, CV_8UC3);
	//addWeighted(convert_img, 0.5, watershedImage, 0.5, 0, wshed);
	//cv::imshow("AddWeighted Image", wshed);
#endif
}




//===========================================================
//	post-processing functions
//===========================================================
/**
 * @brief noiseCorrection: check the pixel, if it belongs to shadow or foreground
 *						 ( F. Edge Noise Correction )
 *						 check a widow around each pixel in the shadowMask, if most pixels in that
 *						 window are shadow classify the pixel as shadow, otherwise as foreground
 *
 * @param shadowMask: a matrix represent background(0), foreground(255) and shadow(127)
 */
void MovingShadRem::noiseCorrection(const cv::Mat& fgMask, cv::Mat& shadowMask, int windowSize, cv::Mat& wAndG) {
	int i1, i2, j1, j2, y, x, m = 0, n = 0, i = 0, j = 0;
	int w = fgMask.cols;
	int h = fgMask.rows;

	wAndG = cv::Mat(fgMask.size(), CV_8UC1, 0.0);
	whiteAndGray(fgMask, shadowMask, wAndG);

	for (y = 0; y < h; y++) {
		uchar* maskPtr = wAndG.ptr<uchar>(y);
		for (x = 0; x < w; x++) {
			if (*(maskPtr + x) != 0) {
				i1 = MAX(0, y - windowSize);
				i2 = MIN(h - 1, y + windowSize);
				j1 = MAX(0, x - windowSize);
				j2 = MIN(w - 1, x + windowSize);

				for (i = i1; i <= i2; ++i)
				{
					uchar* wPtr = wAndG.ptr<uchar>(i);
					for (j = j1; j <= j2; ++j)
					{
						if (*(wPtr + j) == 255)
							++n;
						if (*(wPtr + j) == 128)
							++m;
					}
				}

				if (m > n) *(maskPtr + x) = 128;
				else if (n > m) *(maskPtr + x) = 255;

				m = 0;
				n = 0;
			}
		}
	}
}












//========================================
//-- watershed
//========================================

struct WSNode
{
	int next;
	int mask_ofs;
	int img_ofs;
};

// Queue for WSNodes
struct WSQueue
{
	WSQueue() { first = last = 0; }
	int first, last;
};


static int allocWSNodes(std::vector<WSNode>& storage)
{
	int sz = (int)storage.size();
	int newsz = MAX(128, sz * 3 / 2);

	storage.resize(newsz);
	if (sz == 0)
	{
		storage[0].next = 0;
		sz = 1;
	}
	for (int i = sz; i < newsz - 1; i++)
		storage[i].next = i + 1;
	storage[newsz - 1].next = 0;
	return sz;
}

//the modified version of watershed algorithm from OpenCV
void watershedEx(cv::Mat& src, cv::Mat& dst)
{
	// Labels for pixels
	const int IN_QUEUE = -2; // Pixel visited
	// possible bit values = 2^8
	const int NQ = 256;

	cv::Size size = src.size();
	int channel = src.channels();
	// Vector of every created node
	std::vector<WSNode> storage;
	int free_node = 0, node;
	// Priority queue of queues of nodes
	// from high priority (0) to low priority (255)
	WSQueue q[NQ];
	// Non-empty queue with highest priority
	int active_queue;
	int i, j;
	// Color differences
	int db, dg, dr;
	int subs_tab[513];

	// MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
	// MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

	// Create a new node with offsets mofs and iofs in queue idx
#define ws_push(idx,mofs,iofs)          \
        {                                       \
    if (!free_node)                    \
    free_node = allocWSNodes(storage); \
    node = free_node;                   \
    free_node = storage[free_node].next; \
    storage[node].next = 0;             \
    storage[node].mask_ofs = mofs;      \
    storage[node].img_ofs = iofs;       \
    if (q[idx].last)                   \
    storage[q[idx].last].next = node; \
    else                                \
  q[idx].first = node;            \
  q[idx].last = node;                 \
        }

	// Get next node from queue idx
#define ws_pop(idx,mofs,iofs)           \
        {                                       \
    node = q[idx].first;                \
    q[idx].first = storage[node].next;  \
    if (!storage[node].next)           \
    q[idx].last = 0;                \
    storage[node].next = free_node;     \
    free_node = node;                   \
    mofs = storage[node].mask_ofs;      \
    iofs = storage[node].img_ofs;       \
        }

	// Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
        {                                        \
    db = std::abs((ptr1)[0] - (ptr2)[0]); \
    dg = std::abs((ptr1)[1] - (ptr2)[1]); \
    dr = std::abs((ptr1)[2] - (ptr2)[2]); \
    diff = ws_max(db, dg);                \
    diff = ws_max(diff, dr);              \
    assert(0 <= diff && diff <= 255);  \
        }

	//get absolute difference in diff
#define c_gray_diff(ptr1,ptr2,diff)		\
        {									\
    diff = std::abs((ptr1)[0] - (ptr2)[0]);	\
    assert(0 <= diff&&diff <= 255);		\
        }

	CV_Assert(src.type() == CV_8UC3 || src.type() == CV_8UC1 && dst.type() == CV_32SC1);
	CV_Assert(src.size() == dst.size());

	// Current pixel in input image
	const uchar* img = src.ptr();
	// Step size to next row in input image
	int istep = int(src.step / sizeof(img[0]));

	// Current pixel in mask image
	int* mask = dst.ptr<int>();
	// Step size to next row in mask image
	int mstep = int(dst.step / sizeof(mask[0]));

	for (i = 0; i < 256; i++)
		subs_tab[i] = 0;
	for (i = 256; i <= 512; i++)
		subs_tab[i] = i - 256;

	//for (j = 0; j < size.width; j++)
	//mask[j] = mask[j + mstep*(size.height - 1)] = 0;

	// initial phase: put all the neighbor pixels of each marker to the ordered queue -
	// determine the initial boundaries of the basins
	for (i = 1; i < size.height - 1; i++) {
		img += istep; mask += mstep;
		mask[0] = mask[size.width - 1] = 0; // boundary pixels

		for (j = 1; j < size.width - 1; j++) {
			int* m = mask + j;
			if (m[0] < 0)
				m[0] = 0;
			if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
			{
				// Find smallest difference to adjacent markers
				const uchar* ptr = img + j * channel;
				int idx = 256, t;
				if (m[-1] > 0) {
					if (channel == 3) {
						c_diff(ptr, ptr - channel, idx);
					}
					else {
						c_gray_diff(ptr, ptr - channel, idx);
					}
				}
				if (m[1] > 0) {
					if (channel == 3) {
						c_diff(ptr, ptr + channel, t);
					}
					else {
						c_gray_diff(ptr, ptr + channel, t);
					}
					idx = ws_min(idx, t);
				}
				if (m[-mstep] > 0) {
					if (channel == 3) {
						c_diff(ptr, ptr - istep, t);
					}
					else {
						c_gray_diff(ptr, ptr - istep, t);
					}
					idx = ws_min(idx, t);
				}
				if (m[mstep] > 0) {
					if (channel == 3) {
						c_diff(ptr, ptr + istep, t);
					}
					else {
						c_gray_diff(ptr, ptr + istep, t);
					}
					idx = ws_min(idx, t);
				}

				// Add to according queue
				assert(0 <= idx && idx <= 255);
				ws_push(idx, i * mstep + j, i * istep + j * channel);
				m[0] = IN_QUEUE;//initial unvisited
			}
		}
	}
	// find the first non-empty queue
	for (i = 0; i < NQ; i++)
		if (q[i].first)
			break;

	// if there is no markers, exit immediately
	if (i == NQ)
		return;

	active_queue = i;//first non-empty priority queue
	img = src.ptr();
	mask = dst.ptr<int>();

	// recursively fill the basins
	for (;;)
	{
		int mofs, iofs;
		int lab = 0, t;
		int* m;
		const uchar* ptr;

		// Get non-empty queue with highest priority
		// Exit condition: empty priority queue
		if (q[active_queue].first == 0)
		{
			for (i = active_queue + 1; i < NQ; i++)
				if (q[i].first)
					break;
			if (i == NQ)
			{
				std::vector<WSNode>().swap(storage);
				break;
			}
			active_queue = i;
		}

		// Get next node
		ws_pop(active_queue, mofs, iofs);
		int top = 1, bottom = 1, left = 1, right = 1;
		if (0 <= mofs && mofs < mstep)//pixel on the top
			top = 0;
		if ((mofs % mstep) == 0)//pixel in the left column
			left = 0;
		if ((mofs + 1) % mstep == 0)//pixel in the right column
			right = 0;
		if (mstep * (size.height - 1) <= mofs && mofs < mstep * size.height)//pixel on the bottom
			bottom = 0;

		// Calculate pointer to current pixel in input and marker image
		m = mask + mofs;
		ptr = img + iofs;
		int diff, temp;
		// Check surrounding pixels for labels to determine label for current pixel
		if (left) {//the left point can be visited
			t = m[-1];
			if (t > 0) {
				lab = t;
				if (channel == 3) {
					c_diff(ptr, ptr - channel, diff);
				}
				else {
					c_gray_diff(ptr, ptr - channel, diff);
				}
			}
		}
		if (right) {// Right point can be visited
			t = m[1];
			if (t > 0) {
				if (lab == 0) {//and this point hasn't been labeled before
					lab = t;
					if (channel == 3) {
						c_diff(ptr, ptr + channel, diff);
					}
					else {
						c_gray_diff(ptr, ptr + channel, diff);
					}
				}
				else if (t != lab) {
					if (channel == 3) {
						c_diff(ptr, ptr + channel, temp);
					}
					else {
						c_gray_diff(ptr, ptr + channel, temp);
					}
					diff = ws_min(diff, temp);
					if (diff == temp)
						lab = t;
				}
			}
		}
		if (top) {
			t = m[-mstep]; // Top
			if (t > 0) {
				if (lab == 0) {//and this point hasn't been labeled before
					lab = t;
					if (channel == 3) {
						c_diff(ptr, ptr - istep, diff);
					}
					else {
						c_gray_diff(ptr, ptr - istep, diff);
					}
				}
				else if (t != lab) {
					if (channel == 3) {
						c_diff(ptr, ptr - istep, temp);
					}
					else {
						c_gray_diff(ptr, ptr - istep, temp);
					}
					diff = ws_min(diff, temp);
					if (diff == temp)
						lab = t;
				}
			}
		}
		if (bottom) {
			t = m[mstep]; // Bottom
			if (t > 0) {
				if (lab == 0) {
					lab = t;
				}
				else if (t != lab) {
					if (channel == 3) {
						c_diff(ptr, ptr + istep, temp);
					}
					else {
						c_gray_diff(ptr, ptr + istep, temp);
					}
					diff = ws_min(diff, temp);
					if (diff == temp)
						lab = t;
				}
			}
		}
		// Set label to current pixel in marker image
		assert(lab != 0);//lab must be labeled with a nonzero number
		m[0] = lab;

		// Add adjacent, unlabeled pixels to corresponding queue
		if (left) {
			if (m[-1] == 0)//left pixel with marker 0
			{
				if (channel == 3) {
					c_diff(ptr, ptr - channel, t);
				}
				else {
					c_gray_diff(ptr, ptr - channel, t);
				}
				ws_push(t, mofs - 1, iofs - channel);
				active_queue = ws_min(active_queue, t);
				m[-1] = IN_QUEUE;
			}
		}

		if (right)
		{
			if (m[1] == 0)//right pixel with marker 0
			{
				if (channel == 3) {
					c_diff(ptr, ptr + channel, t);
				}
				else {
					c_gray_diff(ptr, ptr + channel, t);
				}
				ws_push(t, mofs + 1, iofs + channel);
				active_queue = ws_min(active_queue, t);
				m[1] = IN_QUEUE;
			}
		}

		if (top)
		{
			if (m[-mstep] == 0)//top pixel with marker 0
			{
				if (channel == 3) {
					c_diff(ptr, ptr - istep, t);
				}
				else {
					c_gray_diff(ptr, ptr - istep, t);
				}
				ws_push(t, mofs - mstep, iofs - istep);
				active_queue = ws_min(active_queue, t);
				m[-mstep] = IN_QUEUE;
			}
		}

		if (bottom) {
			if (m[mstep] == 0)//down pixel with marker 0
			{
				if (channel == 3) {
					c_diff(ptr, ptr + istep, t);
				}
				else {
					c_gray_diff(ptr, ptr + istep, t);
				}
				ws_push(t, mofs + mstep, iofs + istep);
				active_queue = ws_min(active_queue, t);
				m[mstep] = IN_QUEUE;
			}
		}
	}
}


//========================================
//-- compact watershed
//========================================

typedef struct CvWSNode
{
	struct CvWSNode* next;
	int mask_ofs;
	int img_ofs;
	float compVal;
}
CvWSNode;

typedef struct CvWSQueue
{
	CvWSNode* first;
	CvWSNode* last;
}
CvWSQueue;

static CvWSNode* icvAllocWSNodes(CvMemStorage* storage)
{
	CvWSNode* n = 0;

	int i, count = (storage->block_size - sizeof(CvMemBlock)) / sizeof(*n) - 1;

	n = (CvWSNode*)cvMemStorageAlloc(storage, count * sizeof(*n));
	for (i = 0; i < count - 1; i++)
		n[i].next = n + i + 1;
	n[count - 1].next = 0;

	return n;
}


void compactWatershedSeg(const CvArr* srcarr, CvArr* dstarr, float compValStep) {
	if (compValStep > 5)
		printf("cws::cvWatershed Warning: Large compValStep values can cause seg faults");

	const int IN_QUEUE = -2;
	const int WSHED = -1;
	const int NQ = 1024;
	cv::Ptr<CvMemStorage> storage;

	CvMat sstub, * src;
	CvMat dstub, * dst;
	CvSize size;
	CvWSNode* free_node = 0, * node;
	CvWSQueue q[NQ];
	int active_queue;
	int i, j;
	int db, dg, dr;
	int* mask;
	uchar* img;
	int mstep, istep;
	int subs_tab[2 * NQ + 1];

	// MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
// MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

#define ws_push(idx,mofs,iofs,cV)  \
      {                               \
          if( !free_node )            \
              free_node = icvAllocWSNodes( storage );\
          node = free_node;           \
          free_node = free_node->next;\
          node->next = 0;             \
          node->mask_ofs = mofs;      \
          node->img_ofs = iofs;       \
          node->compVal = cV;    \
          if( q[idx].last )           \
              q[idx].last->next=node; \
          else                        \
              q[idx].first = node;    \
          q[idx].last = node;         \
      }

#define ws_pop(idx,mofs,iofs,cV)   \
      {                               \
          node = q[idx].first;        \
          q[idx].first = node->next;  \
          if( !node->next )           \
              q[idx].last = 0;        \
          node->next = free_node;     \
          free_node = node;           \
          mofs = node->mask_ofs;      \
          iofs = node->img_ofs;       \
          cV = node->compVal;       \
      }

#define c_diff(ptr1,ptr2,diff)      \
      {                                   \
          db = abs((ptr1)[0] - (ptr2)[0]);\
          dg = abs((ptr1)[1] - (ptr2)[1]);\
          dr = abs((ptr1)[2] - (ptr2)[2]);\
          diff = ws_max(db,dg);           \
          diff = ws_max(diff,dr);         \
          assert( 0 <= diff && diff <= 255 ); \
      }

	src = cvGetMat(srcarr, &sstub);
	dst = cvGetMat(dstarr, &dstub);

	if (CV_MAT_TYPE(src->type) != CV_8UC3)
		CV_Error(CV_StsUnsupportedFormat, "Only 8-bit, 3-channel input images are supported");

	if (CV_MAT_TYPE(dst->type) != CV_32SC1)
		CV_Error(CV_StsUnsupportedFormat,
			"Only 32-bit, 1-channel output images are supported");

	if (!CV_ARE_SIZES_EQ(src, dst))
		CV_Error(CV_StsUnmatchedSizes, "The input and output images must have the same size");

	size = cv::Size(src->cols, src->rows);
	storage = cvCreateMemStorage();

	istep = src->step;
	img = src->data.ptr;
	mstep = dst->step / sizeof(mask[0]);
	mask = dst->data.i;

	memset(q, 0, NQ * sizeof(q[0]));

	for (i = 0; i < NQ; i++)
		subs_tab[i] = 0;
	for (i = NQ; i <= 2 * NQ; i++)
		subs_tab[i] = i - NQ;

	// draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
	for (j = 0; j < size.width; j++)
		mask[j] = mask[j + mstep * (size.height - 1)] = WSHED;

	// initial phase: put all the neighbor pixels of each marker to the ordered queue -
	// determine the initial boundaries of the basins
	for (i = 1; i < size.height - 1; i++)
	{
		img += istep; mask += mstep;
		mask[0] = mask[size.width - 1] = WSHED;

		for (j = 1; j < size.width - 1; j++)
		{
			int* m = mask + j;
			if (m[0] < 0) m[0] = 0;
			if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
			{
				uchar* ptr = img + j * 3;
				int idx = 256, t;
				if (m[-1] > 0)
					c_diff(ptr, ptr - 3, idx);
				if (m[1] > 0)
				{
					c_diff(ptr, ptr + 3, t);
					idx = ws_min(idx, t);
				}
				if (m[-mstep] > 0)
				{
					c_diff(ptr, ptr - istep, t);
					idx = ws_min(idx, t);
				}
				if (m[mstep] > 0)
				{
					c_diff(ptr, ptr + istep, t);
					idx = ws_min(idx, t);
				}
				assert(0 <= idx && idx <= NQ - 1);
				ws_push(idx, i * mstep + j, i * istep + j * 3, 0.0);
				m[0] = IN_QUEUE;
			}
		}
	}

	// find the first non-empty queue
	for (i = 0; i < NQ; i++)
		if (q[i].first)
			break;

	// if there is no markers, exit immediately
	if (i == NQ)
		return;

	active_queue = i;
	img = src->data.ptr;
	mask = dst->data.i;

	// recursively fill the basins
	for (;;)
	{
		int mofs, iofs;
		int lab = 0, t;
		int* m;
		uchar* ptr;

		// search for next queue
		if (q[active_queue].first == 0)
		{
			for (i = active_queue + 1; i < NQ; i++)
				if (q[i].first)
					break;
			if (i == NQ)
				break;
			active_queue = i;
		}

		// get next element of this queue
		float compVal;
		ws_pop(active_queue, mofs, iofs, compVal);

		m = mask + mofs; // pointer to element in mask
		ptr = img + iofs; // pointer to element in image

		// have a look at all neighbors, if they have different label, mark
		// as watershed and continue
		t = m[-1];
		if (t > 0) lab = t;
		t = m[1];
		if (t > 0)
		{
			if (lab == 0) lab = t;
			else if (t != lab) lab = WSHED;
		}
		t = m[-mstep];
		if (t > 0)
		{
			if (lab == 0) lab = t;
			else if (t != lab) lab = WSHED;
		}
		t = m[mstep];
		if (t > 0)
		{
			if (lab == 0) lab = t;
			else if (t != lab) lab = WSHED;
		}
		assert(lab != 0);
		m[0] = lab;

		if (lab == WSHED)
			continue;


		// have a look at all neighbors
		// 

		if (m[-1] == 0)
		{
			c_diff(ptr, ptr - 3, t); // store difference to this neighbor in t (current gradient)
			ws_push(int(round(t + compVal)), mofs - 1, iofs - 3, compVal + compValStep); // store in queue
			active_queue = ws_min(active_queue, t); // check if queue of this element is prior to the current queue (and should be proceeded in the next iteration)
			m[-1] = IN_QUEUE; // mark in mask as in a queue
		}
		if (m[1] == 0)
		{
			c_diff(ptr, ptr + 3, t);
			ws_push(int(round(t + compVal)), mofs + 1, iofs + 3, compVal + compValStep);
			active_queue = ws_min(active_queue, t);
			m[1] = IN_QUEUE;
		}
		if (m[-mstep] == 0)
		{
			c_diff(ptr, ptr - istep, t);
			ws_push(int(round(t + compVal)), mofs - mstep, iofs - istep, compVal + compValStep);
			active_queue = ws_min(active_queue, t);
			m[-mstep] = IN_QUEUE;
		}
		if (m[mstep] == 0)
		{
			c_diff(ptr, ptr + istep, t);
			ws_push(int(round(t + compVal)), mofs + mstep, iofs + istep, compVal + compValStep);
			active_queue = ws_min(active_queue, t);
			m[mstep] = IN_QUEUE;
		}
	}
}


void compact_watershed(cv::Mat& img, cv::Mat& B, float dy, float dx, float compValStep, cv::Mat& seeds)
{
	cv::Mat markers = cv::Mat::zeros(img.rows, img.cols, CV_32SC1);
	if (seeds.empty())
	{
		int labelIdx = 1;
		for (float i = dy / 2; i < markers.rows; i += dy)
		{
			for (float j = dx / 2; j < markers.cols; j += dx)
			{
				markers.at<int>(floor(i), floor(j)) = labelIdx;
				labelIdx++;
			}
		}
	}
	else
	{
		// use given seeds
		int labelIdx = 1;
		for (int i = 0; i < seeds.cols; i++)
		{
			//cout << "set "<<round(seeds.at<float>(0,i))<< " "<<round(seeds.at<float>(1,i))<<" to "<<labelIdx<<endl;
			markers.at<int>(round(seeds.at<float>(0, i)), round(seeds.at<float>(1, i))) = labelIdx;
			labelIdx++;
		}
	}

	// run compact watershed
	IplImage tmp = img;
	IplImage tmp2 = markers;
	compactWatershedSeg(&tmp, &tmp2, compValStep);
	markers = cv::cvarrToMat(&tmp2); ;

	// create boundary map
	B = markers < 0;

	// extend boundary map to image borders
	for (int i = 0; i < B.cols; i++)
	{
		if (B.at<uchar>(1, i))
			B.at<uchar>(0, i) = 255;
		else
			B.at<uchar>(0, i) = 0;
		if (B.at<uchar>(B.rows - 2, i))
			B.at<uchar>(B.rows - 1, i) = 255;
		else
			B.at<uchar>(B.rows - 1, i) = 0;
	}

	for (int i = 0; i < B.rows; i++)
	{
		if (B.at<uchar>(i, 1))
			B.at<uchar>(i, 0) = 255;
		else
			B.at<uchar>(i, 0) = 0;
		if (B.at<uchar>(i, B.cols - 2))
			B.at<uchar>(i, B.cols - 1) = 255;
		else
			B.at<uchar>(i, B.cols - 1) = 0;
	}
}


float computeDistance(const cv::Vec3b& x, const cv::Vec3b& y) {
	return (x[0] - y[0]) * (x[0] - y[0])
		+ (x[1] - y[1]) * (x[1] - y[1])
		+ (x[2] - y[2]) * (x[2] - y[2]);
}

void computeLabelsFromBoundaries(const cv::Mat& image, const cv::Mat& boundaries, cv::Mat& labels, int BOUNDARY_VALUE, int INNER_VALUE) {

	cv::Mat tmp_labels(boundaries.rows, boundaries.cols, CV_32FC1, cv::Scalar(0));
	for (int i = 0; i < tmp_labels.rows; i++) {
		for (int j = 0; j < tmp_labels.cols; j++) {
			if (boundaries.at<int>(i, j) > 0) {
				tmp_labels.at<float>(i, j) = BOUNDARY_VALUE;
			}
			else {
				tmp_labels.at<float>(i, j) = INNER_VALUE;
			}
		}
	}

	int label = 1;
	for (int i = 0; i < tmp_labels.rows; i++) {
		for (int j = 0; j < tmp_labels.cols; j++) {
			if (tmp_labels.at<float>(i, j) == INNER_VALUE) {
				cv::floodFill(tmp_labels, cv::Point(j, i), cv::Scalar(label),
					0, cv::Scalar(0), cv::Scalar(0));
				label++;
			}
		}
	}

	tmp_labels.convertTo(tmp_labels, CV_32SC1);
	std::vector<cv::Vec3b> means(label + 1, cv::Vec3b(0, 0, 0));
	std::vector<int> counts(label + 1, 0);

	for (int i = 0; i < tmp_labels.rows; i++) {
		for (int j = 0; j < tmp_labels.cols; j++) {
			if (tmp_labels.at<int>(i, j) != BOUNDARY_VALUE) {
				means[tmp_labels.at<int>(i, j) - 1][0] += image.at<cv::Vec3b>(i, j)[0];
				means[tmp_labels.at<int>(i, j) - 1][1] += image.at<cv::Vec3b>(i, j)[1];
				means[tmp_labels.at<int>(i, j) - 1][2] += image.at<cv::Vec3b>(i, j)[2];
				counts[tmp_labels.at<int>(i, j) - 1]++;
			}
		}
	}

	for (int k = 0; k < label + 1; k++) {
		if (counts[k] > 0) {
			means[k][0] /= counts[k];
			means[k][1] /= counts[k];
			means[k][2] /= counts[k];
		}
	}

	labels.create(image.rows, image.cols, CV_32SC1);
	for (int i = 0; i < tmp_labels.rows; i++) {
		for (int j = 0; j < tmp_labels.cols; j++) {
			if (tmp_labels.at<int>(i, j) == BOUNDARY_VALUE) {

				int min_label = BOUNDARY_VALUE;
				//float min_distance = std::numeric_limits<float>::max();
				float min_distance = 99999999.0;

				if (i + 1 < image.rows && tmp_labels.at<int>(i + 1, j) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[tmp_labels.at<int>(i + 1, j) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = tmp_labels.at<int>(i + 1, j);
					}
				}

				if (j + 1 < image.cols && tmp_labels.at<int>(i, j + 1) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[tmp_labels.at<int>(i, j + 1) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = tmp_labels.at<int>(i, j + 1);
					}
				}

				if (i > 0 && tmp_labels.at<int>(i - 1, j) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[tmp_labels.at<int>(i - 1, j) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = tmp_labels.at<int>(i - 1, j);
					}
				}

				if (j > 0 && tmp_labels.at<int>(i, j - 1) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[tmp_labels.at<int>(i, j - 1) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = tmp_labels.at<int>(i, j - 1);
					}
				}

				labels.at<int>(i, j) = min_label;
			}
			else {
				labels.at<int>(i, j) = tmp_labels.at<int>(i, j);
			}
		}
	}

	// Second pass to resolve diagonal issues.
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (labels.at<int>(i, j) == BOUNDARY_VALUE) {

				int min_label = BOUNDARY_VALUE;
				//float min_distance = std::numeric_limits<float>::max();
				float min_distance = 99999999.0;

				if (i + 1 < image.rows && labels.at<int>(i + 1, j) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[labels.at<int>(i + 1, j) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = labels.at<int>(i + 1, j);
					}
				}

				if (j + 1 < image.cols && labels.at<int>(i, j + 1) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[labels.at<int>(i, j + 1) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = labels.at<int>(i, j + 1);
					}
				}

				if (i > 0 && labels.at<int>(i - 1, j) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[labels.at<int>(i - 1, j) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = labels.at<int>(i - 1, j);
					}
				}

				if (j > 0 && labels.at<int>(i, j - 1) != BOUNDARY_VALUE) {
					float distance = computeDistance(image.at<cv::Vec3b>(i, j), means[labels.at<int>(i, j - 1) - 1]);

					if (distance < min_distance) {
						min_distance = distance;
						min_label = labels.at<int>(i, j - 1);
					}
				}

				labels.at<int>(i, j) = min_label;
			}
		}
	}

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			labels.at<int>(i, j)--;
		}
	}
}

void relabelSuperpixels(cv::Mat& labels) {

	int max_label = 0;
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			if (labels.at<int>(i, j) > max_label) {
				max_label = labels.at<int>(i, j);
			}
		}
	}

	int current_label = 0;
	std::vector<int> label_correspondence(max_label + 1, -1);

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels.at<int>(i, j);

			if (label_correspondence[label] < 0) {
				label_correspondence[label] = current_label++;
			}

			labels.at<int>(i, j) = label_correspondence[label];
		}
	}
}






