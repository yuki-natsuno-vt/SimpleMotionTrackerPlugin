// SimpleMotionTracker.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "SimpleMotionTracker.h"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <map>
#include <chrono>
#include "OpenCVDeviceEnumerator/DeviceEnumerator.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

class SimpleMotionTracker {
public:
	void initVideoDeviceList();
	void init(const std::string& videoDeviceName);
	void init(int cameraId);
	void destroy();
	void update();

	void setUseARMarker(bool useARMarker);
	void setUseFaceTracking(bool useFaceTracking);
	void setUseEyesTracking(bool useEyesTracking);
	void setUseHandTracking(bool useHandTracking);

	void setCaptureShown(bool isShown);
	bool isCaptureShown();

	void setIrisThresh(int thresh) { _irisThresh = thresh; }

	void setARMarkerEdgeLength(float length);
	bool isARMarkerDetected(int id);
	void getARMarker6DoF(int id, float* outArray);


	void setMinHandTranslationThreshold(float thresh) { _minHandTranslationThreshold = thresh; }
	void setMaxHandTranslationThreshold(float thresh) { _maxHandTranslationThreshold = thresh; }
	void setHandUndetectedDuration(int msec) { _handUndetectedDuration = msec; }
	bool isFacePointsDetected();
	void getFacePoints(float* outArray);

	bool isLeftHandDetected();
	bool isRightHandDetected();
	bool isLeftHandDown();
	bool isRightHandDown();
	void getHandPoints(float* outArray);

	int getErrorCode();

private:
	void loadCameraParam(const std::string& fileName);
	static cv::Rect adjustRect(const cv::Rect& rect, const cv::Mat frame);
	void detectARMarker();

	static void detectMultiScaleRecursive(cv::CascadeClassifier& cascade, cv::Mat& frame, std::vector<cv::Rect>& outRects, cv::Size min, cv::Size max, int depth);
	void detectFace();
	void detectFaceDlib();
	void detectIris(int irisThresh, cv::Mat faceFrame, cv::Rect eyeRect, cv::Rect& outIris);
	void detectIrisDlib(int irisThresh, cv::Mat faceFrame, cv::Rect eyeRect, cv::Rect& outIris);

	void detectHandCircle(float* handCircle, bool& isHandDetected, bool& isHandDown,  int& handUndetectedTick,
		                  const cv::Point& point, std::vector<cv::Point>& points,
		                  float minHandRadius, float maxHandRadius,
						  std::list<float>& radiusBuf, float resizeRatio, cv::Mat frame);
	void detectHand();

	bool _useARMarker = false;
	bool _useFaceTracking = false;
	bool _useEyesTracking = false;
	bool _useHandTracking = false;
	bool _isCaptureShown = false;

	static const char* WINDOW_NAME;

	cv::VideoCapture _cap;
	cv::Mat _capFrame; // 取得フレーム.
	cv::Mat _outputFrame; // 表示用(描き込み対象)

	cv::Mat _cameraMatrix;
	cv::Mat _distCoeffs;

	std::vector<int> _markerIds;
	std::vector<std::vector<cv::Point2f>> _markerCorners, _rejectedCandidates;
	cv::Ptr<cv::aruco::DetectorParameters> _parameters;
	cv::Ptr<cv::aruco::Dictionary> _dictionary;
	float _markerEdgeLength = 0.1f; // マーカーの辺の長さ. 単位はメートル

	std::map<int, std::vector<float>> _markerIdTRvec; // マーカーIDをキーにした tvec, rvec の6つ値を保持
	std::map<int, bool> _markerIdDetected; // マーカーIDをキーにした検出済フラグ

	cv::CascadeClassifier _faceCascade; // 顏検出器.
	cv::CascadeClassifier _eyesCascade; // 目検出器.

	dlib::frontal_face_detector _faceDetector;
	dlib::shape_predictor _facePredictor;

	bool _isFacePointsDetected = false;
	float _faceCircle[3]; // X,Y,半径
	float _leftEyeCircle[3];
	float _rightEyeCircle[3];
	float _leftIrisCircle[3];
	float _rightIrisCircle[3];

	int _irisThresh = 30;

	cv::Ptr<cv::BackgroundSubtractor> _handBackSub; // 手検出用の背景除去マスク
	float _minHandTranslationThreshold = 0.05f;
	float _maxHandTranslationThreshold = 3.0f;
	bool _isLeftHandDetected = false;
	bool _isRightHandDetected = false;
	bool _isLeftHandDown = false;
	bool _isRightHandDown = false;
	int _leftHandUndetectedTick = 0;
	int _rightHandUndetectedTick = 0;
	int _handUndetectedDuration = 5000; // ミリ秒
	float _leftHandCircle[3];
	float _rightHandCircle[3];
	const int HAND_RADIUS_BUF_SIZE = 3;
	std::list<float> _leftHandRadiusBuf; // 半径を遅延採用するために使う
	std::list<float> _rightHandRadiusBuf;

	int _errorCode = SMT_ERROR_NOEN;
};

const char* SimpleMotionTracker::WINDOW_NAME = "SimpleMotionTracker(Capture)";


void SimpleMotionTracker::initVideoDeviceList() {
	DeviceEnumerator de;
	std::map<int, Device> devices = de.getVideoDevicesMap();
}

void SimpleMotionTracker::init(const std::string& videoDeviceName) {
	int cameraId = -1;
	DeviceEnumerator de;
	std::map<int, Device> devices = de.getVideoDevicesMap();
	for (auto it : devices) {
		if (it.second.deviceName == videoDeviceName) {
			cameraId = it.first;
			break;
		}
	}
	init(cameraId);
}

void SimpleMotionTracker::init(int cameraId) {
	// デバイスを開く.
	_cap.open(cameraId);

	if (!_cap.isOpened()) {
		_errorCode = SMT_ERROR_UNOPEND_CAMERA;
		return;
	}

	if (!_cap.read(_capFrame)) {
		_errorCode = SMT_ERROR_UNREADABLE_CAMERA;
		return;
	}

	// キャプチャ速度計測50msec以下は非対応
	auto start = std::chrono::system_clock::now();
	_cap.read(_capFrame);
	auto end = std::chrono::system_clock::now();
	auto dur = end - start;
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	if (msec > 500) {
		_errorCode = SMT_ERROR_INSUFFICIENT_CAMERA_CAPTURE_SPEED;
		return;
	}

	if (!_faceCascade.load("data/haarcascade_frontalface_alt.xml")) {
		_errorCode = SMT_ERROR_UNOPEN_FACE_CASCADE;
		return;
	}

	if (!_eyesCascade.load("data/haarcascade_eye.xml")) {
		_errorCode = SMT_ERROR_UNOPEN_EYE_CASCADE;
		return;
	}

	// カメラパラメータ読み込み.
	loadCameraParam("data/camera_param.xml");

	//Dlibテスト
	{
		_faceDetector = dlib::get_frontal_face_detector();
		dlib::deserialize("data/sp_human_face_68_for_mobile.dat") >> _facePredictor;
	}

	// マーカー検出
	_markerIds.clear();
	_markerCorners.clear();
	_rejectedCandidates.clear();
	_parameters = cv::aruco::DetectorParameters::create();
	_dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
	for (int i = 0; i < 50; i++) {
		_markerIdTRvec[i].resize(6);
		_markerIdDetected[i] = false;
	}

	// 顏検出
	_isFacePointsDetected = false;
	for (int i = 0; i < 3; i++) {
		_faceCircle[i] = 0;
		_leftEyeCircle[i] = 0;
		_rightEyeCircle[i] = 0;
		_leftIrisCircle[i] = 0;
		_rightIrisCircle[i] = 0;
	}

	// 手検出
	//_handBackSub = cv::createBackgroundSubtractorMOG2();
	_handBackSub = cv::createBackgroundSubtractorKNN();
	_isLeftHandDetected = false;
	_isRightHandDetected = false;
	_leftHandUndetectedTick = 0;
	_rightHandUndetectedTick = 0;
	for (int i = 0; i < 3; i++) {
		_leftHandCircle[i] = 0;
		_rightHandCircle[i] = 0;
	}
	_leftHandRadiusBuf.clear();
	_rightHandRadiusBuf.clear();
	for (int i = 0; i < HAND_RADIUS_BUF_SIZE; i++) {
		_leftHandRadiusBuf.push_back(0);
		_rightHandRadiusBuf.push_back(0);
	}
}

void SimpleMotionTracker::destroy() {
	_cap.release(); // デストラクタで呼ばれるらしいけど明示的に開放.
	cv::destroyAllWindows();
}

void SimpleMotionTracker::update() {
	if (_errorCode != SMT_ERROR_NOEN) {
		return;
	}

	if (_cap.read(_capFrame)) {
		_outputFrame = _capFrame.clone();
		if (_useARMarker) {
			detectARMarker();
		}
		if (_useFaceTracking || _useEyesTracking) {
			//detectFace();
			detectFaceDlib();
		}
		if (_useHandTracking) {
			detectHand();
		}

		if (_isCaptureShown) {
			cv::imshow(WINDOW_NAME, _outputFrame);
		}
		else {
			cv::destroyWindow(WINDOW_NAME);
		}
		//cv::waitKey(1); // これが無いとWaitが無くて画面が表示されない
	}
}

void SimpleMotionTracker::setUseARMarker(bool useARMarker) {
	_useARMarker = useARMarker;
}

void SimpleMotionTracker::setUseFaceTracking(bool useFaceTracking) {
	_useFaceTracking = useFaceTracking;
}

void SimpleMotionTracker::setUseEyesTracking(bool useEyesTracking) {
	_useEyesTracking = useEyesTracking;
}

void SimpleMotionTracker::setUseHandTracking(bool useHandTracking) {
	_useHandTracking = useHandTracking;
}

void SimpleMotionTracker::setCaptureShown(bool isShown) {
	_isCaptureShown = isShown;
}

bool SimpleMotionTracker::isCaptureShown() {
	return _isCaptureShown;
}

int SimpleMotionTracker::getErrorCode() {
	return _errorCode;
}

void SimpleMotionTracker::loadCameraParam(const std::string& fileName) {
	cv::FileStorage fs(fileName, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		_errorCode = SMT_ERROR_UNOPEND_CAMERA_PARAM_FILE;
		return; // エラー
	}
	fs["camera_matrix"] >> _cameraMatrix;
	fs["distortion_coefficients"] >> _distCoeffs;
	fs.release();
}

cv::Rect SimpleMotionTracker::adjustRect(const cv::Rect& rect, const cv::Mat frame) {
	cv::Rect ret = rect;
	if (ret.x < 0) {
		ret.width += ret.x;
		ret.x = 0;
	}
	if (ret.y < 0) {
		ret.height += ret.y;
		ret.y = 0;
	}
	if (ret.x + ret.width >= frame.cols) {
		int remainder = (ret.x + ret.width) - frame.cols;
		ret.width -= remainder;
	}
	if (ret.y + ret.height >= frame.rows) {
		int remainder = (ret.y + ret.height) - frame.rows;
		ret.height -= remainder;
	}
	return ret;
}

void SimpleMotionTracker::detectARMarker() {
	for (int i = 0; i < 50; i++) {
		_markerIdDetected[i] = false; // 非検出状態へ
	}

	// マーカーを検出
	cv::aruco::detectMarkers(_capFrame, _dictionary, _markerCorners, _markerIds, _parameters, _rejectedCandidates, _cameraMatrix, _distCoeffs);

	// マーカー姿勢推定
	std::vector<cv::Vec3d> rvecs, tvecs;
	cv::aruco::estimatePoseSingleMarkers(_markerCorners, _markerEdgeLength, _cameraMatrix, _distCoeffs, rvecs, tvecs);

	for (int i = 0; i < rvecs.size(); ++i) {
		auto rvec = rvecs[i];
		auto tvec = tvecs[i];
		cv::aruco::drawAxis(_outputFrame, _cameraMatrix, _distCoeffs, rvec, tvec, 0.1);


		// オイラー角を得る
		cv::Vec3d eulerAngles;
		{
			cv::Mat rotMatrix;
			cv::Rodrigues(rvec, rotMatrix);

			double* _r = rotMatrix.ptr<double>();
			double projMatrix[12] = { _r[0],_r[1],_r[2],0,_r[3],_r[4],_r[5],0,_r[6],_r[7],_r[8],0 };
			cv::Mat projMat = cv::Mat(3, 4, CV_64FC1, projMatrix);

			cv::Mat cameraMatrix, rotation_matrix, translation_vector, rotMatrixX, rotMatrixY, rotMatrixZ;
			decomposeProjectionMatrix(projMat, cameraMatrix, rotation_matrix, translation_vector, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles);
		}

		auto id = _markerIds[i];
		_markerIdDetected[id] = true;

		_markerIdTRvec[id][0] = tvec[0];
		_markerIdTRvec[id][1] = tvec[1];
		_markerIdTRvec[id][2] = tvec[2];
		_markerIdTRvec[id][3] = eulerAngles[0];
		_markerIdTRvec[id][4] = eulerAngles[1];
		_markerIdTRvec[id][5] = eulerAngles[2];
	}

	// マーカー情報を書き込み
	cv::aruco::drawDetectedMarkers(_outputFrame, _markerCorners, _markerIds);
}

void SimpleMotionTracker::setARMarkerEdgeLength(float length) {
	_markerEdgeLength = length;
}

bool SimpleMotionTracker::isARMarkerDetected(int id) {
	return _markerIdDetected[id];
}

void SimpleMotionTracker::getARMarker6DoF(int id, float* outArray) {
	auto& vec = _markerIdTRvec[id];
	for (int i = 0; i < 6; i++) {
		outArray[i] = vec[i];
	}
}

void SimpleMotionTracker::detectIris(int irisThresh, cv::Mat faceFrame, cv::Rect eyeRect, cv::Rect& outIris) {
	if (eyeRect.width > 0) {
		auto eyeFrame = faceFrame(eyeRect);
		cv::equalizeHist(eyeFrame, eyeFrame);

		// 虹彩検出
		cv::equalizeHist(eyeFrame, eyeFrame);
		cv::threshold(eyeFrame, eyeFrame, irisThresh, 255, cv::THRESH_BINARY_INV);

		int morphSize = 1;
		cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * morphSize + 1, 2 * morphSize + 1), cv::Point(morphSize, morphSize));
		cv::morphologyEx(eyeFrame, eyeFrame, cv::MORPH_CLOSE, element);
		cv::morphologyEx(eyeFrame, eyeFrame, cv::MORPH_OPEN, element);
		cv::bitwise_not(eyeFrame, eyeFrame);

		cv::GaussianBlur(eyeFrame, eyeFrame, cv::Size(5, 5), 0);
		cv::threshold(eyeFrame, eyeFrame, 0, 255, cv::THRESH_BINARY);


		int cannyThresh = 100;
		cv::Mat cannyOutput;
		cv::Canny(eyeFrame, cannyOutput, cannyThresh, cannyThresh * 2);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(cannyOutput, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		std::vector<std::vector<cv::Point>> contoursPoly(contours.size());
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<cv::Point2f>centers(contours.size());
		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(contours[i], contoursPoly[i], 3, true);
			boundRect[i] = boundingRect(contoursPoly[i]);
		}

		cv::Rect max;
		for (auto rect : boundRect) {
			if (max.width < rect.width) {
				max = rect;
			}
		}
		outIris = max;
		//cv::rectangle(eyeFrame, max.tl(), max.br(), cv::Scalar(0, 0, 0), 2);
		//cv::imshow("test", eyeFrame);
	}
}

void SimpleMotionTracker::detectIrisDlib(int irisThresh, cv::Mat eyeFrame, cv::Rect eyeRect, cv::Rect& outIris) {
	if (eyeRect.width > 0) {
		// 虹彩検出
		cv::equalizeHist(eyeFrame, eyeFrame);
		cv::threshold(eyeFrame, eyeFrame, irisThresh, 255, cv::THRESH_BINARY_INV);

		int morphSize = 1;
		cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * morphSize + 1, 2 * morphSize + 1), cv::Point(morphSize, morphSize));
		cv::morphologyEx(eyeFrame, eyeFrame, cv::MORPH_CLOSE, element);
		cv::morphologyEx(eyeFrame, eyeFrame, cv::MORPH_OPEN, element);
		cv::bitwise_not(eyeFrame, eyeFrame);

		cv::GaussianBlur(eyeFrame, eyeFrame, cv::Size(5, 5), 0);
		cv::threshold(eyeFrame, eyeFrame, 0, 255, cv::THRESH_BINARY);


		int cannyThresh = 100;
		cv::Mat cannyOutput;
		cv::Canny(eyeFrame, cannyOutput, cannyThresh, cannyThresh * 2);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(cannyOutput, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		std::vector<std::vector<cv::Point>> contoursPoly(contours.size());
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<cv::Point2f>centers(contours.size());
		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(contours[i], contoursPoly[i], 3, true);
			boundRect[i] = boundingRect(contoursPoly[i]);
		}

		cv::Rect max;
		for (auto rect : boundRect) {
			if (max.width < rect.width) {
				max = rect;
			}
		}
		outIris = max;
		//cv::rectangle(eyeFrame, max.tl(), max.br(), cv::Scalar(0, 0, 0), 2);
		//cv::imshow("test", eyeFrame);
	}
}

void SimpleMotionTracker::detectMultiScaleRecursive(cv::CascadeClassifier& cascade, cv::Mat& frame, std::vector<cv::Rect>& outRects, cv::Size min, cv::Size max, int depth) {
	if (depth == 0) return;

	std::vector<cv::Rect> rects;
	cascade.detectMultiScale(frame, rects, 1.1, 3, 0, min, max);
	if (rects.size() > 0) {
		auto rect = rects[0];
		cv::Rect scaledRect = rect;
		float scale = 1.1f;
		scaledRect.width *= scale;
		scaledRect.height *= scale;
		scaledRect.x -= (scaledRect.width - rect.width) / 2;
		scaledRect.y -= (scaledRect.height - rect.height) / 2;
		scaledRect = adjustRect(scaledRect, frame);
		cv::Mat roi = frame(scaledRect);

		min.width = rect.width * 0.9f;
		min.height = rect.height * 0.9f;
		max.width = scaledRect.width;
		max.height = scaledRect.height;
		// 再帰的に処理
		detectMultiScaleRecursive(cascade, roi, outRects, min, max, depth - 1);
		if (outRects.size() == 0) {
			outRects.push_back(rect);
		}
		else {
			outRects[0].x += scaledRect.x;
			outRects[0].y += scaledRect.y;
		}
	}
}

void SimpleMotionTracker::detectFace() {
	bool prevIsFacePointsDetected = _isFacePointsDetected;
	_isFacePointsDetected = false;

	// 不可対策で検索範囲を絞る
	int minEdgeLen = (_faceCircle[2] * 2) * 0.9f; // 前回の1割りを最小サイズとする
	cv::Size minSize(minEdgeLen, minEdgeLen);

	float targetScale = 1.2f;
	cv::Rect targetRect(_faceCircle[0] - (_faceCircle[2] * targetScale), // 前回の少し大きめを対象とする 
		                _faceCircle[1] - (_faceCircle[2] * targetScale),
					    (_faceCircle[2] * targetScale) * 2,
						(_faceCircle[2] * targetScale) * 2);
	targetRect = adjustRect(targetRect, _capFrame);
	cv::Size maxSize(targetRect.width, targetRect.height);


	float resizeRate = 0.2f;

	cv::Mat frameGray;
	if (prevIsFacePointsDetected) {
		frameGray = _capFrame(targetRect);
		resizeRate *= _capFrame.rows / (_faceCircle[2] * 2); // 前回を元に縮小率を調整
		if (resizeRate > 1) { resizeRate = 1; }
		minSize.width *= resizeRate;
		minSize.height *= resizeRate;
		maxSize.width *= resizeRate;
		maxSize.height *= resizeRate;
	}
	else {
		frameGray = _capFrame.clone();
		minSize.width = 0;
		minSize.height = 0;
		maxSize.width = 0;
		maxSize.height = 0;
	}
	cv::cvtColor(frameGray, frameGray, cv::COLOR_BGR2GRAY);

	// 顏検出用に縮小画像で負荷軽減.
	cv::Mat faceDetectFrame;
	cv::resize(frameGray, faceDetectFrame, cv::Size(frameGray.cols * resizeRate, (frameGray.rows * resizeRate)));
	cv::equalizeHist(faceDetectFrame, faceDetectFrame);

	float detectedAngle = 0;
	for (auto angle : { 0, -20, 20, -40, 40 }) {
		// Z回転している顔を検出するために画像を回転させる.
		cv::Mat rotatedFaceDetectFrame;
		auto trans = cv::getRotationMatrix2D(cv::Point2f(faceDetectFrame.cols / 2, faceDetectFrame.rows / 2), angle, 1.0f);
		cv::warpAffine(faceDetectFrame, rotatedFaceDetectFrame, trans, cv::Size(faceDetectFrame.cols, faceDetectFrame.rows));

		std::vector<cv::Rect> faces; // 複数検出を考慮.
		//_faceCascade.detectMultiScale(rotatedFaceDetectFrame, faces, 1.1, 3, 0, minSize, maxSize);

		// 多段検出で精度を上げる
		int _faceDetectionLevel = 3;
		detectMultiScaleRecursive(_faceCascade, rotatedFaceDetectFrame, faces, minSize, maxSize, _faceDetectionLevel);

		cv::Rect face;
		cv::Rect leftEye;
		cv::Rect rightEye;
		cv::Rect leftIris;
		cv::Rect rightIris;

		for (auto f : faces) {
			f.x /= resizeRate;
			f.y /= resizeRate;
			f.width /= resizeRate;
			f.height /= resizeRate;
			if (f.width <= face.width) {
				continue;
			}
			face = f;

			// 目の検出.
			auto trans = cv::getRotationMatrix2D(cv::Point2f(frameGray.cols / 2, frameGray.rows / 2), angle, 1.0f);
			cv::warpAffine(frameGray, frameGray, trans, cv::Size(frameGray.cols, frameGray.rows));
			cv::Rect upperFace(f);
			int splitHeight = upperFace.height / 4;
			upperFace.height = splitHeight * 1.5f;
			upperFace.y += splitHeight;
			cv::Mat faceFrame = frameGray(upperFace).clone();
			cv::circle(faceFrame, cv::Point(faceFrame.cols / 2, faceFrame.rows), faceFrame.rows / 4, cv::Scalar(255, 255, 255), -1); // 鼻の孔隠し
			cv::equalizeHist(faceFrame, faceFrame);
			std::vector<cv::Rect> eyes;
			_eyesCascade.detectMultiScale(faceFrame, eyes);
			//cv::imshow("test", faceFrame);

			// 左右の目に振り分け.
			for (auto eye : eyes) {
				if ((eye.x + eye.width / 2) > (faceFrame.cols / 2)) {
					if (eye.width > leftEye.width) {
						leftEye = eye;
					}
				}
				else {
					if (eye.width > rightEye.width) {
						rightEye = eye;
					}
				}
			}
			// 検出できなかった時はそれっぽい場所を指定する
			auto adjustEyeRect = [faceFrame](cv::Rect& rect, int block) {
				int w = faceFrame.cols / 10;
				int h = faceFrame.rows / 10;
				rect.x = w * block;
				rect.y = h * 2;
				rect.width = w * 2;
				rect.height = h * 8;
				if (rect.height == 0) {
					rect.y = 0;
					rect.height = faceFrame.rows;
				}
			};
			if (leftEye.width == 0) {
				adjustEyeRect(leftEye, 6);
			}
			if (rightEye.width == 0) {
				adjustEyeRect(rightEye, 2);
			}
			detectedAngle = angle;

			// 虹彩検出
			if (_useEyesTracking) {
				faceFrame = frameGray(upperFace).clone();
				if (leftEye.width > 0) {
					detectIris(_irisThresh, faceFrame, leftEye, leftIris);
				}
				if (rightEye.width > 0) {
					detectIris(_irisThresh, faceFrame, rightEye, rightIris);
				}
			}

			break; // 検出は1つだけ前提で抜ける.
		}
		if (face.width > 0) {
			cv::Mat_<double> trans = cv::getRotationMatrix2D(cv::Point2f(0, 0), -detectedAngle, 1.0f);
			cv::Point2f frameCenter(frameGray.cols / 2, frameGray.rows / 2);

			auto afinTransform = [=](const cv::Rect& rect) {
				cv::Point2f center(rect.x + rect.width / 2, rect.y + rect.height / 2);
				float x = center.x - frameCenter.x;
				float y = center.y - frameCenter.y;
				float x2 = frameCenter.x + (trans(0, 0) * x + trans(0, 1) * y);
				float y2 = frameCenter.y + (trans(1, 0) * x + trans(1, 1) * y);
				if (prevIsFacePointsDetected) {
					x2 += targetRect.x;
					y2 += targetRect.y;
				}
				return cv::Point2f(x2, y2);
			};

			{
				auto p = afinTransform(face);
				cv::circle(_outputFrame, p, face.width / 2, cv::Scalar(255, 0, 255), 2);
				_faceCircle[0] = p.x;
				_faceCircle[1] = p.y;
				_faceCircle[2] = face.width / 2;
			}
			{
				cv::Rect upperFace(face);
				upperFace.height /= 4;
				upperFace.y += upperFace.height;
				if (leftEye.width > 0) {
					leftEye.x += upperFace.x;
					leftEye.y += upperFace.y;

					auto p = afinTransform(leftEye);
					cv::circle(_outputFrame, p, leftEye.width / 2, cv::Scalar(255, 255, 0), 2);
					_leftEyeCircle[0] = p.x;
					_leftEyeCircle[1] = p.y;
					_leftEyeCircle[2] = leftEye.width / 2;
				}
				if (rightEye.width > 0) {
					rightEye.x += upperFace.x;
					rightEye.y += upperFace.y;

					auto p = afinTransform(rightEye);
					cv::circle(_outputFrame, p, rightEye.width / 2, cv::Scalar(255, 255, 0), 2);
					_rightEyeCircle[0] = p.x;
					_rightEyeCircle[1] = p.y;
					_rightEyeCircle[2] = rightEye.width / 2;
				}
			}
			{
				if (leftEye.width > 0 && leftIris.width > 0) {
					leftIris.x += leftEye.x;
					leftIris.y += leftEye.y;

					auto p = afinTransform(leftIris);
					cv::circle(_outputFrame, p, leftIris.height / 2, cv::Scalar(0, 0, 255), 2);
					_leftIrisCircle[0] = p.x;
					_leftIrisCircle[1] = p.y;
					_leftIrisCircle[2] = (float)(leftIris.height) / leftIris.width;
				}
				if (rightEye.width > 0 && rightIris.width > 0) {
					rightIris.x += rightEye.x;
					rightIris.y += rightEye.y;

					auto p = afinTransform(rightIris);
					cv::circle(_outputFrame, p, rightIris.height / 2, cv::Scalar(0, 0, 255), 2);
					_rightIrisCircle[0] = p.x;
					_rightIrisCircle[1] = p.y;
					_rightIrisCircle[2] = (float)(rightIris.height) / rightIris.width;
				}
			}

			_isFacePointsDetected = true;
			//cv::imshow("test", frameGray);
			return;
		}
	}
}

void SimpleMotionTracker::detectFaceDlib() {
	_isFacePointsDetected = false;

	cv::Mat temp = _capFrame.clone();
	dlib::cv_image<dlib::bgr_pixel> cimg(temp);
	std::vector<dlib::rectangle> faces = _faceDetector(cimg);
	if (faces.size() > 0) {
		auto face = faces[0];
		dlib::full_object_detection shape = _facePredictor(cimg, face);
		auto minPoint = cv::Point(9999, 9999);
		auto maxPoint = cv::Point(0, 0);
		for (int i = 0; i < shape.num_parts(); i++)
		{
			if (shape.part(i).x() < minPoint.x) { minPoint.x = shape.part(i).x(); }
			if (shape.part(i).x() > maxPoint.x) { maxPoint.x = shape.part(i).x(); }
			if (shape.part(i).y() < minPoint.y) { minPoint.y = shape.part(i).y(); }
			if (shape.part(i).y() > maxPoint.y) { maxPoint.y = shape.part(i).y(); }
			cv::circle(_outputFrame, cv::Point2d(shape.part(i).x(), shape.part(i).y()), 3, cv::Scalar(0, 255, 0), -1);
		}

		{ // 顏の位置
			auto faceRect = cv::Rect(minPoint.x, minPoint.y, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y);
			auto p = cv::Point(faceRect.x + faceRect.width / 2, faceRect.y + faceRect.height / 2);
			auto r = (faceRect.width + faceRect.height) / 2 / 2; // 平均の半分
			cv::circle(_outputFrame, p, r, cv::Scalar(255, 0, 255), 2);
			_faceCircle[0] = p.x;
			_faceCircle[1] = p.y;
			_faceCircle[2] = r;
		}

		{ // 左目の位置
			auto& p1 = shape.part(42);
			auto& p2 = shape.part(45);
			auto p = cv::Point(
				(p1.x() + p2.x()) / 2,
				(p1.y() + p2.y()) / 2);
			auto toX = p2.x() - p1.x();
			auto toY = p2.y() - p1.y();
			auto r = std::sqrtf((toX * toX) + (toY * toY)) / 2;
			cv::circle(_outputFrame, p, r, cv::Scalar(255, 255, 0), 2);
			_leftEyeCircle[0] = p.x;
			_leftEyeCircle[1] = p.y;
			_leftEyeCircle[2] = r;
		}

		{ // 右目の位置
			auto& p1 = shape.part(36);
			auto& p2 = shape.part(39);
			auto p = cv::Point(
				(p1.x() + p2.x()) / 2,
				(p1.y() + p2.y()) / 2);
			auto toX = p2.x() - p1.x();
			auto toY = p2.y() - p1.y();
			auto r = std::sqrtf((toX * toX) + (toY * toY)) / 2;
			cv::circle(_outputFrame, p, r, cv::Scalar(255, 255, 0), 2);
			_rightEyeCircle[0] = p.x;
			_rightEyeCircle[1] = p.y;
			_rightEyeCircle[2] = r;
		}

		// 虹彩検出
		if (_useEyesTracking) {
			cv::Rect leftIris;
			cv::Rect rightIris;

			cv::Mat frameGray;
			cv::cvtColor(_capFrame, frameGray, cv::COLOR_BGR2GRAY);

			{ // 左虹彩
				auto eyeRect = cv::Rect(
					_leftEyeCircle[0] - _leftEyeCircle[2],
					_leftEyeCircle[1] - _leftEyeCircle[2],
					_leftEyeCircle[2] * 2,
					_leftEyeCircle[2] * 2);
				if (eyeRect.x >= 0 &&
					eyeRect.y >= 0 &&
					eyeRect.x + eyeRect.width < frameGray.cols &&
					eyeRect.y + eyeRect.height < frameGray.rows) {
					cv::Mat eyeFrame = frameGray(eyeRect);
					detectIrisDlib(_irisThresh, eyeFrame, eyeRect, leftIris);

					if (eyeRect.width > 0 && leftIris.width > 0) {
						leftIris.x += eyeRect.x;
						leftIris.y += eyeRect.y;

						auto& p1 = shape.part(43);
						auto& p2 = shape.part(47);
						auto toX = p2.x() - p1.x();
						auto toY = p2.y() - p1.y();
						auto vertical = std::sqrtf((toX * toX) + (toY * toY));
						auto horizontal = _leftEyeCircle[2];

						auto p = cv::Point(leftIris.x + leftIris.width / 2, leftIris.y + leftIris.height / 2);
						cv::circle(_outputFrame, p, leftIris.height / 2, cv::Scalar(0, 0, 255), 2);
						_leftIrisCircle[0] = p.x;
						_leftIrisCircle[1] = p.y;
						_leftIrisCircle[2] = vertical / horizontal;
					}
				}
			}

			{ // 右虹彩
				auto eyeRect = cv::Rect(
					_rightEyeCircle[0] - _rightEyeCircle[2],
					_rightEyeCircle[1] - _rightEyeCircle[2],
					_rightEyeCircle[2] * 2,
					_rightEyeCircle[2] * 2);
				if (eyeRect.x >= 0 &&
					eyeRect.y >= 0 &&
					eyeRect.x + eyeRect.width < frameGray.cols &&
					eyeRect.y + eyeRect.height < frameGray.rows) {
					cv::Mat eyeFrame = frameGray(eyeRect);
					detectIrisDlib(_irisThresh, eyeFrame, eyeRect, rightIris);

					if (eyeRect.width > 0 && rightIris.width > 0) {
						rightIris.x += eyeRect.x;
						rightIris.y += eyeRect.y;

						auto& p1 = shape.part(38);
						auto& p2 = shape.part(40);
						auto toX = p2.x() - p1.x();
						auto toY = p2.y() - p1.y();
						auto vertical = std::sqrtf((toX * toX) + (toY * toY));
						auto horizontal = _rightEyeCircle[2];


						auto p = cv::Point(rightIris.x + rightIris.width / 2, rightIris.y + rightIris.height / 2);
						cv::circle(_outputFrame, p, rightIris.height / 2, cv::Scalar(0, 0, 255), 2);
						_rightIrisCircle[0] = p.x;
						_rightIrisCircle[1] = p.y;
						_rightIrisCircle[2] = vertical / horizontal;
					}
				}
			}
		}

		_isFacePointsDetected = true;
		return;
	}
}

bool SimpleMotionTracker::isFacePointsDetected() {
	return _isFacePointsDetected;
}

void SimpleMotionTracker::getFacePoints(float* outArray) {
	outArray[0] = _faceCircle[0];
	outArray[1] = _faceCircle[1];
	outArray[2] = _faceCircle[2];
	outArray[3] = _leftEyeCircle[0];
	outArray[4] = _leftEyeCircle[1];
	outArray[5] = _leftEyeCircle[2];
	outArray[6] = _rightEyeCircle[0];
	outArray[7] = _rightEyeCircle[1];
	outArray[8] = _rightEyeCircle[2];
	outArray[9]  = _leftIrisCircle[0];
	outArray[10] = _leftIrisCircle[1];
	outArray[11] = _leftIrisCircle[2];
	outArray[12] = _rightIrisCircle[0];
	outArray[13] = _rightIrisCircle[1];
	outArray[14] = _rightIrisCircle[2];
}

void SimpleMotionTracker::detectHandCircle(float* handCircle, bool& isHandDetected, bool& isHandDown, int& handUndetectedTick, const cv::Point& point, std::vector<cv::Point>& points, float minHandRadius, float maxHandRadius, std::list<float>& radiusBuf, float resizeRatio, cv::Mat frame) {
	bool isDetected = isHandDetected;
	int currentTickCount = GetTickCount();
	int count = points.size();

	// 最小半径時の円の面積と動体検知量比較する
	// 非検知状態から検知状態への移行は大きく動く必要がある.
	// 一度検知が途切れても一定時間内であれば低い閾値で再検知する
	float thresholdRatio = _minHandTranslationThreshold;
	if (!isDetected) {
		if (currentTickCount > handUndetectedTick + _handUndetectedDuration) {
			thresholdRatio = _maxHandTranslationThreshold;
		}
	}
	thresholdRatio = thresholdRatio * resizeRatio * resizeRatio;
	int translationThrashold = (minHandRadius * minHandRadius * 3.14f) * thresholdRatio;
	if (count > translationThrashold) {
		isDetected = true;
	}
	else {
		isDetected = false;

		//半径バッファをリセットして先頭の値で埋めなおす
		float radius = radiusBuf.front();
		radiusBuf.clear();
		for (int i = 0; i < HAND_RADIUS_BUF_SIZE; i++) { radiusBuf.push_back(radius); };
	}

	// 半径の計算
	if (isDetected) {
		float radius = 0;
		int rx = 0;
		int ry = 0;
		// Z値として使うために半径を計算
		// 平均位置から各動体検知座標へのX,Y成分ごとの平均値も計算

		for (int i = 0; i < count; i++) {
			auto x = points[i].x - point.x;
			auto y = points[i].y - point.y;
			radius += std::sqrtf((x * x) + (y * y));
			rx += std::abs(x);
			ry += std::abs(y);
		}
		radius /= count;
		rx /= count;
		ry /= count;

		// X,Y成分の比率が1から離れると、円形ではなく長方形の動体が検知されたことになる.
		// 半径をそのまま使うと長い辺の影響を受けるので、短い辺に合わせて補正する.
		if (rx > 0 && ry > 0) {
			float ratio = (float)rx / ry;
			if (ratio > 1) { ratio = 1.0f / ratio; }
			radius *= ratio;
		}

		// 半径を遅延採用する
		// 動体検知では検出できなくなる直前は対象を小さく判定してしまうため.
		radiusBuf.push_back(radius);
		radius = radiusBuf.front();
		radiusBuf.pop_front();

		if (radius < minHandRadius) {
			radius = minHandRadius;
		}
		else if (radius > maxHandRadius) {
			radius = maxHandRadius;
		}

		handCircle[0] = point.x;
		handCircle[1] = point.y;
		handCircle[2] = radius;
		handUndetectedTick = currentTickCount;

		isHandDown = false;

		cv::circle(_outputFrame, point, radius, cv::Scalar(0, 255, 0), 2);

		//{
		//	static float radiusAve = 0;
		//	radiusAve = radiusAve * 0.9 + radius * 0.1;
		//	printf("%f\n", radiusAve / minHandRadius);
		//	//printf("%f\n", radius);
		//}
	}
	isHandDetected = isDetected;
}

void SimpleMotionTracker::detectHand() {
	cv::Mat handFrame = _capFrame.clone();
	cv::Mat handFrameGray;
	cv::cvtColor(handFrame, handFrameGray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(handFrameGray, handFrameGray);

	auto faceCenter = cv::Point(0, 0);
	auto faceRadius = 0;
	if (_useFaceTracking) {
		faceCenter.x = _faceCircle[0];
		faceCenter.y = _faceCircle[1];
		faceRadius = _faceCircle[2];
	}
	else {
		std::vector<cv::Rect> faces; // 複数検出を考慮.
		cv::Mat resizedFrame;
		const float resizeRate = 0.2f;
		cv::resize(handFrameGray, resizedFrame, cv::Size(handFrameGray.cols * resizeRate, (handFrameGray.rows * resizeRate)));
		_faceCascade.detectMultiScale(resizedFrame, faces);
		if (faces.size() > 0) {
			auto f = faces[0];
			f.x /= resizeRate;
			f.y /= resizeRate;
			f.width /= resizeRate;
			f.height /= resizeRate;
			faceCenter = cv::Point(f.x + f.width / 2, f.y + f.height / 2);
			faceRadius = f.width / 2;
			_faceCircle[0] = faceCenter.x;
			_faceCircle[1] = faceCenter.y;
			_faceCircle[2] = faceRadius;
		}
	}
	// 例外対策
	if (faceRadius == 0) {
		faceRadius = 100;
	}

	int minHandRadius = (faceRadius * 0.25f); // 拳の大きさは顏の半分だが、動体検知平均はさらに半分になるので 1/4を設定 
	int maxHandRadius = faceRadius;

	// 顔を見つけられなかった場合は古い情報を拡大して使う
	if (faceRadius == 0) {
		faceCenter.x = _faceCircle[0];
		faceCenter.y = _faceCircle[1];
		faceRadius = _faceCircle[2] * 1.5f;
	}

	int originalFaceRadius = faceRadius;
	int shoulderY = faceCenter.y + (faceRadius * 1.5f); // 肩の位置推定
	int handShutterMaskMargin = faceRadius * 2;

	float resizeRatio = 1.0f;
	cv::Mat handBSMask;
	cv::resize(handFrameGray, handFrameGray, cv::Size(handFrameGray.cols * resizeRatio, handFrameGray.rows * resizeRatio));
	_handBackSub->apply(handFrameGray, handBSMask, 0.999);

	faceCenter.y -= (faceRadius / 2);
	faceRadius *= 1.5f;

	faceCenter.x *= resizeRatio;
	faceCenter.y *= resizeRatio;
	faceRadius *= resizeRatio;

	cv::circle(handBSMask, faceCenter, faceRadius, cv::Scalar(0, 0, 0), -1);
	cv::circle(handBSMask, cv::Point(faceCenter.x, faceCenter.y + faceRadius), faceRadius * 0.5f, cv::Scalar(0, 0, 0), -1);

	// 手前の数フレーム分も塗りつぶす
	static float prevFCx[3] = { 0 };
	static float prevFCy[3] = { 0 };
	static float prevFCr[3] = { 0 };

	for (int i = 0; i < 3; i++) {
		cv::circle(handBSMask, cv::Point(prevFCx[i], prevFCy[i]), prevFCr[i], cv::Scalar(0, 0, 0), -1);
		cv::circle(handBSMask, cv::Point(prevFCx[i], prevFCy[i] + prevFCr[i]), prevFCr[i] * 0.5f, cv::Scalar(0, 0, 0), -1);
	}
	for (int i = 3 - 1; i > 0; i--) {
		prevFCx[i] = prevFCx[i-1];
		prevFCy[i] = prevFCy[i-1];
		prevFCr[i] = prevFCr[i-1];
	}
	prevFCx[0] = faceCenter.x;
	prevFCy[0] = faceCenter.y;
	prevFCr[0] = faceRadius;

	// 頭の移動量が多い場合、胸周辺を塗りつぶして検知されないようにする
	if (true) {
		float vx = prevFCx[0] - prevFCx[2];
		float vy = prevFCy[0] - prevFCy[2];
		float r = ((prevFCr[0] + prevFCr[2] + prevFCr[2]) / 3) / 1.5f; // 半径は平均値を使う
		float len = std::sqrtf((vx * vx) + (vy * vy));
		float ratio = len / r;

		if (ratio > 0.1f) {	
			cv::Point center(prevFCx[0], prevFCy[0]);
			int height = handBSMask.rows - center.y;
			int halfWidth = prevFCr[0] + (len * 5);
			cv::Rect bodyRect(center.x - halfWidth, center.y, halfWidth * 2, height);
			cv::rectangle(handBSMask, bodyRect, cv::Scalar(0), -1);
		}
	}

	// 前回の手の位置＋顔の直径より下は塗りつぶす
	if (_isRightHandDetected) {
		cv::Rect rightHandUnderMask;
		rightHandUnderMask.x = 0;
		rightHandUnderMask.y = (_rightHandCircle[1] + handShutterMaskMargin) * resizeRatio;
		if (rightHandUnderMask.y >= handBSMask.rows) { rightHandUnderMask.y = handBSMask.rows - 2; }
		rightHandUnderMask.width = faceCenter.x;
		rightHandUnderMask.height = handBSMask.rows - rightHandUnderMask.y;
		cv::rectangle(handBSMask, rightHandUnderMask, cv::Scalar(0, 0, 0), -1);
		//cv::rectangle(_outputFrame, rightHandUnderMask, cv::Scalar(0, 0, 0), -1); // 位置確認用
	}
	if (_isLeftHandDetected) {
		cv::Rect leftHandUnderMask;
		leftHandUnderMask.x = faceCenter.x;
		leftHandUnderMask.y = (_leftHandCircle[1] + handShutterMaskMargin) * resizeRatio;
		if (leftHandUnderMask.y >= handBSMask.rows) { leftHandUnderMask.y = handBSMask.rows - 2; }
		leftHandUnderMask.width = handBSMask.cols - leftHandUnderMask.x;
		leftHandUnderMask.height = handBSMask.rows - leftHandUnderMask.y;
		cv::rectangle(handBSMask, leftHandUnderMask, cv::Scalar(0, 0, 0), -1);
		//cv::rectangle(_outputFrame, leftHandUnderMask, cv::Scalar(0, 0, 0), -1); // 位置確認用
	}

	// 検出された部分を一回り小さくする
	{
		int erosion_size = 1;
		int erosion_type = 0;
		erosion_type = cv::MORPH_RECT;
		//erosion_type = cv::MORPH_CROSS;
		//erosion_type = cv::MORPH_ELLIPSE;
		cv::Mat element = getStructuringElement(erosion_type,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		cv::erode(handBSMask, handBSMask, element);
		// 小さくした後拡大
		//cv::dilate(handBSMask, handBSMask, element);
	}

	// 孤立している小さな検出点(ノイズ)を除去
	{
		int morphSize = 1;
		cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * morphSize + 1, 2 * morphSize + 1), cv::Point(morphSize, morphSize));
		cv::morphologyEx(handBSMask, handBSMask, cv::MORPH_OPEN, element);
		cv::threshold(handBSMask, handBSMask, 0, 255, cv::THRESH_BINARY);
	}

	std::vector<cv::Point> leftPoints;
	std::vector<cv::Point> rightPoints;
	int leftCount = 0;
	int rightCount = 0;
	cv::Point leftPoint, rightPoint;
	for (int y = 0; y < handBSMask.rows; y++) {
		for (int x = 0; x < handBSMask.cols; x++) {
			auto c = handBSMask.at<uchar>(y, x);
			if (c > 0) {
				if (x > faceCenter.x) { // 画面右側（左手）
					leftCount++;
					leftPoint.x += x / resizeRatio;
					leftPoint.y += y / resizeRatio;

					leftPoints.push_back(cv::Point( x / resizeRatio, y / resizeRatio));
				}
				else {
					rightCount++;
					rightPoint.x += x / resizeRatio;
					rightPoint.y += y / resizeRatio;

					rightPoints.push_back(cv::Point(x / resizeRatio, y / resizeRatio));
				}
			}
		}
	}
	// 座標の平均と半径を計算
	if (leftCount > 0) {
		if (leftPoints[0].y + _leftHandCircle[2] > shoulderY) {
			leftPoint.x /= leftCount;
			leftPoint.y /= leftCount;
		}
		else {
			leftPoint.x /= leftCount;
			leftPoint.y = leftPoints[0].y + (originalFaceRadius / 2);
			//leftPoint.y += +_leftHandCircle[2];
		}
		detectHandCircle(_leftHandCircle, _isLeftHandDetected, _isLeftHandDown, _leftHandUndetectedTick, leftPoint, leftPoints, minHandRadius, maxHandRadius, _leftHandRadiusBuf, resizeRatio, handBSMask);
	}
	else {
		_isLeftHandDetected = false;
		// 非検知状態 かつ 画面下部で見失っているときは手を下げたと判定
		if (_leftHandCircle[1] + _leftHandCircle[2] >= handFrame.rows / 15 * 14) {
			_isLeftHandDown = true;
			if (_leftHandUndetectedTick > _handUndetectedDuration) { _leftHandUndetectedTick -= _handUndetectedDuration; }
			//cv::rectangle(_outputFrame, cv::Rect(0, 0, 30, 30), cv::Scalar(0, 0, 255), -1); // 判定確認用
		}
	}
	if (rightCount > 0) {
		if (rightPoints[0].y + _rightHandCircle[2] > shoulderY) {
			rightPoint.x /= rightCount;
			rightPoint.y /= rightCount;
		}
		else {
			rightPoint.x /= rightCount;
			rightPoint.y = rightPoints[0].y + (originalFaceRadius / 2);
			//rightPoint.y += +_rightHandCircle[2];
		}
		detectHandCircle(_rightHandCircle, _isRightHandDetected, _isRightHandDown, _rightHandUndetectedTick, rightPoint, rightPoints, minHandRadius, maxHandRadius, _rightHandRadiusBuf, resizeRatio, handBSMask);
	}
	else {
		_isRightHandDetected = false;
		// 非検知状態 かつ 画面下部で見失っているときは手を下げたと判定
		if (_rightHandCircle[1] + _rightHandCircle[2] >= handFrame.rows / 15 * 14) {
			_isRightHandDown = true;
			if (_rightHandUndetectedTick > _handUndetectedDuration) { _rightHandUndetectedTick -= _handUndetectedDuration; }
		}
	}
	cv::imshow("Live", handBSMask);
}

bool SimpleMotionTracker::isLeftHandDetected() {
	return _isLeftHandDetected;
}

bool SimpleMotionTracker::isRightHandDetected() {
	return _isRightHandDetected;
}

bool SimpleMotionTracker::isLeftHandDown() {
	return _isLeftHandDown;
}

bool SimpleMotionTracker::isRightHandDown() {
	return _isRightHandDown;
}

void SimpleMotionTracker::getHandPoints(float* outArray) {
	outArray[0] = _leftHandCircle[0];
	outArray[1] = _leftHandCircle[1];
	outArray[2] = _leftHandCircle[2];
	outArray[3] = _rightHandCircle[0];
	outArray[4] = _rightHandCircle[1];
	outArray[5] = _rightHandCircle[2];
}

/*----------------------------------------------------------
                         DLL用API
----------------------------------------------------------*/
static SimpleMotionTracker* instance = nullptr;

void SMT_initRaw(int cameraId) {
	if (instance == nullptr) {
		instance = new SimpleMotionTracker();
	}
	instance->init(cameraId);
}

void SMT_init(const char* videoDeviceName) {
	if (instance ==nullptr) {
		instance = new SimpleMotionTracker();
	}
	std::string name = videoDeviceName;
	instance->init(name);
}

void SMT_destroy() {
	if (instance == nullptr) return;
	instance->destroy();
	delete instance;
	instance = nullptr;
}

void SMT_update() {
	if (instance == nullptr) return;
	instance->update();
}

void SMT_setUseARMarker(bool useARMarker) {
	if (instance == nullptr) return;
	instance->setUseARMarker(useARMarker);
}

void SMT_setUseFaceTracking(bool useFaceTracking) {
	if (instance == nullptr) return;
	instance->setUseFaceTracking(useFaceTracking);
}

void SMT_setUseEyeTracking(bool useEyesTracking) {
	if (instance == nullptr) return;
	instance->setUseEyesTracking(useEyesTracking);
}

void SMT_setUseHandTracking(bool useHandTracking) {
	if (instance == nullptr) return;
	instance->setUseHandTracking(useHandTracking);
}

void SMT_setCaptureShown(bool isShown) {
	if (instance == nullptr) return;
	instance->setCaptureShown(isShown);
}

bool SMT_isCaptureShown() {
	if (instance == nullptr) return false;
	return instance->isCaptureShown();
}

void SMT_setARMarkerEdgeLength(float length) {
	if (instance == nullptr) return;
	instance->setARMarkerEdgeLength(length);
}

bool SMT_isARMarkerDetected(int id) {
	if (instance == nullptr) return false;
	return instance->isARMarkerDetected(id);
}

void SMT_getARMarker6DoF(int id, float* outArray) {
	if (instance == nullptr) return;
	instance->getARMarker6DoF(id, outArray);
}

bool SMT_isFacePointsDetected() {
	if (instance == nullptr) return false;
	return instance->isFacePointsDetected();
}

void SMT_getFacePoints(float* outArray) {
	if (instance == nullptr) return;
	instance->getFacePoints(outArray);
}

void SMT_setIrisThresh(int thresh) {
	if (instance == nullptr) return;
	instance->setIrisThresh(thresh);
}

void SMT_setMinHandTranslationThreshold(float thresh) {
	if (instance == nullptr) return;
	instance->setMinHandTranslationThreshold(thresh);
}

void SMT_setMaxHandTranslationThreshold(float thresh) {
	if (instance == nullptr) return;
	instance->setMaxHandTranslationThreshold(thresh);
}

void SMT_setHandUndetectedDuration(int msec) {
	if (instance == nullptr) return;
	instance->setHandUndetectedDuration(msec);
}

bool SMT_isLeftHandDetected() {
	if (instance == nullptr) return false;
	return instance->isLeftHandDetected();
}

bool SMT_isRightHandDetected() {
	if (instance == nullptr) return false;
	return instance->isRightHandDetected();
}

bool SMT_isLeftHandDown() {
	if (instance == nullptr) return false;
	return instance->isLeftHandDown();
}

bool SMT_isRightHandDown() {
	if (instance == nullptr) return false;
	return instance->isRightHandDown();
}

void SMT_getHandPoints(float* outArray) {
	if (instance == nullptr) return;
	instance->getHandPoints(outArray);
}

void SMT_cvWait() {
	cv::waitKey(1);
}

int SMT_getErrorCode() {
	if (instance == nullptr) return SMT_ERROR_NOEN;
	return instance->getErrorCode();
}
