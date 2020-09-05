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

	void setCaptureShown(bool isShown);
	bool isCaptureShown();

	void setIrisThresh(int thresh) { _irisThresh = thresh; }

	void setARMarkerEdgeLength(float length);
	bool isARMarkerDetected(int id);
	void getARMarker6DoF(int id, float* outArray);

	bool isFacePointsDetected();
	void getFacePoints(float* outArray);



	int getErrorCode();

private:
	void loadCameraParam(const std::string& fileName);
	void detectARMarker();

	void detectFace();
	void detectIris(int irisThresh, cv::Mat faceFrame, cv::Rect eyeRect, cv::Rect& outIris);

	bool _useARMarker = false;
	bool _useFaceTracking = false;
	bool _useEyesTracking = false;
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

	bool _isFacePointsDetected = false;
	float _faceCircle[3]; // X,Y,半径
	float _leftEyeCircle[3];
	float _rightEyeCircle[3];
	float _leftIrisCircle[3];
	float _rightIrisCircle[3];

	int _irisThresh = 30;


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
			detectFace();
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

void SimpleMotionTracker::detectFace() {
	_isFacePointsDetected = false;

	cv::Mat frameGray;
	cv::cvtColor(_capFrame, frameGray, cv::COLOR_BGR2GRAY);

	// 顏検出用に縮小画像で負荷軽減.
	cv::Mat faceDetectFrame;
	const float resizeRate = 0.2f;
	cv::resize(frameGray, faceDetectFrame, cv::Size(frameGray.cols * resizeRate, (frameGray.rows * resizeRate)));
	cv::equalizeHist(faceDetectFrame, faceDetectFrame);

	float detectedAngle = 0;
	for (auto angle : { 0, -20, 20, -40, 40 }) {
		// Z回転している顔を検出するために画像を回転させる.
		cv::Mat rotatedFaceDetectFrame;
		auto trans = cv::getRotationMatrix2D(cv::Point2f(faceDetectFrame.cols / 2, faceDetectFrame.rows / 2), angle, 1.0f);
		cv::warpAffine(faceDetectFrame, rotatedFaceDetectFrame, trans, cv::Size(faceDetectFrame.cols, faceDetectFrame.rows));

		std::vector<cv::Rect> faces; // 複数検出を考慮.
		_faceCascade.detectMultiScale(rotatedFaceDetectFrame, faces);

		cv::Rect face;
		cv::Rect leftEye;
		cv::Rect rightEye;
		cv::Rect leftIris;
		cv::Rect rightIris;

		for (auto f : faces) {
			f.x *= 5;
			f.y *= 5;
			f.width *= 5;
			f.height *= 5;
			if (f.width <= face.width) {
				continue;
			}
			face = f;

			// 目の検出.
			auto trans = cv::getRotationMatrix2D(cv::Point2f(frameGray.cols / 2, frameGray.rows / 2), angle, 1.0f);
			cv::warpAffine(frameGray, frameGray, trans, cv::Size(frameGray.cols, frameGray.rows));
			cv::Rect upperFace(f);
			upperFace.height /= 4;
			upperFace.y += upperFace.height;
			cv::Mat faceFrame = frameGray(upperFace).clone();
			cv::equalizeHist(faceFrame, faceFrame);
			std::vector<cv::Rect> eyes;
			_eyesCascade.detectMultiScale(faceFrame, eyes);

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
		}

		if (face.width > 0) {
			cv::Mat_<double> trans = cv::getRotationMatrix2D(cv::Point2f(0,0), -detectedAngle, 1.0f);
			cv::Point2f frameCenter(frameGray.cols / 2, frameGray.rows / 2);

			auto afinTransform = [=](const cv::Rect& rect) {
				cv::Point2f center(rect.x + rect.width / 2, rect.y + rect.height / 2);
				float x = center.x - frameCenter.x;
				float y = center.y - frameCenter.y;
				float x2 = frameCenter.x + (trans(0, 0) * x + trans(0, 1) * y);
				float y2 = frameCenter.y + (trans(1, 0) * x + trans(1, 1) * y);
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

void SMT_cvWait() {
	cv::waitKey(1);
}

int SMT_getErrorCode() {
	if (instance == nullptr) return SMT_ERROR_NOEN;
	return instance->getErrorCode();
}
