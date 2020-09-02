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
	void setUseEyeTracking(bool useEyeTracking);

	void setCaptureShown(bool isShown);
	bool isCaptureShown();

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

	bool _useARMarker = false;
	bool _useFaceTracking = false;
	bool _useEyeTracking = false;
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
	if (msec > 50) {
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
		if (_useFaceTracking) {
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

void SimpleMotionTracker::setUseEyeTracking(bool useEyeTracking) {
	_useEyeTracking = useEyeTracking;
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

void SimpleMotionTracker::detectFace() {
	_isFacePointsDetected = false;

	cv::Mat frameGray;
	cv::cvtColor(_capFrame, frameGray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(frameGray, frameGray);

	// 顏検出用に縮小画像で負荷軽減.
	cv::Mat faceDetectFrame;
	const float resizeRate = 0.2f;
	cv::resize(frameGray, faceDetectFrame, cv::Size(frameGray.cols * resizeRate, (frameGray.rows * resizeRate)));

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
			cv::Mat faceFrame = frameGray(upperFace);
			std::vector<cv::Rect> eyes;
			_eyesCascade.detectMultiScale(faceFrame, eyes);

			// 左右の目に振り分け.
			for (auto eye : eyes) {
				if ((eye.x + eye.width / 2) < (faceFrame.cols / 2)) {
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
		}

		if (face.width > 0) {
			cv::Mat_<double> trans = cv::getRotationMatrix2D(cv::Point2f(0,0), -detectedAngle, 1.0f);
			cv::Point2f frameCenter(frameGray.cols / 2, frameGray.rows / 2);
			{
				cv::Point2f faceCenter(face.x + face.width / 2, face.y + face.height / 2);
				float x = faceCenter.x - frameCenter.x;
				float y = faceCenter.y - frameCenter.y;
				float x2 = frameCenter.x + (trans(0, 0) * x + trans(0, 1) * y);
				float y2 = frameCenter.y + (trans(1, 0) * x + trans(1, 1) * y);
				cv::circle(_outputFrame, cv::Point(x2, y2), face.width / 2, cv::Scalar(255, 0, 255), 2);
				_faceCircle[0] = x2;
				_faceCircle[1] = y2;
				_faceCircle[2] = face.width / 2;
			}
			{
				cv::Rect upperFace(face);
				upperFace.height /= 4;
				upperFace.y += upperFace.height;
				if (leftEye.width > 0) {
					leftEye.x += upperFace.x;
					leftEye.y += upperFace.y;

					cv::Point2f eyeCenter(leftEye.x + leftEye.width / 2, leftEye.y + leftEye.height / 2);
					float x = eyeCenter.x - frameCenter.x;
					float y = eyeCenter.y - frameCenter.y;
					float x2 = frameCenter.x + (trans(0, 0) * x + trans(0, 1) * y);
					float y2 = frameCenter.y + (trans(1, 0) * x + trans(1, 1) * y);
					cv::circle(_outputFrame, cv::Point(x2, y2), leftEye.width / 2, cv::Scalar(255, 255), 2);
					_leftEyeCircle[0] = x2;
					_leftEyeCircle[1] = y2;
					_leftEyeCircle[2] = leftEye.width / 2;
				}
				if (rightEye.width > 0) {
					rightEye.x += upperFace.x;
					rightEye.y += upperFace.y;

					cv::Point2f eyeCenter(rightEye.x + rightEye.width / 2, rightEye.y + rightEye.height / 2);
					float x = eyeCenter.x - frameCenter.x;
					float y = eyeCenter.y - frameCenter.y;
					float x2 = frameCenter.x + (trans(0, 0) * x + trans(0, 1) * y);
					float y2 = frameCenter.y + (trans(1, 0) * x + trans(1, 1) * y);
					cv::circle(_outputFrame, cv::Point(x2, y2), rightEye.width / 2, cv::Scalar(255, 255), 2);
					_rightEyeCircle[0] = x2;
					_rightEyeCircle[1] = y2;
					_rightEyeCircle[2] = rightEye.width / 2;
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

void SMT_setUseEyeTracking(bool useEyeTracking) {
	if (instance == nullptr) return;
	instance->setUseEyeTracking(useEyeTracking);
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

void SMT_cvWait() {
	cv::waitKey(1);
}

int SMT_getErrorCode() {
	if (instance == nullptr) return SMT_ERROR_NOEN;
	return instance->getErrorCode();
}
