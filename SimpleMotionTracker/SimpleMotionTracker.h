#pragma once
#ifdef NATIVEPLUGINSAMPLE_EXPORTS
#define SIMPLEMOTIONTRACKER_API __declspec(dllexport)
#else
#define SIMPLEMOTIONTRACKER_API __declspec(dllimport)
#endif


extern "C" {
	static const int SMT_ERROR_NOEN = 0;
	static const int SMT_ERROR_UNOPEND_CAMERA = -1;
	static const int SMT_ERROR_UNOPEND_CAMERA_PARAM_FILE = -2;
	static const int SMT_ERROR_UNOPEN_FACE_CASCADE = -3;
	static const int SMT_ERROR_UNOPEN_EYE_CASCADE = -4;
	static const int SMT_ERROR_UNREADABLE_CAMERA = -5;
	static const int SMT_ERROR_INSUFFICIENT_CAMERA_CAPTURE_SPEED = -6;

	SIMPLEMOTIONTRACKER_API void SMT_initRaw(int cameraId);
	SIMPLEMOTIONTRACKER_API void SMT_init(const char* videoDeviceName);
	SIMPLEMOTIONTRACKER_API void SMT_destroy();
	SIMPLEMOTIONTRACKER_API void SMT_update();

	SIMPLEMOTIONTRACKER_API void SMT_setUseARMarker(bool useARMarker);
	SIMPLEMOTIONTRACKER_API void SMT_setUseFaceTracking(bool useFaceDetect);
	SIMPLEMOTIONTRACKER_API void SMT_setUseEyeTracking(bool useFaceDetect);
	SIMPLEMOTIONTRACKER_API void SMT_setCaptureShown(bool isShown);
	SIMPLEMOTIONTRACKER_API bool SMT_isCaptureShown();

	SIMPLEMOTIONTRACKER_API void SMT_setARMarkerEdgeLength(float length);
	SIMPLEMOTIONTRACKER_API bool SMT_isARMarkerDetected(int id);
	SIMPLEMOTIONTRACKER_API void SMT_getARMarker6DoF(int id, float* outArray);

	SIMPLEMOTIONTRACKER_API bool SMT_isFacePointsDetected();
	SIMPLEMOTIONTRACKER_API void SMT_getFacePoints(float* outArray);

	SIMPLEMOTIONTRACKER_API void SMT_cvWait();
	SIMPLEMOTIONTRACKER_API int SMT_getErrorCode();
}
