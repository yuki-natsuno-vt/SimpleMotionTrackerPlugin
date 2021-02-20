#include "../SimpleMotionTracker/SimpleMotionTracker.h"
#include <cstdio>

int main() {
	//SMT_init("Logicool HD Webcam C270");
	//SMT_init("FaceRig Virtual Camera");
	SMT_initRaw(1, "data/");
	SMT_setCaptureShown(true);
	SMT_setARMarkerEdgeLength(0.036f);
	SMT_setUseFaceTracking(true);
	SMT_setUseEyeTracking(true);
	SMT_setUseHandTracking(true);
	SMT_setHandUndetectedDuration(5000);
	while (1) {
		SMT_update();

		if (SMT_isARMarkerDetected(0)) {
			float v[6];
			SMT_getARMarker6DoF(0, v);
			printf("%f, %f, %f, %f, %f, %f\n", v[0], v[1], v[2], v[3], v[4], v[5]);
		}
		SMT_cvWait();
	}

	SMT_destroy();

	return 0;
}

