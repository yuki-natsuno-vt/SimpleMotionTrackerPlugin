using System.Runtime.InteropServices;

public class SMT
{
    public const int SMT_ERROR_NOEN = 0; 
    public const int SMT_ERROR_UNOPEND_CAMERA = -1;
    public const int SMT_ERROR_UNOPEND_CAMERA_PARAM_FILE = -2;
    public const int SMT_ERROR_UNOPEN_FACE_CASCADE = -3;
    public const int SMT_ERROR_UNOPEN_EYE_CASCADE = -4;
    public const int SMT_ERROR_UNREADABLE_CAMERA = -5;
    public const int SMT_ERROR_INSUFFICIENT_CAMERA_CAPTURE_SPEED = -6;

    [DllImport("SimpleMotionTracker", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern void SMT_init(string videoDeviceName);
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_destroy();
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_update();
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_setUseARMarker(bool useARMarker);
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_setUseFaceTracking(bool useFaceTracking);
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_setUseEyeTracking(bool useEyeTracking);
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_setCaptureShown(bool isShown);
    [DllImport("SimpleMotionTracker")]
    private static extern bool SMT_isCaptureShown();
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_setARMarkerEdgeLength(float length);
    [DllImport("SimpleMotionTracker")]
    private static extern bool SMT_isARMarkerDetected(int id);
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_getARMarker6DoF(int id, System.IntPtr outArray);
    [DllImport("SimpleMotionTracker")]
    private static extern bool SMT_isFacePointsDetected();
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_getFacePoints(System.IntPtr outArray);
    [DllImport("SimpleMotionTracker")]
    private static extern void SMT_setIrisThresh(int thresh);
    [DllImport("SimpleMotionTracker")]
    private static extern int SMT_getErrorCode();
}
