using System.Runtime.InteropServices;

public class SMT
{
    [DllImport("WebCamEyesTracker")]
	public static extern void WebCamEyesTracker_init(int camId);
    [DllImport("WebCamEyesTracker")]
    public static extern void WebCamEyesTracker_destroy();
    [DllImport("WebCamEyesTracker")]
    public static extern void WebCamEyesTracker_update();
    [DllImport("WebCamEyesTracker")]
    public static extern void WebCamEyesTracker_setCaptureFrameVisible(int visible);
    [DllImport("WebCamEyesTracker")]
    public static extern float WebCamEyesTracker_getParam(int paramId);
}
