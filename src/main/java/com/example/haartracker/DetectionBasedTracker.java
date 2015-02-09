package com.example.haartracker;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import android.content.res.AssetManager;

public class DetectionBasedTracker
{
    public DetectionBasedTracker(String cascadeName, String cascadeName2, String cascadeName3, int minFaceSize) {
        mNativeObj = nativeCreateObject(cascadeName, cascadeName2, cascadeName3, minFaceSize);
    }

    public void start() {
        nativeStart(mNativeObj);
    }
    
    public void reStart() {
        nativeRestart(mNativeObj);
    }

    public void stop() {
        nativeStop(mNativeObj);
    }

    public void setMinFaceSize(int size) {
        nativeSetFaceSize(mNativeObj, size);
    }

    public long detect(Mat imageGray, MatOfRect faces, MatOfRect eyes, String storagePath, AssetManager mgr) {
        return nativeDetect(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr(), eyes.getNativeObjAddr(), storagePath, mgr);
    }

    public void release() {
        nativeDestroyObject(mNativeObj);
        mNativeObj = 0;
    }

    public double recognize(Mat imageGray, MatOfRect faces) {
        return nativeRecognize(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr());
    }
    
    public void extract() {
        nativeExtract(mNativeObj);
    }
    
    public void deleteAll() {
        nativeDeleteAll(mNativeObj);
    }
    
    private long mNativeObj = 0;

    private static native long nativeCreateObject(String cascadeName, String cascadeName2, String cascadeName3, int minFaceSize);
    private static native void nativeDestroyObject(long thiz);
    private static native void nativeStart(long thiz);
    private static native void nativeRestart(long thiz);
    private static native void nativeStop(long thiz);
    private static native void nativeSetFaceSize(long thiz, int size);
    private static native long nativeDetect(long thiz, long inputImage, long faces, long eyes, String storagePath, AssetManager mgr);
    private static native double nativeRecognize(long thiz, long inputImage, long faces);
    private static native void nativeExtract(long thiz);
    private static native void nativeDeleteAll(long thiz);
}
