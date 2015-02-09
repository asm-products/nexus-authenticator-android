package com.example.haartracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import com.example.haartracker.R;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.view.KeyEvent;

public class HaarTracker extends Activity implements CvCameraViewListener2 {

	private static final String    TAG                 = "HaarTracker";
	private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 0, 255, 255);
	private static final Scalar    TEXT_FONT_COLOR     = new Scalar(255, 0, 0, 255);
	public static final int        JAVA_DETECTOR       = 0;
	public static final int        NATIVE_DETECTOR     = 1;
	public static final int        TRAINING_DETECTOR   = 2;
	public static final int        RECOG_DETECTOR      = 3;
	public static final int        EXTRACT_DETECTOR    = 4;
	public static final int        DELETE_DETECTOR     = 5;
	public static final double     IMAGE_SCALE         = 1; // 1 for native code, 8 for java
	public static final double     FONT_SIZE           = 1.5;
	public static int		   	   FRAME_COUNTER	   = 0;
	
	private static Ringtone        r;
	private static Uri             notification;
	private long                   startTimeYaw        = 0;
	private long                   elapsedTimeYaw      = 0;
	private static final int       SPEED_FAST          = 750;
	private int                    durationLimit       = SPEED_FAST;

	private MenuItem               mItemFace50;
	private MenuItem               mItemFace40;
	private MenuItem               mItemFace30;
	private MenuItem               mItemFace20;
	private MenuItem               mItemFace10;
	private MenuItem               mItemType;

	private Mat                    mRgba;
	private Mat                    mGray;
	private Mat                    smallGray;
	private File                   mCascadeFile, mCascadeProfileFile;
	private File                   mCascadeFaceFile, mCascadeEyesFile, mCascadeEyeglassesFile;
	private CascadeClassifier      mJavaDetector, mJavaProfileDetector;
	private DetectionBasedTracker  mNativeDetector;
    private CountDownTimer         waitTimer;

	private int                    mDetectorType       = TRAINING_DETECTOR;
	private String[]               mDetectorName;

	private float                  mRelativeFaceSize   = 0.4f;
	private int                    mAbsoluteFaceSize   = 0;
	private AssetManager mgr;

    // Intent Activities
    private static final String CLASSNAME = HaarTracker.class.getName();
    public static final String ACTION_RECOGNIZE_FACE = CLASSNAME
            + ".recognize_face";
    public static final String ACTION_TRAIN_FACE = CLASSNAME
            + ".train_face";
    public static final String ACTION_DELETE_FACE = CLASSNAME
            + ".delete_face";
	
	private CameraBridgeViewBase   mOpenCvCameraView;
	File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);

    private View mFooter;
    private Button mBtnCancel;
    private Button mBtnConfirm;

	static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        } else {
        	Log.i(TAG, "OpenCV loaded successfully");
        	// Load native library after(!) OpenCV initialization
            System.loadLibrary("detection_based_tracker");
        }
    }
	
	public HaarTracker() {
		mDetectorName = new String[2];
		mDetectorName[JAVA_DETECTOR] = "Java";
		mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

//		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.face_detect_surface_view);

        Boolean btnOkEnabled = mBtnConfirm != null ? mBtnConfirm.isEnabled()
                : null;

        mFooter = findViewById(R.id.alp_viewgroup_footer);
        mBtnCancel = (Button) findViewById(R.id.alp_button_cancel);
        mBtnConfirm = (Button) findViewById(R.id.alp_button_confirm);
        mBtnCancel.setOnClickListener(mBtnCancelOnClickListener);

        if (ACTION_TRAIN_FACE.equals(getIntent().getAction())) {
            mBtnCancel.setVisibility(View.VISIBLE);
            mFooter.setVisibility(View.VISIBLE);
            mDetectorType = RECOG_DETECTOR;
        }
        else if (ACTION_RECOGNIZE_FACE.equals(getIntent().getAction())) {
            mDetectorType = EXTRACT_DETECTOR;
            waitTimer = new CountDownTimer(10000, 1000) {

                public void onTick(long millisUntilFinished) {
                    //called every 1000 milliseconds, which could be used to
                    //send messages or some other action
                }

                public void onFinish() {
                    //After 10000 milliseconds (10 sec) finish current
                    //if you would like to execute something when time finishes
                    setResult(Activity.RESULT_CANCELED);
                    finish();
                }
            }.start();

        }
        else if (ACTION_DELETE_FACE.equals(getIntent().getAction())) {
            mDetectorType = DELETE_DETECTOR;
        }

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(new View.OnTouchListener() {
            public boolean onTouch(View v, MotionEvent event) {
                if(waitTimer != null) {
                    waitTimer.cancel();
                    waitTimer = null;
                }
                setResult(Activity.RESULT_CANCELED);
                finish();
                return true;
            }
        });

		notification = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
		r = RingtoneManager.getRingtone(getApplicationContext(), notification);
		
		mgr = getResources().getAssets();
	}

	@Override
	public void onPause()
	{
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume()
	{
		super.onResume();
		try {
			// load cascade file from application resource
			InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
			// InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
			File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
			mCascadeFaceFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
			//  mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
			FileOutputStream os = new FileOutputStream(mCascadeFaceFile);

			byte[] buffer = new byte[4096];
			int bytesRead;
			while ((bytesRead = is.read(buffer)) != -1) {
				os.write(buffer, 0, bytesRead);
			}
			is.close();
			os.close();

			mJavaDetector = new CascadeClassifier(mCascadeFaceFile.getAbsolutePath());
			if (mJavaDetector.empty()) {
				Log.e(TAG, "Failed to load cascade classifier");
				mJavaDetector = null;
			} else
				Log.i(TAG, "Loaded cascade classifier from " + mCascadeFaceFile.getAbsolutePath());

			// load cascade file from application resources
			is = getResources().openRawResource(R.raw.haarcascade_eye);
			cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
			mCascadeEyesFile = new File(cascadeDir, "haarcascade_eye.xml");
			os = new FileOutputStream(mCascadeEyesFile);

			buffer = new byte[4096];
			bytesRead = 0;
			while ((bytesRead = is.read(buffer)) != -1) {
				os.write(buffer, 0, bytesRead);
			}
			is.close();
			os.close();

			// load cascade file from application resources
			is = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
			cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
			mCascadeEyeglassesFile = new File(cascadeDir, "haarcascade_eye_tree_eyeglasses.xml");
			os = new FileOutputStream(mCascadeEyeglassesFile);

			buffer = new byte[4096];
			bytesRead = 0;
			while ((bytesRead = is.read(buffer)) != -1) {
				os.write(buffer, 0, bytesRead);
			}
			is.close();
			os.close();

			mJavaProfileDetector = new CascadeClassifier(mCascadeEyeglassesFile.getAbsolutePath());
			if (mJavaProfileDetector.empty()) {
				Log.e(TAG, "Failed to load cascade classifier");
				mJavaProfileDetector = null;
			} else
				Log.i(TAG, "Loaded cascade classifier from " + mCascadeEyeglassesFile.getAbsolutePath());

			cascadeDir.delete();

			mNativeDetector = new DetectionBasedTracker(mCascadeFaceFile.getAbsolutePath(), mCascadeEyesFile.getAbsolutePath(),
			mCascadeEyeglassesFile.getAbsolutePath(), 0);

		} catch (IOException e) {
			e.printStackTrace();
			Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
		}

		mOpenCvCameraView.enableView();
//		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
		smallGray = new Mat();
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
		smallGray.release();
	}

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            /*
             * Use this hook instead of onBackPressed(), because onBackPressed()
             * is not available in API 4.
             */
            setResult(Activity.RESULT_CANCELED);
            finish();
//            return true;
        }

        return super.onKeyDown(keyCode, event);
    }

	private class TrainFacesTask extends AsyncTask<Object, Integer, Object> {
	     protected Object[] doInBackground(Object... params) {
	    	 
	    	 if (mNativeDetector != null)
					mNativeDetector.detect(smallGray, (MatOfRect)params[0], (MatOfRect)params[1], getOutputMediaFile(""), mgr);
			return params;
	         
	     }

	     protected void onPostExecute(Object[] result) {
	    	 Rect[] facesArray = ((MatOfRect)result[0]).toArray();
	 		Rect[] eyesArray = null;
	 		if (!((MatOfRect)result[1]).empty()) {
	 			eyesArray = ((MatOfRect)result[1]).toArray();
	 			Log.w(TAG, "NOT EMPTY");
	 		}
	 		for (int i = 0; i < facesArray.length; i++) {
	 			Core.rectangle(mRgba, new Point(facesArray[i].tl().x * IMAGE_SCALE, facesArray[i].tl().y * IMAGE_SCALE), 
	 					new Point(facesArray[i].br().x * IMAGE_SCALE, facesArray[i].br().y * IMAGE_SCALE), FACE_RECT_COLOR, 3);
	 			if (eyesArray != null) {
	 				Core.rectangle(mRgba, new Point((facesArray[i].tl().x + eyesArray[0].tl().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[0].tl().y) * IMAGE_SCALE), 
	 						new Point((facesArray[i].tl().x + eyesArray[0].br().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[0].br().y) * IMAGE_SCALE), TEXT_FONT_COLOR, 3);
	 				Core.rectangle(mRgba, new Point((facesArray[i].tl().x + eyesArray[1].tl().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[1].tl().y) * IMAGE_SCALE), 
	 						new Point((facesArray[i].tl().x + eyesArray[1].br().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[1].br().y) * IMAGE_SCALE), TEXT_FONT_COLOR, 3);
	 				Core.rectangle(mRgba, new Point((facesArray[i].tl().x + eyesArray[2].tl().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[2].tl().y) * IMAGE_SCALE), 
	 						new Point((facesArray[i].tl().x + eyesArray[2].br().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[2].br().y) * IMAGE_SCALE), TEXT_FONT_COLOR, 3);
	 				Core.rectangle(mRgba, new Point((facesArray[i].tl().x + eyesArray[3].tl().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[3].tl().y) * IMAGE_SCALE), 
	 						new Point((facesArray[i].tl().x + eyesArray[3].br().x) * IMAGE_SCALE, (facesArray[i].tl().y + eyesArray[3].br().y) * IMAGE_SCALE), TEXT_FONT_COLOR, 3);
	 			}
	 		}
	     }
	 }
	
	private class RecognizeFacesTask extends AsyncTask<Object, Integer, Object> {
		TextView tv;
		double confidence;
	     protected Object[] doInBackground(Object... params) {
	    	 
	    	 if (mNativeDetector != null)
					confidence = mNativeDetector.recognize(smallGray, (MatOfRect)params[0]);
			return params;
	     }

	     protected void onPostExecute(Object[] result) {
	    	 
	    	 tv = (TextView) findViewById(R.id.txtDisp);

             tv.setText(Double.toString((Double)result[0]));
	     }
	 }
	
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		
		mRgba = inputFrame.rgba();
		Point center = new Point(mRgba.cols()/2, mRgba.rows()/2);
		Mat rotImage = Imgproc.getRotationMatrix2D(center, 90, 1.0);
//		Mat dummy = new Mat();
		Imgproc.warpAffine(mRgba, mRgba, rotImage, mRgba.size());
//		mRgba = dummy;
		
		Core.flip(mRgba, mRgba, 1);
		
		

		
//		mGray = inputFrame.gray();
//		smallGray = new Mat((int)Math.round(mRgba.rows()/IMAGE_SCALE), (int)Math.round(mRgba.cols()/IMAGE_SCALE), mRgba.type());
//		Imgproc.resize(mRgba, smallGray, smallGray.size(), 0, 0, Imgproc.INTER_LINEAR);
//		Imgproc.equalizeHist( smallGray, smallGray );
		
//		if(FRAME_COUNTER++%2 != 0)
//		{
//			return smallGray;
//		}
		
		

//		if (mAbsoluteFaceSize == 0) {
//		 	int height = smallGray.rows();
//			if (Math.round(height * mRelativeFaceSize) > 0) {
//				mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
//			}
//			mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
//		}
//
		MatOfRect faces = new MatOfRect();
		MatOfRect eyes = new MatOfRect();

		if (mDetectorType == JAVA_DETECTOR) {
			if (mJavaDetector != null)
			{
				mJavaDetector.detectMultiScale(smallGray, faces, 1.02, 2, 0
//												| org.opencv.objdetect.Objdetect.CASCADE_FIND_BIGGEST_OBJECT
												| org.opencv.objdetect.Objdetect.CASCADE_DO_ROUGH_SEARCH
//												| org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE
//												| org.opencv.objdetect.Objdetect.CASCADE_DO_CANNY_PRUNING
						,
						new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
				if (!faces.empty())
				{
					Core.putText(mRgba, "-45 45", new Point(15, 100),
							Core.FONT_HERSHEY_SIMPLEX, FONT_SIZE, new Scalar(0, 255, 0, 255), 3);
					Log.w(TAG, "LOOKING FRONT");
					startTimeYaw = 0;
					elapsedTimeYaw = 0;
				}
			}
			if (faces.empty())
			{
				if (mJavaProfileDetector != null)
				{
					mJavaProfileDetector.detectMultiScale(smallGray, faces, 1.2, 2, 0
														| org.opencv.objdetect.Objdetect.CASCADE_FIND_BIGGEST_OBJECT
//														| org.opencv.objdetect.Objdetect.CASCADE_DO_ROUGH_SEARCH
														| org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE
//														| org.opencv.objdetect.Objdetect.CASCADE_DO_CANNY_PRUNING
							,
							new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
					if (!faces.empty())
					{
						Core.putText(mRgba, "-90 -45", new Point(15, 100),
								Core.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_FONT_COLOR, 3);
						Log.w(TAG, "LOOKING LEFT");
						if (startTimeYaw == 0) {
							startTimeYaw = System.currentTimeMillis();
						} else {
							elapsedTimeYaw = System.currentTimeMillis() - startTimeYaw;
						}
						if (elapsedTimeYaw >= durationLimit)
							try {
								r.play();
							} catch (Exception e) {
							}
					}
					else {
						Core.flip(smallGray, smallGray, 1);
						mJavaProfileDetector.detectMultiScale(smallGray, faces, 1.2, 2, 0
//																| org.opencv.objdetect.Objdetect.CASCADE_FIND_BIGGEST_OBJECT
//																| org.opencv.objdetect.Objdetect.CASCADE_DO_ROUGH_SEARCH
//																| org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE
//																| org.opencv.objdetect.Objdetect.CASCADE_DO_CANNY_PRUNING
								,
								new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
						if (!faces.empty()) 
						{
							Core.putText(mRgba, "45 90", new Point(15, 100),
											Core.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_FONT_COLOR, 3);
							Log.w(TAG, "LOOKING RIGHT");
							if (startTimeYaw == 0) {
								startTimeYaw = System.currentTimeMillis();
							} else {
								elapsedTimeYaw = System.currentTimeMillis() - startTimeYaw;
							}
							if (elapsedTimeYaw >= durationLimit)
								try {
									r.play();
								} catch (Exception e) {
								}
						}
					}
				}
			}
		}
		else if (mDetectorType == TRAINING_DETECTOR) {
			if (mNativeDetector != null)
			{
				long result= mNativeDetector.detect(mRgba, faces, eyes, getOutputMediaFile("eigenfaces.yml"), mgr);
//				new TrainFacesTask().execute(faces, eyes);
                Log.w(TAG, "RESULT IS: " + result);
				if(result==1)
				{
					ProgressBar myBar = (ProgressBar)findViewById(R.id.progressBar1);
					
					myBar.incrementProgressBy(4);
					if(myBar.getProgress()==100)
					{
						myBar.setProgress(0);
						mNativeDetector.reStart();
						
						runOnUiThread(new Runnable() {
						     @Override
						     public void run() {
						    	 
						    	 ProgressBar myBar = (ProgressBar)findViewById(R.id.progressBar1);
						    	 myBar.setVisibility(View.INVISIBLE);
//								 Button addPersonButton = (Button) findViewById(R.id.addPersonButton);
//								 Button recognizePersonButton = (Button) findViewById(R.id.recognizePersonButton);
//								 Button deleteAllButton = (Button) findViewById(R.id.deleteAllButton);
//								 addPersonButton.setVisibility(View.VISIBLE);
//								 recognizePersonButton.setVisibility(View.VISIBLE);
//								 deleteAllButton.setVisibility(View.VISIBLE);
                                 mBtnConfirm.setVisibility(View.VISIBLE);
                                 mBtnConfirm.setEnabled(true);
                                 mBtnConfirm.setOnClickListener(mBtnConfirmOnClickListener);
						    }
						});


					}
				}
                else if(result == 2){
                    // extracted face successfully
                    if(waitTimer != null) {
                        waitTimer.cancel();
                        waitTimer = null;
                    }
                    setResult(Activity.RESULT_OK);
                    finish();
                }
				
			}
		}
		else if (mDetectorType == RECOG_DETECTOR) {
			if (mNativeDetector != null)
			{
				mNativeDetector.recognize(mRgba, faces);
//				new RecognizeFacesTask().execute(faces);
				mDetectorType = TRAINING_DETECTOR;
			}
		}
		else if (mDetectorType == EXTRACT_DETECTOR) {
			if (mNativeDetector != null)
			{
				mNativeDetector.extract();
//				new RecognizeFacesTask().execute(faces);
				mDetectorType = TRAINING_DETECTOR;

			}
		}
        else if (mDetectorType == DELETE_DETECTOR) {
            if (mNativeDetector != null)
            {
                mNativeDetector.deleteAll();
//				new RecognizeFacesTask().execute(faces);
                mDetectorType = TRAINING_DETECTOR;
            }
        }
		else {
			Log.e(TAG, "Detection method is not selected!");
		}
		
//		if (!eyes.empty()) {
//		Rect leftRoi = new Rect((int)((facesArray[0].tl().x + eyesArray[0].tl().x) * IMAGE_SCALE),
//				(int)((facesArray[0].tl().y + eyesArray[0].tl().y) * IMAGE_SCALE),
//				(int)(eyesArray[1].br().x * IMAGE_SCALE - eyesArray[0].tl().x * IMAGE_SCALE),
//				(int)(eyesArray[1].br().y * IMAGE_SCALE - eyesArray[0].tl().y * IMAGE_SCALE));
////		Mat mIntermediateMat = new Mat();
////		Imgproc.cvtColor(mRgba, mIntermediateMat, Imgproc.COLOR_BGR2RGB, 3);
//		Highgui.imwrite(getOutputMediaFile("left-eye"), new Mat(smallGray, leftRoi));
//		
//		Rect rightRoi = new Rect((int)((facesArray[0].tl().x + eyesArray[2].tl().x) * IMAGE_SCALE),
//				(int)((facesArray[0].tl().y + eyesArray[2].tl().y) * IMAGE_SCALE),
//				(int)(eyesArray[3].br().x * IMAGE_SCALE - eyesArray[2].tl().x * IMAGE_SCALE),
//				(int)(eyesArray[3].br().y * IMAGE_SCALE - eyesArray[2].tl().y * IMAGE_SCALE));
////		Mat mIntermediateMat = new Mat();
////		Imgproc.cvtColor(mRgba, mIntermediateMat, Imgproc.COLOR_BGR2RGB, 3);
//		Highgui.imwrite(getOutputMediaFile("right-eye"), new Mat(smallGray, rightRoi));
//		}
		return mRgba;
	}

    private final View.OnClickListener mBtnConfirmOnClickListener = new View.OnClickListener() {

        @Override
        public void onClick(View v) {
//            if (ACTION_TRAIN_FACE.equals(getIntent().getAction())) {


                    mBtnConfirm.setEnabled(false);
                    setResult(Activity.RESULT_OK);
                    finish();
//                }// ACTION_CREATE_PATTERN
//                else if (ACTION_COMPARE_PATTERN.equals(getIntent().getAction())) {
                /*
                 * We don't need to verify the extra. First, this button is only
                 * visible if there is this extra in the intent. Second, it is
                 * the responsibility of the caller to make sure the extra is an
                 * Intent of Activity.
                 */
//                startActivity((Intent) getIntent().getParcelableExtra(
//                        EXTRA_INTENT_ACTIVITY_FORGOT_PATTERN));
//                finishWithNegativeResult(RESULT_FORGOT_PATTERN);
//                }// ACTION_COMPARE_PATTERN
            }// onClick()
    };

    private final View.OnClickListener mBtnCancelOnClickListener = new View.OnClickListener() {

        @Override
        public void onClick(View v) {
//            if (ACTION_TRAIN_FACE.equals(getIntent().getAction())) {


            mBtnConfirm.setEnabled(false);
            setResult(Activity.RESULT_CANCELED);
            finish();
//                }// ACTION_CREATE_PATTERN
//                else if (ACTION_COMPARE_PATTERN.equals(getIntent().getAction())) {
                /*
                 * We don't need to verify the extra. First, this button is only
                 * visible if there is this extra in the intent. Second, it is
                 * the responsibility of the caller to make sure the extra is an
                 * Intent of Activity.
                 */
//                startActivity((Intent) getIntent().getParcelableExtra(
//                        EXTRA_INTENT_ACTIVITY_FORGOT_PATTERN));
//                finishWithNegativeResult(RESULT_FORGOT_PATTERN);
//                }// ACTION_COMPARE_PATTERN
        }// onClick()
    };

	private void storageLocation(){
	    File externalAppDir = getExternalFilesDir(null);
	    String storagePath = externalAppDir.getAbsolutePath();
	}

    private static String getOutputMediaFile(String face) {
        // To be safe, i should check that the SDCard is mounted
        // using Environment.getExternalStorageState() before doing this.
        File mediaStorageDir = new File(
                Environment
                        .getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "FaceDatabase");
        // This location works best if i want the created images to be shared
        // between applications and persist after the app has been uninstalled.

        // Create the storage directory if it does not exist
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                Log.d("EyeDatabase", "failed to create directory");
                return null;
            }
        }

        // Create a media file name
        String timeStamp = new SimpleDateFormat("hh_mm_ss.SSS", Locale.US)
                .format(new Date());

        return mediaStorageDir.getPath() + File.separator + face;
    }

//	@Override
//	public boolean onCreateOptionsMenu(Menu menu) {
//		Log.i(TAG, "called onCreateOptionsMenu");
////		mItemFace50 = menu.add("Face size 50%");
////		mItemFace40 = menu.add("Face size 40%");
////		mItemFace30 = menu.add("Face size 30%");
////		mItemFace20 = menu.add("Face size 20%");
//		mItemFace10 = menu.add("Add Person");
//		mItemType   = menu.add("Exctract Person");
////		mItemType   = menu.add(mDetectorName[mDetectorType]);
//		return true;
//	}

//	@Override
//	public boolean onOptionsItemSelected(MenuItem item) {
//		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
//		if (item == mItemFace50)
//			setMinFaceSize(0.5f);
//		else if (item == mItemFace40)
//			setMinFaceSize(0.4f);
//		else if (item == mItemFace30)
//			setMinFaceSize(0.3f);
//		else if (item == mItemFace20)
//		{
////			setMinFaceSize(0.2f);
//			setDetectorType(DELETE_DETECTOR);
//		}
//		else if (item == mItemFace10)
//		{
//			setDetectorType(RECOG_DETECTOR);
//		}
//		else if (item == mItemType) {
////			int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
////			item.setTitle(mDetectorName[tmpDetectorType]);
////			setDetectorType(tmpDetectorType);
//			setDetectorType(EXTRACT_DETECTOR);
//		}
//		return true;
//	}

//	public void addPersonButtonClicked(View v)
//	{
//		setDetectorType(RECOG_DETECTOR);
//		Button addPersonButton = (Button) findViewById(R.id.addPersonButton);
//		Button recognizePersonButton = (Button) findViewById(R.id.recognizePersonButton);
//		Button deleteAllButton = (Button) findViewById(R.id.deleteAllButton);
//		addPersonButton.setVisibility(View.INVISIBLE);
//		recognizePersonButton.setVisibility(View.INVISIBLE);
//		deleteAllButton.setVisibility(View.INVISIBLE);
//		ProgressBar myBar = (ProgressBar) findViewById(R.id.progressBar1);
//		myBar.setVisibility(View.VISIBLE);
//	}
	public void extractPersonButtonClicked(View v)
	{
		setDetectorType(EXTRACT_DETECTOR);
	}
	public void deleteAllButtonClicked(View v)
	{
		setDetectorType(DELETE_DETECTOR);
	}
	private void setMinFaceSize(float faceSize) {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
	}

	private void setDetectorType(int type) {
		if (mDetectorType != type) {
			mDetectorType = type;
			
			if (type == TRAINING_DETECTOR) {
				Log.i(TAG, "Training enabled");
//				mNativeDetector.start();
			}
			else if (type == RECOG_DETECTOR) {
				Log.i(TAG, "Recognition enabled");
				}
			} 
//			else {
//				Log.i(TAG, "Cascade detector enabled");
//				mNativeDetector.stop();
//			}
		
	}
}
