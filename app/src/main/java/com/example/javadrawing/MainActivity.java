package com.example.javadrawing;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.Toast;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.lang.*;


import butterknife.BindView;
import butterknife.ButterKnife;

import static org.opencv.core.Core.bitwise_not;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;

public class MainActivity extends Activity {

    private static final String LOG_TAG = "MainActivity";

    @BindView(R.id.classify)
    protected View classifyButton;

    @BindView(R.id.reset)
    protected View resetButton;

    @BindView(R.id.viewContainer)
    protected ViewGroup viewContainer;

    @BindView(R.id.drawingView)
    protected DrawingView drawingView;

    @BindView(R.id.bitmapTester)
    protected ImageView bitmapTester;

    private Executor executor = Executors.newSingleThreadExecutor();
    private Classifier classifier;

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private final static String IMAGEPATH = "samples/sample1.jpg";
    private final static String[] DIGITS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    ArrayList<RoiObject> mRoiImages = new ArrayList<RoiObject>(50);
    private static final String MODEL_FILE = "file:///android_asset/data.pb";
    private static final String LABEL_FILE = "file:///android_asset/labels.txt";
    //   private static final String TAG = ;
    static {
        if (OpenCVLoader.initDebug()) {
            Log.d(LOG_TAG, "OpenCV Successfully Loaded");
        } else {
            Log.d(LOG_TAG, "OpenCV Load Not Successfully");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        ButterKnife.bind(this);
        loadModel();

        resetButton.setOnClickListener(v -> reset());
        classifyButton.setOnClickListener(v -> {
            onClassify();
        });
    }

    private void onClassify()
    {
        String re1="",re2="",re3="",re4="";
        StringBuilder ResultDigits  = new StringBuilder("");
        mRoiImages.clear();
        Bitmap bitmap = drawingView.getBitmap();
        Log.i(LOG_TAG, "wi"+bitmap.getWidth());
        Log.i(LOG_TAG, "hei"+bitmap.getHeight());
        Bitmap gray = toGrayScale(bitmap);
        Mat imageMat = new Mat();
        Utils.bitmapToMat(bitmap,imageMat);
        Mat grey = new Mat();
        cvtColor(imageMat,grey,Imgproc.COLOR_BGR2GRAY);
        //     Imgproc.Canny(grey,grey,50,200);
        Imgproc.adaptiveThreshold(grey,grey,255,Imgproc.ADAPTIVE_THRESH_MEAN_C,Imgproc.THRESH_BINARY_INV,7,7);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Mat roiImage;
        Imgproc.findContours(grey,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);




        // Draw blue contours on a white image
        //   Mat result= imageMat;
        Mat result = grey;

        cvtColor(result, result, COLOR_GRAY2BGR);
        for(int i=0;i<contours.size();i++)
        {
            drawContours(result, contours,
                    i, // draw all contours
                    new Scalar(255, 255, 255),
                    40);

        }

        //Start to iterate to each contour founded
        ListIterator<MatOfPoint> itc = contours.listIterator();


        int k=0,y=0;

        //Remove patch that are no inside limits of aspect ratio and area.
        while(itc.hasNext())
        {


            MatOfPoint mp = new MatOfPoint(itc.next().toArray());
            Rect mr = boundingRect(mp);
            // Imgproc.rectangle(result,new Point(mr.x,mr.y),new Point(mr.x+mr.width,mr.y+mr.height),new Scalar(0,255,0));
            Imgproc.rectangle(result, new org.opencv.core.Point(mr.x, mr.y), new org.opencv.core.Point(mr.x + mr.width, mr.y + mr.height), new Scalar(0, 255,0));
            //      Mat auxRoi=new Mat(result,mr);
            Mat auxRoi = result.submat(mr.y,mr.y + mr.height,mr.x,mr.x + mr.width);

            // Imgproc.fillConvexPoly(auxRoi,mp,new Scalar(255,255,255));
            Core.copyMakeBorder(auxRoi, auxRoi, 100, 100, 100, 100, Core.BORDER_ISOLATED);


            Bitmap b = Bitmap.createBitmap(auxRoi.width(),auxRoi.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(auxRoi,b);
            b = scaling(b,28,28);

            b = toGrayScale(b);
            k=k+1;
            RoiObject roiObject = new RoiObject(mr.x,b);
            Log.i(LOG_TAG, "wi" + mr.x);
            mRoiImages.add(roiObject);
            SaveImage(b,k);
            y=y+1;

        }
        Collections.sort(mRoiImages);
        int z=5;

        int max = (mRoiImages.size() > 4) ? 4 : mRoiImages.size();
        for (int i = 0; i < max; i++) {
            RoiObject roi = mRoiImages.get(i);
            roi.bmp= toGrayScale(roi.bmp);
            roi.bmp = scaling(roi.bmp,28,28);
            int width = 28;
            int[] pixels = new int[width * width];

            roi.bmp.getPixels(pixels, 0, width, 0, 0, width, width);

            float[] retPixels = createInputPixels(pixels);

            Classification classification = classifier.recognize(retPixels);
            String result1 = String.format("It's a %s with confidence: %f", classification.getLabel(), classification.getConf());
            String dig = classification.getLabel();
            //     Toast.makeText(MainActivity.class, result, Toast.LENGTH_SHORT).show();
            Log.i(LOG_TAG, "digit =" + dig);
            if(i==0)
                re1=dig;
            else if(i==1)
                re2=dig;
            else if(i==2)
                re3=dig;
            else if(i==3)
                re4=dig;
        }
        Log.i(LOG_TAG, "digit1 =" + re1);
        Log.i(LOG_TAG, "digit2 =" + re2);
        Log.i(LOG_TAG, "digit3 =" + re3);
        //1+21
        if(re2.equals("10"))
        {
            re4 = re3+re4;
            Integer r=Integer.parseInt(re1) + Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
        else if(re2.equals("11"))
        {
            re4 = re3+re4;
            Integer r=Integer.parseInt(re1) - Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
        else if(re2.equals("12"))
        {
            re4 = re3+re4;
            Integer r=Integer.parseInt(re1) * Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
        else if(re2.equals("13"))
        {
            re4 = re3+re4;
            Integer r=Integer.parseInt(re1) / Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }

        if(re3.equals("10"))
        {
            re1 = re1+re2;
            Integer r=Integer.parseInt(re1) + Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
        else if(re3.equals("11"))
        {
            re1 = re1+re2;
            Integer r=Integer.parseInt(re1) - Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
        else if(re3.equals("12"))
        {
            re1 = re1+re2;
            Integer r=Integer.parseInt(re1) * Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
        else if(re3.equals("13"))
        {
            re1 = re1+re2;
            Integer r=Integer.parseInt(re1) / Integer.parseInt(re4);
            ResultDigits.append(""+r.toString());
        }
     //   if(re2.isEmpty())
       //     ResultDigits.append(""+re1);


        Log.i(LOG_TAG,"result"+ResultDigits);
        Toast.makeText(getApplicationContext(),ResultDigits,Toast.LENGTH_LONG).show();
        Bitmap b = Bitmap.createBitmap(result.width(),result.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result,b);
        SaveImage(b,0);

    }
    public Bitmap scaling(Bitmap bitmap, int newWidth, int newHeight)
    {

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) /height;

        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        Bitmap resizedBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }


    public Bitmap toGrayScale(Bitmap bmp)
    {
        int width,height;
        height=bmp.getHeight();
        width=bmp.getWidth();
        Bitmap bmpGrayScale = Bitmap.createBitmap(width,height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayScale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmp,0,0,paint);
        return bmpGrayScale;

    }

    public void SaveImage(Bitmap finalBitmap,int i) {
        File root = getApplicationContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File myDir = new File(root + "/saved");
        myDir.mkdirs();
        String fname = "Image-"+ i+".jpg";
        File file = new File (myDir, fname);
        if (file.exists ()) file.delete ();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    private int [] getPixelData(Bitmap tempImage) {
        Log.d(LOG_TAG,"Image Size : " + tempImage.getWidth() + " , " + tempImage.getHeight());
        int [] pixels = new int[tempImage.getWidth() * tempImage.getHeight()];
        tempImage.getPixels(pixels, 0, tempImage.getWidth(), 0, 0, tempImage.getWidth(),tempImage.getHeight());
        int[] retPixels = new int[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            pix = pix & 0xff;
            int b = pix & 0xff;
            retPixels[i] = 0xff - b;
        }
        return retPixels;

    }


    private void classifyData(float[] retPixels) {
        Classification classification = classifier.recognize(retPixels);
        String result = String.format("It's a %s with confidence: %f", classification.getLabel(), classification.getConf());
        Toast.makeText(this, result, Toast.LENGTH_SHORT).show();
    }

    private int[] createPixelsPreview(int[] pixels, float[] retPixels) {
        int[] again = new int[pixels.length];
        for (int a = 0; a < pixels.length; a++) {
            again[a] = ColorConverter.tfToPixel(retPixels[a]);
        }
        return again;
    }

    private float[] createInputPixels(int[] pixels) {
        float[] normalized = ColorConverter.convertToTfFormat(pixels);
        return normalized;
    }

    private void switchPreviewVisibility() {
        bitmapTester.setVisibility(View.VISIBLE);
        drawingView.setVisibility(View.GONE);
    }

    private void reset() {
        drawingView.reset();
        bitmapTester.setVisibility(View.GONE);
        drawingView.setVisibility(View.VISIBLE);
    }

    private void loadModel() {
        executor.execute(() -> {
            try {
                classifier = Classifier.create(getApplicationContext().getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        INPUT_NAME,
                        OUTPUT_NAME);
            } catch (final Exception e) {
                throw new RuntimeException("Error initializing TensorFlow!", e);
            }
        });
    }
}