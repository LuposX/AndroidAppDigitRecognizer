package org.pytorch.digitrecognizer;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

  private Bitmap toGrayscale(Bitmap bmpOriginal)
  {
    int width, height;
    height = bmpOriginal.getHeight();
    width = bmpOriginal.getWidth();

    Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    Canvas c = new Canvas(bmpGrayscale);
    Paint paint = new Paint();
    ColorMatrix cm = new ColorMatrix();
    cm.setSaturation(0);
    ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
    paint.setColorFilter(f);
    c.drawBitmap(bmpOriginal, 0, 0, paint);
    return bmpGrayscale;
  }

  private static Bitmap changeBitmapContrastBrightness(Bitmap bmp, float contrast, float brightness)
  {
    ColorMatrix cm = new ColorMatrix(new float[]{contrast, 0, 0, 0,
            brightness, 0, contrast, 0, 0, brightness, 0, 0, contrast,
            0, brightness, 0, 0, 0, 1, 0});

    Bitmap ret = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(),
            bmp.getConfig());/*from  www . j a va  2 s .  c om*/

    Canvas canvas = new Canvas(ret);

    Paint paint = new Paint();
    paint.setColorFilter(new ColorMatrixColorFilter(cm));
    canvas.drawBitmap(bmp, 0, 0, paint);

    return ret;
  }

    public static Bitmap createContrast(Bitmap src, double value) {
        // image size
        int width = src.getWidth();
        int height = src.getHeight();
        // create output bitmap
        Bitmap bmOut = Bitmap.createBitmap(width, height, src.getConfig());
        // color information
        int A, R, G, B;
        int pixel;
        // get contrast value
        double contrast = Math.pow((100 + value) / 100, 2);

        // scan through all pixels
        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                // get pixel color
                pixel = src.getPixel(x, y);
                A = Color.alpha(pixel);
                // apply filter contrast for every channel R, G, B
                R = Color.red(pixel);
                R = (int)(((((R / 255.0) - 0.5) * contrast) + 0.5) * 255.0);
                if(R < 0) { R = 0; }
                else if(R > 255) { R = 255; }

                G = Color.red(pixel);
                G = (int)(((((G / 255.0) - 0.5) * contrast) + 0.5) * 255.0);
                if(G < 0) { G = 0; }
                else if(G > 255) { G = 255; }

                B = Color.red(pixel);
                B = (int)(((((B / 255.0) - 0.5) * contrast) + 0.5) * 255.0);
                if(B < 0) { B = 0; }
                else if(B > 255) { B = 255; }

                // set new pixel color to output bitmap
                bmOut.setPixel(x, y, Color.argb(A, R, G, B));
            }
        }

        // return final image
        return bmOut;
    }

  public Bitmap preprocess_images(Bitmap input_bitmap) {
      Bitmap gray_bitmap = toGrayscale(input_bitmap);
      Bitmap resize_bitmap = Bitmap.createScaledBitmap(gray_bitmap, 100, 100, true); //if filter is set to false worse quality
      Bitmap crop_bitmap = Bitmap.createBitmap(resize_bitmap, (resize_bitmap.getWidth() - 28) / 2, (resize_bitmap.getHeight() - 28) / 2, 28, 28);
      Bitmap out_bitmap = createContrast(crop_bitmap, 20);

      return out_bitmap;
  }

  public String predict_digits(Bitmap input_bitmap) {
      Module module = null;

      try {
          module = Module.load(assetFilePath(this, "cnn_inv_noise.pt"));
      } catch (IOException e) {
          Log.e("PytorchDigitRecognizer", "Error reading assets", e);
          finish();
      }

      // preparing input tensor
      final Tensor inputTensor = TensorImageUtilsExtra.bitmapToFloat32Tensor(input_bitmap);

      // running the model
      final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

      // getting tensor content as java array of floats
      final float[] scores = outputTensor.getDataAsFloatArray();

      // searching for the index with maximum score
      float maxScore = -Float.MAX_VALUE;
      int maxScoreIdx = -1;
      for (int i = 0; i < scores.length; i++) {
          if (scores[i] > maxScore) {
              maxScore = scores[i];
              maxScoreIdx = i;
          }
      }
      return String.valueOf(maxScoreIdx);
  }

  public void updateScreen(Bitmap bitmap, String className) {
      // showing image on UI
      ImageView imageView = findViewById(R.id.image);
      imageView.setImageBitmap(bitmap);

      // showing className on UI
      TextView textView = findViewById(R.id.prediction);
      textView.setText("Predicted digit: " + className);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
    super.onActivityResult(requestCode, resultCode, imageReturnedIntent);

    Bitmap photo = null;

    switch(requestCode) {
      case 0:
        if(resultCode == RESULT_OK) {
            if (resultCode != RESULT_CANCELED) {
                if (imageReturnedIntent != null) {
                    PackageManager m = getPackageManager();
                    PackageInfo p = null;
                    try {
                        p = m.getPackageInfo(getPackageName(), 0);
                    } catch (PackageManager.NameNotFoundException e) {
                        e.printStackTrace();
                    }
                    String dataDir = p.applicationInfo.dataDir;

                    Bundle extras = imageReturnedIntent.getExtras();
                    photo = (Bitmap) extras.get("data");

                    preprocessImage(photo, dataDir);

                    Bitmap bitmap = BitmapFactory.decodeFile(dataDir + "/out.png");
                    updateScreen(bitmap, predict_digits(bitmap));
                }
            }
        }
        break;
      case 1:
          if(resultCode == RESULT_OK) {
              if (resultCode != RESULT_CANCELED) {
                      PackageManager m = getPackageManager();
                  PackageInfo p = null;
                  try {
                      p = m.getPackageInfo(getPackageName(), 0);
                  } catch (PackageManager.NameNotFoundException e) {
                      e.printStackTrace();
                  }
                  String dataDir = p.applicationInfo.dataDir;

                      Uri uri_photo = imageReturnedIntent.getData();
                      try {
                          photo = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri_photo);
                      } catch (IOException e) {
                          e.printStackTrace();
                      }
                      preprocessImage(photo, dataDir);
                  Bitmap bitmap = BitmapFactory.decodeFile(dataDir + "/out.png");
                  updateScreen(bitmap, predict_digits(bitmap));
                  }
              }
        break;
    }
  }

  public void preprocessImage(Bitmap bitmap, String dataDir) {
      //save the bitmap
      try (FileOutputStream out = new FileOutputStream(dataDir + "/photo.png")) {
          bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
          // PNG is a lossless format, the compression factor (100) is ignored
      } catch (IOException e) {
          e.printStackTrace();
      }

      if(!Python.isStarted()) {
          Python.start(new AndroidPlatform(MainActivity.this));
      }

      Python py = Python.getInstance();
      PyObject pym = py.getModule("preprocess");
      PyObject pyf = pym.callAttr("save_image", dataDir);
    }

  public void btnpressed(View view) {

    AlertDialog.Builder alert  = new AlertDialog.Builder(MainActivity.this);
    alert.setTitle("Open Camera or Choose a Image from gallery");
    alert.setPositiveButton("Camera", new DialogInterface.OnClickListener() {
      public void onClick(DialogInterface dialog, int which) {
        Intent takePicture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
          if (takePicture.resolveActivity(getPackageManager()) != null) {
              startActivityForResult(takePicture, 0);//zero can be replaced with any action code (called requestCode)
          }
      } });

    alert.setNegativeButton("Gallery", new DialogInterface.OnClickListener() {
      public void onClick(DialogInterface dialog, int which) {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickPhoto , 1);//one can be replaced with any action
      } });

    alert.show();
  }
}
