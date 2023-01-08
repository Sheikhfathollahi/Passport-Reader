# Passport-Number Reader
Optical Character Recognition/Reader (OCR) is the mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo.
In this project a Numerical information of persian passport has been recognized.

![img](Me.jpg)

## First Step
 The first step after loading the image is to remove heavy shadows.
![img](image_Homo.jpg)

## second Step
In the second step we have to apply threshold on image.
![img](Threshold_Image.jpg)

## Third Step
After applying threshold on image, we have to crop the rigon of interest. 
![img](imgout_Pass.jpg)

## Fourth Step
The cropped image then splited into eight sub-images and then feed the model for prediction.
![img](Final_Result.jpg)

## Install requirements

```bash
pip install -r .\requirements.txt
```

