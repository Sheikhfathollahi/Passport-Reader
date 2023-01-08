import cv2
import imutils
from keras.models import load_model
import numpy as np
import Homomorphic


class PassportInfo():
    def __init__(self, passport_image=0,Pass_image_no_shadow=0, Pass_image_BW=0, Pass_image_BW_cropped=0,
                 BW_Cropped_images = 0 ,Pass_image_segment=0,Passport_Number = 0 ,
                 farsi_digit_model=load_model('Passport_Model2.h5'),
                 reject_Code=0, reject_reason=0):

        self.passport_image             = passport_image
        self.Pass_image_no_shadow   = Pass_image_no_shadow
        self.Pass_image_BW          = Pass_image_BW
        self.Pass_image_BW_cropped  = Pass_image_BW_cropped
        self.BW_Cropped_images      = BW_Cropped_images
        self.Pass_image_segment     = Pass_image_segment
        self.Passport_Number          = Passport_Number
        self.farsi_digit_model      = farsi_digit_model
        self.reject_Code            = reject_Code
        self.reject_reason          = reject_reason


    def RemoveShadow(self):
            image = self.Post_image
            image = cv2.resize(image, (1200, 900))
            image = image[:,:,2]
            homo_filter = Homomorphic.HomomorphicFilter(a = .77, b = .74)
            image_Homo = homo_filter.filter(I=image, filter_params=[5 , 1])
            self.Post_image_no_shadow = image_Homo

            cv2.imshow("image_Homo.jpg", image_Homo)
            cv2.imwrite("image_Homo.jpg" ,image_Homo )
            cv2.waitKey(0)

    def preprocess(self):

        image = self.Post_image_no_shadow
        #image = cv2.resize(image , (1200,900))

        norm_img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        blured = cv2.GaussianBlur(norm_img ,  (5,5), 1)
        image_th = 255 - blured

        self.Post_image_BW = image_th

        cv2.imshow("th", image_th)
        cv2.imwrite("BW_Image.jpg",image_th)
        cv2.waitKey(0)

    def crop_front(self):

            image_th = self.Post_image_BW
            height , width = image_th.shape

############## Passport No. section #######################################

            Passport = np.float32([[385, 5], [385, 185], [3, 3], [3, 185]])
#########################Cropping Passport No. ###################################

            crop = np.float32([[0, 0],  [height,0], [0,width],[height , width]])
            Matrix_Pass = cv2.getPerspectiveTransform(Passport, crop)
            imgout_Pass = cv2.warpPerspective(image_th, Matrix_Pass, (height , width))
            imgout_Pass= cv2.rotate(imgout_Pass, cv2.ROTATE_90_CLOCKWISE)
            homo_filter = Homomorphic.HomomorphicFilter(a=0.7, b=0.7)
            imgout_Pass = homo_filter.filter(I=imgout_Pass, filter_params=[5, 1])
            imgout_Pass = cv2.threshold(imgout_Pass,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


############ Morphological Operation #############################

            #kernel_Pass = np.ones((17,3) , dtype=np.uint8)
            #imgout_Pass = cv2.morphologyEx(imgout_Pass, cv2.MORPH_OPEN, kernel_Pass , iterations = 2)
            #imgout_Pass = cv2.erode(imgout_Pass, kernel_Pass, iterations=3)

##################################################################
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow("output", imgout_Pass)
            cv2.imwrite("imgout_Pass.jpg",imgout_Pass)
            cv2.waitKey(0)
##################################################################

            self.Pass_image_BW_cropped = imgout_Pass



    def segment_front(self):

            M_Passport  = []
            C_Passport = []


            imgout_Pass = self.Pass_image_BW_cropped

            contours, hier = cv2.findContours(imgout_Pass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])


            MedianPass = []
            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)

                if (w * h) > 1000:
                    MedianPass.append(round((y + h + y) / 2))

            MedianPass = np.array(MedianPass)
            MedianPass = np.sort(MedianPass)
            MedianPassC = MedianPass[int(len(MedianPass) / 2)]

            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)
                if (w * h) > 1000:
                    if (((y + h + y) / 2) < MedianPassC + 200 and ((y + h + y) / 2) > MedianPassC - 200):
                        #cv2.rectangle(imgout_Pass, (x, y), (x + w, y + h), (21, 50, 255), 3)
                        #cv2.circle(imgout_Pass, [round((x + w + x) / 2), round((y + h + y) / 2)], 5, (21, 50, 255), 5)
                        roi = imgout_Pass[y:y + h, x:x + w]
                        M_Passport.append(roi)

            # cv2.namedWindow("output1", cv2.WINDOW_NORMAL)
            # cv2.imshow("output1", imgout_Pass)
            # cv2.waitKey(0)
            try:

                for i in range(0,8):
                    C_Passport.append(M_Passport[i])

            except:
                self.reject_reason = " تصویر نامناسب است.(2)"
                self.reject_Code = 2
                return

            self.Pass_image_segment = C_Passport




    def classify_front(self):

            Pass = self.Pass_image_segment
            model = self.farsi_digit_model

############# Special Resizing for Better Predicting #####################

            def true_resize (img):
                img = imutils.resize(img,height=28)
                if img.shape[1]<28:
                    new_w= round((28 - img.shape[1]) /2)
                    center_image = np.zeros((28,28) , np.uint8)
                    center_image[:,new_w:new_w+img.shape[1]] = img
                else:
                    center_image = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                return  center_image

##########################################################################

            def fix_dimension(img):
                new_img = np.zeros((28, 28, 3))
                for i in range(1):
                    new_img[:, :, i] = img
                    return new_img

            dic = {}
            characters = '0123456789'
            for i, c in enumerate(characters):
                dic[i] = c

            output_Pass = []
            for i, ch in enumerate(Pass):
                img_Pass= true_resize(ch)
                # cv2.imwrite("Passport_Raw_Samples/train/test.jpg" ,img_Pass )
                img_Pass = fix_dimension(img_Pass)
                img = img_Pass.reshape(1, 28, 28, 3)
                y_ = model.predict(img)[0]
                classes_Pass = np.argmax(y_, axis=0)
                character = dic[classes_Pass]
                output_Pass.append(character)

            PassPort_Number = ''.join(output_Pass)

            print("PassportNo.:{}".format(PassPort_Number))
            self.Passport_Number = PassPort_Number


    def load_Passport_file(self, file):
        self.Post_image = cv2.imread(file)
        self.RemoveShadow()
        self.preprocess()
        self.crop_front()
        self.segment_front()
        self.classify_front()

if __name__ == '__main__':
    p1 = PassportInfo()
    p1.load_Passport_file("Me.jpg")
