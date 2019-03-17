# Flipkart_object_localization


## [Our Solution for the Flipkart Grid Engineering Campus Challenge](https://dare2compete.com/o/Flipkart-GRiD-Teach-The-Machines-2019-74928)
   
  - Team name
  
       - **using_keras_as_backend**
  
  - Members
  
      - [Mukul Ranjan](https://github.com/mukul54)
  
      - [Rangithala Mahesh](https://github.com/Mahesh1735)
                   
      - [Jayant Praksh Singh](https://github.com/jayantp07)
  
  - Score(IOU) 
  
       - 0.90274 (in level 3)
  
       - 0.696396(in level 2)
  
  - Architecture
  
     - Modified ResNet50(for level 3 we tried resnext101 as well but could not train due to GPU limitations)
     
     - A simple CNN model(for level 2)

     
## Problem
   We are given 24k images(around 15gb) of dimension 480x640 and the 4 co-ordinates(of the upper left corner and the lower right corner) of the bounding box as the label.We have to use supervised machine learning algorithm to predict the bounding box of the test images.
   
   Here is the actual bounding box(in blue) and the predicted bounding box(in red) from the traing set.
   
   ![given bounding box in blue while predicted in red](https://github.com/mukul54/Flipkart_object_localization/blob/master/images/pred_det.png)

## Ideas Which Boosted Our Accuracy and helped in training
   
   - **Adding 4th Channel:**
   
       Since this is an object localization problem where we are trying to predict just the edges of the object so we thought it might be useful to add an extra channels in our image which just contain the edges of the image. We found [cv2.canny](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html) to be useful for this purpose. Due to lack of time we just trained our image by resizing it to 64x64 for 2nd level and in just 3 layered network we got a score of around 0.62. But after adding extra channel our score boosted by 0.07 to 0.696396. This motivated us to use the 4th channel even in the 3rd level where we used 224x224x4 image.
     
   - **Reducing the number of channels in the hidden layers of resnet50:**
        
        Since we were limited by the resources(GPU and Graphics memory) and time (due to coming midsem) we thought using large number of channels won't be helpful. The major reason was again the problem, since it tis not the classification problem our network need not learn a large number of features. More channels make it sure that a large number of features are learnt in case of classification problem. Our problem was more like a regression problem with four output. We didn't check the actual accuracy but reducing the number of channels in resnet50 indeed halved the number of features in resnet50 from around 11M to 5.9M which fasten the training process.
        
Here is the architecture of our model.
        
   ![our model architecture](https://github.com/mukul54/Flipkart_object_localization/blob/master/images/model.png)
        
