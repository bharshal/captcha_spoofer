# captcha_spoofer

Captchas are used to verify whether the entity accessing a particular webpage is a human or a computer.

Ususally they look like this, 

![alt text](https://github.com/bharshal/captcha_spoofer/blob/master/data/32Q2G.png)

I made this application just for fun to see if captchas can be beaten by the power of CV and DL.
The verdict? I think so :P

## How to run it:

1) Run `create_train_data.py`
  This will parse through all the images of captchas in data folder and use opencv to detect individual symbols and crop them separately while giving them lables (which is the name of the image file). Thus we'll get our dataset.
2) Run `train.py` 
  This has the model definition (simple 4 layer CNN) and training script. As dataset is limited and model is small, training shouldn't take much time.  
3) Run `test.py`
  This will run inference on some random images from dataset and use ground truth to give accuracy.
  
  
## How it works:

When creating the dataset, it reads every image where the name of the image is the ground truth.
It then separates the 5 symbols in each image using CV techniques like:

Original captcha:
![alt text](https://github.com/bharshal/captcha_spoofer/blob/master/index2.png)

Erosion and morphology to smoothen the image:
![alt text](https://github.com/bharshal/captcha_spoofer/blob/master/index3.png)

Contours detection to detect the symbols:
![alt text](https://github.com/bharshal/captcha_spoofer/blob/master/index.png)
![alt text](https://github.com/bharshal/captcha_spoofer/blob/master/index1.png)

The symbols are cropped and supplied as training set.
![alt text](https://github.com/bharshal/captcha_spoofer/blob/master/train_loss.png)

After model has been trained, it is used for inference after doing same process as above.


P.S. If someone has more labelled data of captchas please email me.



  
  
