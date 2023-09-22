My current
<a href="pdf/SatoMario_CV_2023.pdf">CV</a>

## Blog: 
---

#### **2023:** What is Data Leakage in Machine Learning (Time series data) 
In this blog, I will explain about what is Data Leakage in Time series data, how it can happen and how you can avoid this to happening in your machine learning model.
Data leakage happens in the moment of feature engineering. It consists of the introduction to the feature of the information that is not available in the moment of prediction.
For example, let's say that you want to predict if the price of a stock will go up or down. Then, you select one specific feature that contains the information of the date x+10. However, your label reflects the result of increase or decrease in the stock value of date x+5. Then you are using an information of the future that in the moment of prediction, it will not be available to you. In other words, in this case, you are including an information to the feature that you are trying to predict.
Some of the best practices to avoid data leakage are the followings:
- Split the data to train and test subsets before any type of preprocessing steps.
- Use the technique of Purging and Embargo to avoid mixing the training and testing dataset information. In other words, even if you separate the dataset to two parts, always include an 'cussion' of data that will not be used between the training and testing datasets.

  continues...

## Computer Science related Projects: 
---

#### **2023:** Deploy your Object Detection app in Streamlit using YOLOv8 model (COCO dataset) 
Github: <a href="https://github.com/mariotsato/YOLOv8_object_detection_streamlit/">HERE</a><br> 
DEMO in Streamlit: <a href="https://mariotsato-yolov8-object-detection-streamlit-app-9gw2rr.streamlit.app/">HERE</a><br>
This is a project of object detection using YOLOv8 model with COCO dataset deployed in streamlit.
<a href="https://github.com/mariotsato/YOLOv8_object_detection_streamlit" class="image fit"><img src="images/obj_detection.png" alt=""></a><br>

---
#### **2023:** Deep Learning - Cat and Dog classification with pre-trained Resnet50
Github: <a href="https://github.com/mariotsato/cat_dog_classification_resnet50">HERE</a><br>
Cat and Dog classification using Jupyter Notebook (ipynb) with pre-trained Resnet50.<br>
<a href="https://github.com/mariotsato/cat_dog_classification_resnet50" class="image fit"><img src="images/dog.png" alt=""></a><br>

---
#### **2022:** Deep Learning - U-Net model applied to the rope detection
Github: <a href="https://github.com/mariotsato/unet_rope_detection">HERE</a><br>
A project carried on to apply the semantic segmentation to the rope detection.<br>
<a href="https://github.com/mariotsato/unet_rope_detection" class="image fit"><img src="images/unet.png" alt=""></a><br>

---
#### **2022:** For and While loop comparison
PDF: <a href="pdf/assignment_2_Sato Mario.pdf">HERE</a><br>
An experiment conducted with statistical analysis of comparison between For and While loop.<br>
Conclusion: For loop is relatively faster to do the counting rather than using while loops.<br>
Check the analysis in the pdf clicking on the image.<br>

<a href="pdf/assignment_2_Sato Mario.pdf" class="image fit"><img src="images/for_while.png" alt=""></a>

---
## Experiences
- [University of Tsukuba] **2022, 2023:** Research Assistant - OPERA project: LAI Index estimation using deep learning and image processing
- [University of Tsukuba] **2022, 2023:** Teaching Assistant - Introduction to Information Science
- [University of Tsukuba] **2022:** Teaching Assistant - Introduction to Programming using Python

---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
