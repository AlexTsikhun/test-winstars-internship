# test-winstars-internship
## Test task from R&D Center WINSTARS.AI based on [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)
![img_ship](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoknUBpPgSqMCwahq_I8rKBAO88uhg4yvIXw&usqp=CAU)
---
## Project Structure:
```
├───README.md                           <- The top-level README for developers using this project
|
├───notebooks                           <- Eda
│   ├───winstars_model_inference.ipynb  <- Model inference results
│   ├───winstars_model_train.ipynb      <- Training notebook
│   └───eda_airbus.ipynb                <- Eda notebook
|
├───src                                 <- Code
|   ├───constants.py     
|   ├───inference.py                    <- Model inference
|   ├───model.py                        <- Model training
|   ├───preprocessing.py                <- Constant variables
|   ├───train.py                        <- Useful and repeated func
|   ├───utils.py                        <- Constant variables
|   └───visualization.py                <- Plotting results
|
├───.gitignore                          <- Ignore files
|
└───requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
```
## Solution

Through my work, I have demonstrated my familiarity with CNN and its associated subjects.
### Project setup
*Problem Definition:* detect ships on sateline images and  put an aligned bounding box segment around the ships, dataset is highly imbalanced. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

![image](https://github.com/AlexTsikhun/test-winstars-internship/assets/83775762/2613fa72-2100-4283-a322-367e5760c1a5)

*Data Collection and Annotation:* data provided by Airbus, you can download them [here](https://www.kaggle.com/competitions/airbus-ship-detection/overview). Many photos have duplicates - they have been removed for better performance.

*Data Preprocessing:* function `preprocess_data()` identifies and counts the non-empty masks, filters out empty images, calculates the file size of each image, and adds columns indicating the presence of ships and the file size. It then balances the dataset by randomly sampling a fixed number of images for each ship count. The function returns a DataFrame with the balanced samples and provides information about the number of masks in the final dataset.

*Model Selection:* between writing your own model and using a pretrained one was chosen to use pretrained model (Transfer Learning approach).

*Model Architecture:* U-net - very suitable for image segmantation tasks. `sm.Unet` imported from `segmentation_models`, backbone architecture is ResNet-34, pre-trained weights from the ImageNet datase. The activation function used in the final layer is the sigmoid activation. Input image shape is 768, output - 128.

*Model Training:* the model is compiled using the Adam  and Jaccard loss, Dice loss - for evaluating the model's performance during training.
Generators (`create_image_generator`) provide batches of images and corresponding segmentation masks during training. Two callbacks are defined: `ModelCheckpoint` saves the model weights whenever there is an improvement in the validation loss; `EarlyStopping` callback stops the training process if there is no improvement in the validation loss for 5 epochs. 10 epoch with 1106 steps.

*Model Evaluation:* eavluated on test data with Dice coefficient. (plot below is deprecated)

![dice](https://user-images.githubusercontent.com/83775762/187643841-efde5d72-aa04-45ae-8b5e-3818a90e1f29.png)

*Results:* after a long wait I applied pretrained U-Net model witch to predict ships Metrics - loss: 0.9993 - dice_coef: 1.0000. Model is able to recognize position where is ship, but can't segment area around him. 

Model try to segment ships:

![res_img](https://user-images.githubusercontent.com/83775762/188265259-f6b10136-6501-405b-9983-cf86414f1d5b.png)

Worse result:

![image](https://github.com/AlexTsikhun/test-winstars-internship/assets/83775762/4b3c6be6-1ca7-4a64-a5be-6a6bd22e7c56)

*Future Work:* enter data augmentation (increase diversity of dataset), modification of the architecture, hyperparameter tuning or ensemble techniques (was planned to implement classification model (to classify if there is a ship in the image (CNN)) with segmentation model (to segmentate area around ship (U-Net))).

## Usage
### Train model:
```
cd src
py train.py --train_epoch 99 --train_steps 50 --batch_size 32
```
### Inference model:
```
cd src
py inference.py
```
