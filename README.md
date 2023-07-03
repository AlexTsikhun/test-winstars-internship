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

![disb](https://user-images.githubusercontent.com/83775762/187706461-04ffb675-6475-48c7-a8ef-60fb7cc71838.png)

*Data Collection and Annotation:* data provided by Airbus, you can download them [here](https://www.kaggle.com/competitions/airbus-ship-detection/overview). Many photos have duplicates - they have been removed for better performance.

*Data Preprocessing:* function `preprocess_data()` identifies and counts the non-empty masks, filters out empty images, calculates the file size of each image, and adds columns indicating the presence of ships and the file size. It then balances the dataset by randomly sampling a fixed number of images for each ship count. The function returns a DataFrame with the balanced samples and provides information about the number of masks in the final dataset.

*Model Selection:* between writing your own model and using a pretrained one was chosen to use pretrained model (Transfer Learning approach).

*Model Architecture:* U-net - very suitable for image segmantation tasks. `sm.Unet` imported from `segmentation_models`, backbone architecture is ResNet-34, pre-trained weights from the ImageNet datase. The activation function used in the final layer is the sigmoid activation. Input image shape is 768, output - 128.

*Model Training:* the model is compiled using the Adam  and Jaccard loss, Dice loss - for evaluating the model's performance during training.
Generators (`create_image_generator`) provide batches of images and corresponding segmentation masks during training. Two callbacks are defined: `ModelCheckpoint` saves the model weights whenever there is an improvement in the validation loss; `EarlyStopping` callback stops the training process if there is no improvement in the validation loss for 5 epochs. 10 epoch with 1106 steps.

*Model Evaluation:* eavluated on test data with Dice coefficient.

![dice](https://user-images.githubusercontent.com/83775762/187643841-efde5d72-aa04-45ae-8b5e-3818a90e1f29.png)

*Results:* in the conclusion I made simple model witch try to predict ships (in some pics model work very well, shown above. But the model has room for improvement). (Want to remember, that dice coef=99%. It's very hight, and I don't confidence in about this res...).

![bad](https://user-images.githubusercontent.com/83775762/188190635-4289599d-ef1a-44f9-bc61-2a93ab6851a4.png)

After change epoch (to 99) and training steps (to 50) my NN shows better result! (but it took 9 hours...)

![res_img](https://user-images.githubusercontent.com/83775762/188265259-f6b10136-6501-405b-9983-cf86414f1d5b.png)

![better_img](https://user-images.githubusercontent.com/83775762/188257206-b38ce394-d06d-4af9-9878-cb7165c7fbec.png)

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