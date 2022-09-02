# test-winstars-internship
## Test task based on [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)
![img_ship](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSoknUBpPgSqMCwahq_I8rKBAO88uhg4yvIXw&usqp=CAU)
---
## Project Structure:
```
├── LICENSE
├───README.md                           <- The top-level README for developers using this project
│   └─── masks_v2                       <- RLE decoded masks
|
├───eda                                 <- Eda
│   └───eda_airbus.ipynb                <- Eda notebook
|
├───train-infer                         <- Contain kaggle notebooks with training and inferencing
│   ├───winstars_model_inference.ipynb  <- Model inference results
│   ├───winstars_model_train.ipynb      <- Model training results
|
├───winstars-model-inference.py         <- Model inference
|
├───winstars-model-tr.py                <- Model training 
|
├───.gitignore                          <- Ignore files
|
└───requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g.
                                           generated with `pip freeze > requirements.txt`
```
## Solution


<!-- In next step I used:
```python 
$ jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```
for convert .ipynb to .py. -->
In my work I have shown that I am familiar with CNN and related topics. For better result can be used augmentation techniques, deeper NN and tuning hyperparameters - it takes a little more time.
### Setup
Features in the data - class imbalance.

### In solution an error was made! Working in progress

![disb](https://user-images.githubusercontent.com/83775762/187706461-04ffb675-6475-48c7-a8ef-60fb7cc71838.png)

Many photos have duplicates - for better berfomance they can be deleted. Data was split with proportion 80/20 (train/validate)
Unet architecture was chosen - because she is well suited for segmentation tasks.
Model was trained on 10 epoch with batch_size=32, evaluation metric - dice score.
![dice](https://user-images.githubusercontent.com/83775762/187643841-efde5d72-aa04-45ae-8b5e-3818a90e1f29.png)
Model evaluate - 
![dice1000](https://user-images.githubusercontent.com/83775762/187703653-dda21eab-55cf-44a5-907d-9dfbb0c9399a.png)

#### Inference

Picture from my code:
![inference](https://user-images.githubusercontent.com/83775762/187712942-69a04698-d9b4-4625-bc3e-7ef96cf8e81b.png)
![q](https://user-images.githubusercontent.com/83775762/187740999-a7fd9326-6a1f-4303-b7e8-3cfc99f1ccff.png)

If I would to compare original images with prediction, sometimes I had wrong results, model working very bad...(it'll be fixed) My classificator can't distinguish the sea and the shore
![bad](https://user-images.githubusercontent.com/83775762/188190635-4289599d-ef1a-44f9-bc61-2a93ab6851a4.png)


In the conclusion I made simple model witch try to predict ships (in some pics model work very well, shown above). (Want to remember, that dice coef=99%. It's very hight, and I don't confidence in 100% about this res...)

Powerful resources that helped to cope with this task:

* https://www.kaggle.com/kmader/baseline-u-net-model-part-1
* https://www.kaggle.com/code/hmendonca/u-net-model-with-submission
* https://www.kaggle.com/code/ammarnassanalhajali/sartorius-segmentation-keras-u-net-inference/notebook
* https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1/notebook#Run-the-test-data
* https://www.kaggle.com/code/ammarnassanalhajali/sartorius-segmentation-keras-u-net-inference#Prediction
---
## License

Distributed under the MIT License. See LICENSE.txt for more information.

## Acknowledgments

First of all thank you Kaggle for competition and data. Thank you very much to the company [Winstars](https://www.winstars.tech/) than gave such interesting test task. Thanks to the recruiter, who helped me when something was unclear.
