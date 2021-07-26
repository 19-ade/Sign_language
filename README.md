# Sign_language

## Description
This is a project which detects the alphabets from the hand gestures based on American Sign Language. There are two models implemented here one trained on the **MNIST** dataset and the other trained on **ASL** dataset . Both the models are CNN architecture . All contributions are welcome .

## Setting up the environment 
### Datasets
  - [MNIST](https://www.kaggle.com/datamunge/sign-language-mnist?select=sign_mnist_train) Dataset
  - [ALS](https://www.kaggle.com/grassknoted/asl-alphabet) Dataset
### Cloning the Repo
`git clone https://github.com/19-ade/Sign_language.git`

Once the repo has been cloned ,the folder with the checkpoints for the ASL Model needs to be downloaded and pasted in the same folder as the project. Due to github size limitation for uploading files I had o take this path . Don't change the name of the folder or the files within. Here's the [link](https://drive.google.com/drive/folders/1zajq-tT7PcV2q2AMXvIpewcXiK_xe4B_?usp=sharing).

Run the Requirements.py script to install all the required libraries.

`python requirements.py`

Run capture.py once everything has been configured and achieved

`python capture.py`

## Model
### MNIST CNN
The CNN model was trained for 15 epochs. The following plots show the variation of accuracy and loss of the validation and training split wrt epochs


![Screenshot from 2021-07-25 14-26-51](https://user-images.githubusercontent.com/64825911/127031204-6a9924f0-9002-47bd-9a09-c950d65ce99c.png)

                                    

![Screenshot from 2021-07-25 14-27-02](https://user-images.githubusercontent.com/64825911/127031211-3f8c20c7-beaa-4edf-8691-5771a16d3093.png)

### ASL CNN
The CNN model was trained for 10 epochs . It is a much more computation-intensive model, so it is advised to use GPU for training the model. The following plots show the variation of accuracy and loss of the validation and training split wrt epochs

![Screenshot from 2021-07-26 17-57-34](https://user-images.githubusercontent.com/64825911/127031919-008aa265-2481-49d8-bb85-4789f387e479.png)



![Screenshot from 2021-07-26 17-57-51](https://user-images.githubusercontent.com/64825911/127031924-8c8eb775-8d05-41e6-9120-8722d5d6e798.png)

## Scope
- The ASL CNN can be modified to learn from RGB data (in our program it is (64 X 64 X 1) dimension, grayscale data). Might imporve the accuracy even more.
- As of now no proper measures have been taken to isolate the hand area from ROI in the opencv Script . Proper algorithms can be added to isolate said hand , remove noise from the data . 
- The red rectangle is a fixed ROI . Perhaps an algorithm can be implemented that can recognise the hand in the video , thus allowing flexibility. 
- The dataset can be expanded to include numbers, or modified to read sentences

## Output (Some Examples)

![Screenshot from 2021-07-26 23-00-49](https://user-images.githubusercontent.com/64825911/127033015-d7de06eb-52a0-4f41-91ea-8abffee17d8b.png)       ![Screenshot from 2021-07-26 23-01-10](https://user-images.githubusercontent.com/64825911/127033019-cdb4cdba-de5c-49a9-a2d1-b4d1a3d70915.png)

![Screenshot from 2021-07-26 23-01-25](https://user-images.githubusercontent.com/64825911/127033021-0676f187-d56f-4c8b-8fd4-2259cfafb9aa.png)       ![Screenshot from 2021-07-26 23-01-34](https://user-images.githubusercontent.com/64825911/127033023-809a6964-81eb-4b72-93ba-7655c246a662.png)




