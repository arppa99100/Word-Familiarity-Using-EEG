# Word-Familiarity-Using-EEG
In this project we present a tool to predict Word familiarity using EEG signals of the subject.We use brain activation  maps which is a transformation of the recorded EEG signals to train our model. We train a deep recurrent, convolutional neural network inspired by state-of-the-art video classification techniques
to learn robust representations from the sequence of images. This approach preserves the spatial, spectral and temporal structure of EEG[2] which leads to finding features that are less sensitive to variations and distortions within each dimension.  Convolutional neural networks have been used to train a set of images with their corresponding labels. These images are cropped and averaged version of video corresponding to brain activation maps.

The entire basis of this project is the EEG signals which are electroencephalograms or saying in  layman language the Brain Signals. The motivation behind using these signals is BCI the Brain Computer Interface where we try to guess the state of the brain or the intent of the person using just the EEG signals collected through the EMOTIV Epoc+ device.

![emotiv_epoc](https://cloud.githubusercontent.com/assets/8604660/22143452/15c14ede-df20-11e6-98df-88256636f6da.png)


# Approach

We have followed an approach quite similar to one proposed in [2]. Firstly, we collect data using Emotiv, it is device to record electroencephalography. These recorded signals can be visualized using the ‘Emotiv Epoc Brain Activity Map’ software. It demonstrates the brain activation  maps of signals corresponding to all four bands namely, alpha, beta, theta and delta. These brain activation  maps are then fed to the deep learning model proposed in [2] for training and testing.

# Data Collection

We took 25 words of mix difficulties and arranged them in a powerpoint presentation with a 2 second gap between occurrence of two consecutive words during presentation. A total of twelve subjects were called upon for data collection. Subject was given the Emotiv headset to set it on his head and the prepared presentation was run. For each word he saw on the screen, he was directed to write yes if he was familiar with the word, otherwise write no on a piece of paper. During this process, the signals are being recorded by the system which could be simultaneously visualised using the Emotiv Epoc Brain Activity Map software (Fig. 1). This software displays a real-time map of your mental activity in four significant brainwave frequency bands (taken from 
https://www.emotiv.com/product/epoc-brain-activity-map/):

![image](https://cloud.githubusercontent.com/assets/8604660/22143975/08433f9a-df22-11e6-94b2-081d57ca8327.png)

**_Snapshot of brain activation  maps recorded by EMOTIV Epoc Brain Activity Map software_**

# Data transformation

Brain activation  maps recorded by the software are played at 4 different quarters of the window. We play the recorded brain activation maps using the software and simultaneously record the particular area of screen which plays brain activation maps corresponding to just the theta band of the signal. For recording the screen we used the tool available https://showmore.com/ . For each subject we get approximately 105 seconds of brain activation map video corresponding to each of the alpha and theta bands. 

From visual observation, it was observed that significant variations were found only in theta band and to some extent in alpha band. So, we decided to use just the theta band in our first model. In our second model we used pre-trained VGG model and used the weights to get the feature representation of the brain activation  maps.  Lastly, in our third model we used both alpha and theta bands to train our model. 

All the 105 seconds videos are converted to 64 X 64 pixel frames and these frames are then divided into 25 different subset of frames. These subsets represent a single instance of data and these are further divided into 7 parts. All the frames in these parts are aggregated into a single image. So, a group of 7 such images corresponds to single instance of data. 

# Training

Brain activation map images were trained using the model shown in figure 2. In this model, we used 2 different architectures of Convolutional Neural Network (CNN) and Long-Short Term Memory (LSTM) Recurrent Neural Network (RNN).

![imageedit_2_6951072268](https://cloud.githubusercontent.com/assets/8604660/22144571/35c5daa2-df24-11e6-99ef-f6bdf31baa8a.png)

_(Taken from [2]) , Conv: 7- layer Convolutional Neural Network;  Conv NN: 1-D convolution layer across time frames; LSTM: Long-short term memory_

The first version of convolutional neural network follows the architecture shown in figure 3. It takes as input 3-channel images of size 64X64, and is passed to 4 subsequent convolutional layer each of which have 32 filters of 3X3 dimension followed by a 2X2 maxpool layer which reduces width and height to half the original width and height of image. Reduced images are further passed to two subsequent  convolutional layers, each of which have 64 filters of 3X3 dimension followed by another 2X2 maxpool layer and lastly passed to a convolutional layer having 128 filters of 3X3 dimension. Finally we are left with an output of shape: 8X8X128. 

![imageedit_2_5743716135](https://cloud.githubusercontent.com/assets/8604660/22144590/428d7ba0-df24-11e6-82f1-3df7d9d1ca1b.png)

_**Model of 7-layer Convolutional Neural Network**_ 


Output of first version of CNN is input to LSTM as well as to the second version of CNN. In our model we have used 7 cell LSTM. Each cell requires an input from the first version of CNN and from its previous cell, except the first cell which requires input only from the first version of CNN. Output of each cell is fed to its next LSTM cell, except the last cell, whose output is fed to the second version of CNN and to the fully connected layer.

Second version of CNN is simply a 1 layer 1D convolutional neural network which takes input from the first version of CNN and the output is fed to the fully connected layer. Output of fully connected layer is then fed to a softmax layer.

In our first approach, we used only the theta band of brain activation map for training our model. Transformed data (using the technique described Data Transformation section) is fed the first version of CNN for training. Learned weights are then used to test the test data. In our second approach we used pre-trained weights of VGG-16 to assign weights the first version of CNN. Lastly, in our third approach, we used alpha and theta bands of brain activation map for training our model. In this approach we combined and resized (to 64X64) the images obtained after data transformation from videos of alpha and theta bands respectively.

# Results

We employed three models for the task of Word Familiarity, one in which we extract and train the entire convolutional neural network and the other in which we have used a pre-trained model VGG-Net16 to extract the features from a middle convolutional layer. Since our data is skewed towards the positive side we also calculate other performance metrics as well the results are provided in the following table.

![imageedit_2_2257077686](https://cloud.githubusercontent.com/assets/8604660/22144662/8002bcc0-df24-11e6-9b82-222e76c60b61.png)

The first model seems to outperform other models in terms of all the performance metrics conveying that most of the information is contained in the theta band the sudden drop in the accuracy of the model in second case suggests that the spatial and spectral information are more significant than the temporal information though this might be because we have made use of unigram approach.

# Future Work

We can implement a more robust classifier by carefully selecting the words in our dataset. In our current dataset, the words are selected merely at the whim of the person compiling the dataset. It would be a significant step forward if a large set of very different words are used.Interestingly, when aggregating results also, increasing the size of dataset will almost certainly increase our accuracy. Another interesting direction is to combine this method with more modalities, for instance the facial features of the subject. We also plan on concatenating the feature extracted from our model for each frequency band and passing them to the fully connected layer. And employ a N-gram approach to better understand the task of Word Familiarity.

# Conclusion

In this paper, we explore several methods for the analysis of EEG signal to predict word familiarity. A dataset containing recordings of 13 subjects each watching 25 words one after another is utilized for evaluation of these methods. In a binary classification of familiar or not significant results are obtained. While in most cases we were able to predict correctly ,the number of samples limited us from providing a definite effectiveness of our approach. Unfortunately, when using EEG, it is often difficult to obtain much larger sample sizes, due to the limited time participants can use the equipment before fatigue sets in and effectivity of the electrode gel becomes an issue. In addition, having participants take part in multiple sessions can degrade performance as significant differences in brain activity can occur between sessions, due to mood changes, slightly different positioning of electrodes, or even the time of day.  This limits our capabilities to a large extent. 

The results were shown to be significantly better than random classification. Finally, fusion of alpha and delta signals yielded a modest increase in the performance, indicating at least some complementarity to them. It is our hope that other researchers will try their methods to solve this highly challenging problem.


## References

[1].	Berka C, Levendowski D, Cvetinovic M, et al.” Real-time analysis of EEG indices of alertness, cognition, and memory with a wireless EEG headset” Int J Hum Comput Interact 2004; 17:151– 70

[2].	Bashivan, P., Rish, I., Yeasin, M., Codella, N.: Learning representations from EEG with deep recurrent-convolutional neural networks. arXiv preprint arXiv:​1511.​06448 (2015)

[3].	Y. LeCun, K. Kavukcuoglu, and C. Farabet. Convolutional networks and applications in vision. In Circuits and Systems (ISCAS), Proceedings of 2010 IEEE International Symposium on, pages 253–256. IEEE, 2010.

[4].   Krizhevsky, Alex, Sutskever, Ilya, and Hinton, Geoffrey E. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, pp. 1097–1105, 2012. ISSN 10495258.
