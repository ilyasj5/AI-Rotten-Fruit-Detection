# AI-Rotten-Fruit-Detection

## About 🤖 🥗
According to the EPA, food retail space and households wasted roughly 66 million tons of food in the United States. As a result, our group wanted to help reduce food waste using modern technology such as Artificial Intelligence (AI). We leveraged a Kaggle dataset of images of common fruits and vegetables such as apples, oranges, and tomatoes in various states of freshness to classify whether they were rotten or not. Doing so, we hope that there can be a broader use of this technology in the real world to help people know their food quality status before consumption, helping users reduce food waste. 

## Experiment Design 🎨
* We first preprocessed the image data using Keras packages in order for the images to be passed into three Neural Network image classification models
  - We resized images to match the input size requirements of the  neural network models, optimizing them for each model's capabilities
  - We also augmented the images through rotations, shifts, zooms, and flips to diversify the dataset and reduce overfitting
 
* We then used TensorFlow and Keras to build a custom Convolutional Neural Network as well as well as pre-existing models such as MobileNet and ResNet
  - The first custom model was created to be used as a baseline and to gain a better understanding of the dataset
  - The MobileNet model was then used because of its relatively small size that does not require large amounts of data to be efficient
  - We then used the ResNet model because it is a direct complement to the MobileNet model since it is larger in size and requires more data to be efficient

## Model Development 👨‍💻
* Baseline CNN: Constructed with multiple convolutional, max-pooling, and dense layers, including dropout to prevent overfitting and an exponential decay learning rate schedule starting at 0.001, reducing by 0.96 every 100,000 steps, trained over 20 epochs using the Adam optimizer
* MobileNet and ResNet Models: Employed with pre-trained ImageNet weights, appended with a global average pooling layer and a dense output layer featuring softmax activation.
  - Training involved two phases: initially freezing pre-existing layers while training new ones for 5 epochs, followed by unfreezing all layers for another 5 epochs of comprehensive training.
  - Used SGD with momentum and Nesterov acceleration - learning rate started at 0.01 and was reduced to 0.0001 for fine-tuning
 * Configured with a batch size of 64 and varied epoch counts (20 for baseline CNN, 10 each for MobileNet and ResNet), using categorical crossentropy as the loss function

## Results 📊

* Custom Convolutional Neural Network (CNN):
  - Achieved a validation accuracy of 89.91%
  - Despite the lowest accuracy, it's beneficial due to its simpler architecture, suitable for real-time applications given its low computational demands and smaller size 
 * ResNet Model:
    - Reached a validation accuracy of 98.16%
    - Uses residual learning for deeper networks and higher accuracy, ideal for complex visual tasks like medical image analysis
    - Some training-validation accuracy fluctuations suggested initial overfitting, but both accuracies remained high at the last epoch
  * MobileNet Model:
    - Best-performing model with the highest validation accuracy of 99.39%
    - Training and validation accuracies show upward trends with minimal gaps, indicating strong generalization
    - Ideal for applications requiring both high accuracy and efficiency, such as mobile or IoT devices
   
## Conclusions 📝
*  By using advanced image classification through Convolutional Neural Networks, we have developed a system that can accurately distinguish between fresh and rotten produce
*  This capability allows users to make informed decisions about food consumption and disposal, which allows them to reduce unnecessary waste
*  Integrating this technology into mobile applications or devices commonly used in households and retail environments could streamline the process of checking food freshness
    - This would enable the general public to use our project and help reduce the amount of food waste in both households and restaurants
 
      

## Further Reading 📚
* Feel free to view our code, report, and presentation in the repo to get a better understanding of our project!
* Graphs for a more visual representation of our results are available there


## Authors ✍️
* Maseel Shah
* Hayden Johnson
* Ilyas Jaghoori
* Drew Hollar
* Yusef Essawi
