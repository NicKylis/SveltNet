# Sveltnet: An accurate lightweight Sequential CNN model
The Sveltnet model aims to achieve the competent accuracy figures of common CNN models while being much smaller parameter-wise.<br>
This repository holds the testing results of a model on various known datasets. These include:
- [The MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [The fashion MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- [The EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [The CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

The model we used has the following architecture:

![test](res/architecture.png)

The testing results of the model are:

| Test dataset       | Accuracy    | Loss   | Precision | Recall  |
|--------------------|-------------|--------|-----------|---------|
| MNIST              | 99.63%      | 0.5359 | 99.72%    | 99.47%  |
| fashion MNIST      | 89.45       | 0.7564 | 92%       | 86.93%  |
| EMNIST             | 85.68%      | 1.109  | 88.72%    | 81.63%  |
| CIFAR10            | 79.94%      | 0.9675 | 89.69%    | 68.22%  |

## Instructions
- This project features an AI model through a web application, available [HERE](https://letter-recognition-1.onrender.com/)
- The AI model is written in Python using Jupyter Notebook and utilizes CUDA programming lanuage through tensorflow. It can be built both on google colabs or locally.

### Prerequisites (local execution)
- Tensorflow library installed
- A compatible NVIDIA GPU
- UNIX based system (or WSL for windows 10 and 11)

#### Building the model

1. Clone the repository:
```sh
git clone https://github.com/NicKylis/letter_recognition.git
```

2. Compile and run all shells, using your preferable IDE.

### Prerequisites (cloud execution)
- An account on google
- Access to google Colab Pro (optional)

#### Building the model
1. Open the main.ipynb file
2. Click on the __Open in Colab__ button on the top left of the file
3. Navigate to the Execution time (runtime) menu and select Run all (Ctrl+F9)

> **Note**: Remember to adjust the number of epochs to your system's limitations. If you are using google Colabs without Colab Pro access, you might encounter very slow compilation times or even be kicked out from the session mid training. Setting the number of epochs to 10 should be sufficient for more than 96% accuracy.

## Authors
Kylintireas Nikolaos, Lourmpakis Evangelos, Toramanidou Christodouli
