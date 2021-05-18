# Convolutional_Neural_Network_From_Scratch

## Download VGG16 weights
Weights obtained based on Keras VGG16 model pretrained on ImageNet from official repository: https://github.com/fchollet/keras/blob/44bf298ec3236f4a7281be04716a58163469b4de/keras/applications/vgg16.py
## Running Locally
Clone or download this repository: 

```bash
https://github.com/rayenelayaida/Convolutional_Neural_Network_From_Scratch
```

## Excute the following command on the command prompt

```bash
cd "path_of_the_project"
```

```bash
gcc -O3 -fopenmp -lm Convolutional_Neural_Network.c -o Convolutional_Neural_Network.exe
```

```bash
Convolutional_Neural_Network.exe "vgg16_weights.txt" "test\images_list.txt" "result_prediction.txt"
```



