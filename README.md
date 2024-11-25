# NeoPolyp

This is just an assignment in my Deep learning course. You can refer more about this competition on Kaggle [here](https://www.kaggle.com/c/bkai-igh-neopolyp/overview).

## Installation
Clone the repository:
```sh
git clone https://github.com/minhduong814/NeoPolyp.git
```
Download `model.zip` file here and put it in the folder named `checkpoint`. You may not need to unzip the file.
```sh
https://drive.google.com/drive/u/0/folders/1LLT6yK7rVimg5W3X-Gs7tnNjn3thUkoa
```

## Inference
To run inference on an image, use the following command. Just replace "image.jpeg" with any other image path.
```sh
python3 infer.py --image_path image.jpeg
```
or try
```sh
python infer.py --image_path image.jpeg
```