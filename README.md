# Tumor Cell Segmentation 

## Step 1: check the cell whether is tumor cell

Original picture is 2048 x 2048 tiff style, and tumor pictures have regional annotations with SVG format.

### 1. Image Preprocess

* Divide the big picture into a small picture (299 x 299), label them, then do data argument.
* Image cropping: expand the original picture to 2219 x 2219 (2048 + 299 - 128 = 2219),then use 299 x 299 size`s window to crop, step is 128, finally we have 17 x 17 = 289  small size pictures.
* Image label：expand svg style picture to  2219 x 2219 (canvas), check whether has tumor in the small picture after cropping (128 x 128), if yes, labels 1,  otherwise labels 0.
* Image argument：for each tumor image, we have 4 rotate operation and get 4 pictures; for those 4 pictures, let`s do left-to-right-flip operation and get other 4 pictures; for those 8 pictures to do argument again. At the same time, we do jitter operation in order to enhance randomness.
* Perturb operation: used tensorflow`s image library to control brightness、saturation、hue and contrast, max_delta are 64/255、0.25、0.04 and 0.75.
* Jitter operation: for the origin point of cropping, we add 0~8 offset randomly.
* Used pip to set multiprocess in order to improve efficiency.
* Source Code:

    ```shell
    python main_cancer_annotation.py  # Tumor Image Process
    python main_non_cancer_annotation.py  # Normal Image Process
    ```

### 2. Inception v3

* Used Inception v3 to train, input is the patch which are in TFRecord file，output is the probability of tumor.
* All patch will be in two files: 0 and 1, 0 is normal, 1 is tumor. 0 and 1 are in images file. In images, we write in `labels.txt`. Every line is 0 or 1, means two class in the images file. Then, count the number of patch, 1/10 in validation，change the value of validation in `process_medical.sh`. Compile：

    ```shell
    cd /path/to/models/inception
    bazel build //inception:process_medical
    ```
    run:
    ```shell
    bazel-bin/inception/process_medical /path/to/input/images
    ```
* Train: batch size = 32 * GPU number, learning rate = 0.05, learning rate decay factor = 0.5. Compile：

    ```shell
    cd /path/to/models/inception
    bazel build //inception:medical_train_with_eval
    ```
    run:

    ```shell
    bazel-bin/inception/medical_train_with_eval \
    --num_gpus=2 \
    --batch_size=64 \
    --train_dir=/path/to/save/checkpoints/and/summaries \
    --data_dir=/path/to/input/images \
    --pretrained_model_checkpoint_path=/path/to/restore/checkpoints \
    --initial_learning_rate=0.05 \
    --learning_rate_decay_factor=0.5 \
    --log_file=.txt
    ```
    Attention: After train, all the files in `train_dir` need to be copied to `pretrained_model_checkpoint_path`, and change checkpoint file, make all the ckpt point to `pretrained_model_checkpoint_path` .
    Log is in`log_file`, including information of ACC or loss.
* Evaluation. Compile：

    ```shell
    cd /path/to/models/inception
    bazel build //inception:medical_eval
    ```
    run:

    ```shell
    bazel-bin/inception/medical_eval \
    --checkpoint_dir=/path/to/restore/checkpoints \
    --eval_dir=/path/to/save/summaries \
    --data_dir=/path/to/input/images \
    --num_example=30000 \
    --subset=validation \
    --eval_file=.txt
    ```
    Attention: Log is in`log_file`, including information of ACC or loss.
* Prediction:Crop a big picture into small picture, write in the TFRecord file, output the possibility of each pictures whether is has tumor or not.


## Step 2: Calculate the proportion of tumor regions

### 1. FCN

* Using FCN create each small pictures` heatmap, calculate the proportion of tumor regions in the small picture.
* Train: inpout is patch and it`s annotation, annotation is the cropping of step 1. FCN is based on VGG Net. Change last fully connection networks to 1 x 1 convolution network, then use the last 3 and 4 pooling layer to upsample, output the same size`s picture of patch. Learning rate = 0.0001, batch size = 32. Run：

    ```shell
    python FCN.py \
    --batch_size=32 \
    --checkpoint_dir=/path/to/pretrained/models/ \
    --logs_dir=/path/to/save/logs/ \
    --data_dir=/path/to/data/ \
    --mode=train
    ```
* Evaluation: input is patch and it`s annotation, output is expectant annotation. Calculate the proportion of expectant regions and compare the real regions and calculate the accuracy. Run：

    ```shell
    python FCN.py \
    --batch_size=32 \
    --checkpoint_dir=/path/to/models/ \
    --logs_dir=/path/to/save/logs/ \
    --data_dir=/path/to/data/ \
    --mode=test \
    --subset=validation
    ```
    
