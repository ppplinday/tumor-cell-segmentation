# Image-Detection

## Step 1: 判断是否 tumor

原始图像为 2048 x 2048 的 tiff 格式大图，其中 tumor 图像附带 svg 格式的区域标注。

### 1. Image Preprocess

* 将大图分成若干 299 x 299 的小图，进行标注，然后对 tumor 图像做增强。
* 图像裁剪：将原图扩展成 2219 x 2219 (2048 + 299 - 128 = 2219)，然后用 299 x 299的窗口依次裁剪，步长为 128，裁剪得到 17 x 17 = 289 张小图。
* 图像标注：将 svg 格式的标注转换成 2219 x 2219 的图像(canvas)，判断裁剪出的小图中心 (128 x 128) 是否有像素落在 tumor 区域内，若有则标为1，否则为0。
* 图像增强：对 tumor 图像，进行 4 次 rotate 操作，得到 4 个图像；对 4 个图像再做 left-to-right-flip 操作，得到 4 个图像；对以上 8 个图像做 perturb 操作，再得到 8 个图像。同时，裁剪图像的时候做 jitter 操作，增加随机性。
* perturb 操作：利用 tensorflow 的 image 库调整图像的brightness、saturation、hue 和 contrast 参数，其中max_delta分别为 64/255、0.25、0.04 和 0.75。
* jitter 操作：对图像裁剪的起点坐标增加一个 0～8 的随机 offset。
* 用pip安装multiprocess库有助于提高服务器上的运行效率！！！
* Source Code:

    ```shell
    python main_cancer_annotation.py  # Tumor Image Process
    python main_non_cancer_annotation.py  # Normal Image Process
    ```

### 2. Inception v3

* 采用 Inception v3 模型进行训练，输入为 1 中得到的 patch 打包成的 TFRecord 文件，输出为每张 patch 是 tumor 的概率。
* 打包：将所有的 patch 分成两个文件夹：0 和 1，其中 0 为 normal，1 为 tumor。0 和 1 放在 images 目录下。在 images 的同级目录下写入一个`labels.txt`，每行写入 0 和 1，分别代表 images 目录下的两个分类。然后统计 patch 个数，分出 1/10 给 validation，修改`process_medical.sh`脚本中的validation数值。完成后编译：

    ```shell
    cd /path/to/models/inception
    bazel build //inception:process_medical
    ```
    编译完成后运行:
    ```shell
    bazel-bin/inception/process_medical /path/to/input/images
    ```
* 训练：打包完成后进行训练。其中 batch size = 32 * GPU number，learning rate = 0.05，learning rate decay factor = 0.5。编译：

    ```shell
    cd /path/to/models/inception
    bazel build //inception:medical_train_with_eval
    ```
    编译完成后运行：

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
    注意事项：训练结束后，`train_dir` 目录下的文件必须备份到 `pretrained_model_checkpoint_path` 目录下，并修改 checkpoint 文件，使得所有 ckpt 指向 `pretrained_model_checkpoint_path` 目录。
    运行日志保存在`log_file`下，包含 ACC、loss等信息。
* 评估：对模型进行评估。编译：

    ```shell
    cd /path/to/models/inception
    bazel build //inception:medical_eval
    ```
    编译完成后运行：

    ```shell
    bazel-bin/inception/medical_eval \
    --checkpoint_dir=/path/to/restore/checkpoints \
    --eval_dir=/path/to/save/summaries \
    --data_dir=/path/to/input/images \
    --num_example=30000 \
    --subset=validation \
    --eval_file=.txt
    ```
    注意事项：运行日志保存在`eval_file`下，包含 ACC、loss等信息。
* 预测：类似评估的操作。将一张大图裁剪成小图后，打包成 TFRecord 输入评估程序，输出每张 patch 是否 tumor 的概率，根据置信度确定是否tumor。


## Step 2: 计算 tumor 区域占比

### 1. FCN

* 利用 FCN 生成每个小图的 heatmap，计算小图中的 tumor 区域面积。
* 训练：输入为 patch 和对应的 annotation，其中 annotation 为 step 1 中对 canvas 的裁剪。FCN 为基于 VGG Net 改造的全卷积网络。把最后几层全连接层替换成 1 x 1 卷积层，再利用 倒数第三和第四层的 pooling 进行上采样操作，得到 patch 原图大小的输出。其中 learning rate = 0.0001，batch size = 32。运行：

    ```shell
    python FCN.py \
    --batch_size=32 \
    --checkpoint_dir=/path/to/pretrained/models/ \
    --logs_dir=/path/to/save/logs/ \
    --data_dir=/path/to/data/ \
    --mode=train
    ```
* 评估：输入为 patch 和对应的 annotation，输出为预测的 annotation。统计预测的 tumor 区域 S1 与实际的 tumor 区域 S 的交集，和交集与 S1 和 S 的并集的比值。运行：

    ```shell
    python FCN.py \
    --batch_size=32 \
    --checkpoint_dir=/path/to/models/ \
    --logs_dir=/path/to/save/logs/ \
    --data_dir=/path/to/data/ \
    --mode=test \
    --subset=validation
    ```
    
