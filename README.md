<div align="center">


# HACL-Net [![python](https://img.shields.io/badge/python-%20%203.7-blue.svg)]()
This is the pytorch implementation of MICCAI 2023, "HACL-Net: Hierarchical Attention and Contrastive Learning Network for MRI-Based Placenta Accreta Spectrum Diagnosis". 
________________________________________________________________

</div>

## Citation

```
Lu, M., Wang, T., Zhu, H., Li, M. (2023). HACL-Net: Hierarchical Attention and Contrastive Learning Network for MRI-Based Placenta Accreta Spectrum Diagnosis. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14226. Springer, Cham. https://doi.org/10.1007/978-3-031-43990-2_29
```



## Dependencies

Our experiments are implemented on the following packages. It is recommended to use anaconda to manage your python packages.

- Ubuntu 16.04
- Python 3.7.11
- PyTorch 1.10 / torchvision 0.11.2
- NVIDIA CUDA 11.3
- Numpy 1.19.5
- scikit-learn 1.0.2
- tqdm 4.62.3
- pandas 1.3.5

## Train

1. Save all patient information, including `pid`, `img_path`, `label` in a csv file.

2. Save each patient's all MRI slices and the corresponding patient-level label into one npz file. For example, a patient with `pid=1`,should correspond to a file named `1.npz`.

3. Specify your hyperparamter in `train/main.py` and train the model.

    ```bash
    bash bin/train.sh
    ```
