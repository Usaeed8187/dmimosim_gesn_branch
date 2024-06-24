## Instructions for setting up sionna environment on Linux

1. Install Anaconda (don't use latest version)
```
>> wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
>> sh Anaconda3-2023.09-0-Linux-x86_64.sh
>> conda config --set auto_activate_base false
```

2. Create Anaconda environment
```
>> conda create -n sionna python=3.11
>> conda activate sionna
```

3. Install Tensorflow 2.15  (2.16 and above not supported yet)
```
>> conda install -c conda-forge tensorflow-gpu=2.15
```

4. Install ipython and Jupyter
```
>> conda install -c conda-forge ipyparallel
>> conda install -c conda-forge jupyter
```

5. Install sionna
```
>> pip install sionna
```

6. Check GPU support in Tensorflow
```
>> python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices('GPU')
```

7. (Optional) Set environment variable to reduce warning messages from Tensorflow. Add the following in your shell init files (.bashrc, .zshrc, etc.)
```
export TF_CPP_MIN_LOG_LEVEL=3
```





