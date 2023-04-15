# Installation

### Prepare the environment

This repository is built in PyTorch 1.12.1 and tested on Linux version 5.4.0-146-generic environment (Python 3.8.13, CUDA Version: 11.4).
**Follow these intructions**

1. Clone our repository

```
git clone https://github.com/JackAILab/MARNet.git
cd MARNet
```

2. Make conda environment

```
conda create -n pytorch1121 python=3.8
conda activate pytorch1121
```

3. Install dependencies

`pip install packages_name` 

or

`conda install packages_name`

#### packages

**Name                    Version                   Build  Channel**

_libgcc_mutex             0.1                        main    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
_openmp_mutex             5.1                       1_gnu    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
_pytorch_select           0.1                       cpu_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
_tflow_select             2.3.0                       mkl  
abseil-cpp                20211102.0           h27087fc_1    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
absl-py                   1.3.0              pyhd8ed1ab_0    conda-forge
addict                    2.4.0                    pypi_0    pypi
aiohttp                   3.8.1            py38h0a891b7_1    conda-forge
aiosignal                 1.3.1              pyhd8ed1ab_0    conda-forge
astunparse                1.6.3              pyhd8ed1ab_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
async-timeout             4.0.2            py38h06a4308_0    https://mirrors.ustc.edu.cn/anaconda/pkgs/main
attrs                     22.1.0             pyh71513ae_1    conda-forge
autopep8                  2.0.0              pyhd8ed1ab_0    conda-forge
blas                      1.0                         mkl    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
blinker                   1.5                pyhd8ed1ab_0    conda-forge
bottleneck                1.3.5            py38h7deecbd_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
brotli                    1.0.9                h5eee18b_7    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
brotli-bin                1.0.9                h5eee18b_7    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
brotlipy                  0.7.0           py38h0a891b7_1004    conda-forge
bzip2                     1.0.8                h7f98852_4    conda-forge
c-ares                    1.18.1               h7f98852_0    conda-forge
ca-certificates           2022.9.24            ha878542_0    conda-forge
cached-property           1.5.2                hd8ed1ab_1    conda-forge
cached_property           1.5.2              pyha770c72_1    conda-forge
cachetools                5.2.0              pyhd8ed1ab_0    conda-forge
certifi                   2022.9.24          pyhd8ed1ab_0    conda-forge
cffi                      1.15.1           py38h74dc2b5_0  
charset-normalizer        2.1.1              pyhd8ed1ab_0    conda-forge
click                     8.1.3           unix_pyhd8ed1ab_2    conda-forge
colorama                  0.4.6              pyhd8ed1ab_0    conda-forge
cryptography              35.0.0           py38h3e25421_2    conda-forge
cudatoolkit               11.3.1               h2bc3f7f_2  
cvxopt                    1.3.0                    pypi_0    pypi
cycler                    0.11.0             pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
dbus                      1.13.18              hb2f20db_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
et_xmlfile                1.1.0            py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
expat                     2.4.9                h6a678d5_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ffmpeg                    4.3                  hf484d3e_0    pytorch
fftw                      3.3.9                h27cfd23_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
flatbuffers               2.0.0                h2531618_0  
fontconfig                2.13.1               h6c09931_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
fonttools                 4.25.0             pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
freetype                  2.11.0               h70c0345_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
frozenlist                1.2.0            py38h7f8727e_0    https://mirrors.ustc.edu.cn/anaconda/pkgs/main
gast                      0.4.0              pyh9f0ad1d_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
giflib                    5.2.1                h7b6447c_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
glib                      2.69.1               h4ff587b_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
gmp                       6.2.1                h58526e2_0    conda-forge
gnutls                    3.6.13               h85f3911_1    conda-forge
google-auth               2.14.1             pyh1a96a4e_0    conda-forge
google-auth-oauthlib      0.4.6              pyhd8ed1ab_0    conda-forge
google-pasta              0.2.0              pyh8c360ce_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
grpc-cpp                  1.46.1               h33aed49_1  
grpcio                    1.50.0                   pypi_0    pypi
gst-plugins-base          1.14.0               h8213a91_2    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
gstreamer                 1.14.0               h28cd5cc_2    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
h5py                      3.1.0           nompi_py38hafa665b_100    conda-forge
hdf5                      1.10.6               h3ffc7dd_1    https://mirrors.ustc.edu.cn/anaconda/pkgs/main
icu                       58.2                 he6710b0_3    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
idna                      3.4                pyhd8ed1ab_0    conda-forge
importlib-metadata        5.0.0              pyha770c72_1    conda-forge
intel-openmp              2021.4.0          h06a4308_3561    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
joblib                    1.2.0              pyhd8ed1ab_0    conda-forge
jpeg                      9e                   h7f8727e_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
keras                     2.10.0           py38h06a4308_0  
keras-preprocessing       1.1.2              pyhd8ed1ab_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
kiwisolver                1.4.2            py38h295c915_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
krb5                      1.19.2               hac12032_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lame                      3.100             h7f98852_1001    conda-forge
lcms2                     2.12                 h3be6417_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ld_impl_linux-64          2.38                 h1181459_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lerc                      3.0                  h295c915_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libblas                   3.9.0            12_linux64_mkl    conda-forge
libbrotlicommon           1.0.9                h5eee18b_7    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libbrotlidec              1.0.9                h5eee18b_7    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libbrotlienc              1.0.9                h5eee18b_7    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libcblas                  3.9.0            12_linux64_mkl    conda-forge
libclang                  15.0.6.1                 pypi_0    pypi
libcurl                   7.87.0               h91b91d3_0  
libdeflate                1.8                  h7f8727e_5    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libedit                   3.1.20210910         h7f8727e_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libev                     4.33                 h516909a_1    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
libevent                  2.1.12               h8f2d780_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libffi                    3.3                  he6710b0_2    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgcc-ng                 11.2.0               h1234567_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgfortran-ng            11.2.0               h00389a5_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgfortran5              11.2.0               h1234567_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgomp                   11.2.0               h1234567_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libiconv                  1.17                 h166bdaf_0    conda-forge
libllvm10                 10.0.1               hbcb73fb_5    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libnghttp2                1.46.0               hce63b2e_0  
libpng                    1.6.37               hbc83047_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libpq                     12.9                 h16c4e8d_3    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libprotobuf               3.20.3               he621ea3_0  
libssh2                   1.10.0               ha56f1ee_2    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
libstdcxx-ng              11.2.0               h1234567_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libtiff                   4.4.0                hecacb30_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libuuid                   1.0.3                h7f8727e_2    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libuv                     1.40.0               h7b6447c_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libwebp                   1.2.4                h11a3e52_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libwebp-base              1.2.4                h5eee18b_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libxcb                    1.15                 h7f8727e_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libxkbcommon              1.0.1                hfa300c1_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libxml2                   2.9.14               h74e7548_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libxslt                   1.1.35               h4e12654_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
lz4-c                     1.9.3                h295c915_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
markdown                  3.4.1              pyhd8ed1ab_0    conda-forge
markupsafe                2.1.1            py38h0a891b7_1    conda-forge
matplotlib                3.5.2            py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
matplotlib-base           3.5.2            py38hf590b9c_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl                       2021.4.0           h06a4308_640    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl-service               2.4.0            py38h7f8727e_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl_fft                   1.3.1            py38hd3c417c_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
mkl_random                1.2.2            py38h51133e4_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
multidict                 6.0.2            py38h0a891b7_1    conda-forge
munkres                   1.1.4                      py_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ncurses                   6.3                  h5eee18b_3    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
nettle                    3.6                  he412f7d_0    conda-forge
ninja                     1.10.2               h06a4308_5    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ninja-base                1.10.2               hd09550d_5    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
nspr                      4.33                 h295c915_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
nss                       3.74                 h0370c37_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
numexpr                   2.8.3            py38h807cd23_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
numpy                     1.24.2                   pypi_0    pypi
oauthlib                  3.2.2              pyhd8ed1ab_0    conda-forge
opencv-contrib-python     4.6.0.66                 pypi_0    pypi
opencv-python-headless    4.6.0.66                 py38_0    fastai
openh264                  2.1.1                h4ff587b_0  
openpyxl                  3.0.10           py38h5eee18b_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
openssl                   1.1.1s               h7f8727e_0    https://mirrors.ustc.edu.cn/anaconda/pkgs/main
opt_einsum                3.3.0              pyhd8ed1ab_1    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
packaging                 21.3               pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pandas                    1.4.4            py38h6a678d5_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pcre                      8.45                 h295c915_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pillow                    6.2.2                    pypi_0    pypi
pip                       22.2.2           py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ply                       3.11                     py38_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
protobuf                  3.19.6                   pypi_0    pypi
pyasn1                    0.4.8                      py_0    conda-forge
pyasn1-modules            0.2.8                    pypi_0    pypi
pycodestyle               2.9.1              pyhd8ed1ab_0    conda-forge
pycparser                 2.21               pyhd8ed1ab_0    conda-forge
pyjwt                     2.6.0              pyhd8ed1ab_0    conda-forge
pyopenssl                 22.0.0             pyhd8ed1ab_1    conda-forge
pyparsing                 3.0.9            py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyqt                      5.15.7           py38h6a678d5_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyqt5-sip                 12.11.0          py38h6a678d5_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pysocks                   1.7.1              pyha2e5f31_6    conda-forge
python                    3.8.13               h12debd9_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-dateutil           2.8.2              pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-flatbuffers        23.1.21            pyhd8ed1ab_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
python_abi                3.8                      2_cp38    conda-forge
pytorch                   1.12.1          py3.8_cuda11.3_cudnn8.3.2_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.1           py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pyu2f                     0.1.5              pyhd8ed1ab_0    conda-forge
pyyaml                    6.0                      pypi_0    pypi
qt-main                   5.15.2               h327a75a_7    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
qt-webengine              5.15.9               hd2b0992_4    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
qtwebkit                  5.212                h4eab89a_4    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
re2                       2022.04.01           h27087fc_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
readline                  8.1.2                h7f8727e_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
requests                  2.28.1             pyhd8ed1ab_1    conda-forge
requests-oauthlib         1.3.1              pyhd8ed1ab_0    conda-forge
rsa                       4.9                pyhd8ed1ab_0    conda-forge
scikit-learn              1.1.2                    pypi_0    pypi
scipy                     1.9.1            py38h14f4228_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
seaborn                   0.12.1                   pypi_0    pypi
setuptools                63.4.1           py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
sip                       6.6.2            py38h6a678d5_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
six                       1.16.0             pyhd3eb1b0_1    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
snappy                    1.1.9                hbd366e4_1    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
sqlite                    3.39.3               h5082296_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tensorboard               2.10.0           py38h06a4308_0  
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1              pyhd8ed1ab_0    conda-forge
tensorboardx              2.5.1                    pypi_0    pypi
tensorflow                2.10.0          mkl_py38hd2379f1_0  
tensorflow-base           2.10.0          mkl_py38hb9daa73_0  
tensorflow-estimator      2.10.0           py38h06a4308_0  
tensorflow-io-gcs-filesystem 0.30.0                   pypi_0    pypi
termcolor                 2.2.0              pyhd8ed1ab_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
threadpoolctl             3.1.0              pyh8a188c0_0    conda-forge
tk                        8.6.12               h1ccaba5_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
toml                      0.10.2             pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tomli                     2.0.1              pyhd8ed1ab_0    conda-forge
torch-tb-profiler         0.4.0                    pypi_0    pypi
torchaudio                0.12.1               py38_cu113    pytorch
torchsummary              1.5.1                    pypi_0    pypi
torchvision               0.13.1               py38_cu113    pytorch
tornado                   6.2              py38h5eee18b_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tqdm                      4.64.1             pyhd8ed1ab_0    conda-forge
typing_extensions         4.3.0            py38h06a4308_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
urllib3                   1.26.11            pyhd8ed1ab_0    conda-forge
werkzeug                  2.2.2              pyhd8ed1ab_0    conda-forge
wheel                     0.37.1             pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
wrapt                     1.14.1           py38h0a891b7_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
xlrd                      2.0.1              pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xlsxwriter                3.0.3              pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xlutils                   2.0.0              pyhd3eb1b0_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xlwt                      1.3.0                    py38_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xz                        5.2.6                h5eee18b_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
yapf                      0.32.0                   pypi_0    pypi
yarl                      1.7.2            py38h0a891b7_2    conda-forge
zipp                      3.10.0             pyhd8ed1ab_0    conda-forge
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.2                ha4553b6_0    http://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main

***

### Format of the dataset

First, the videos are extracted at a rate of one frame per second, and the extracted images are converted into grayscale images. Second, we take all the image data obtained every 20 frames as individual data.

These saved data paths and their labels should be written into **Data/TrainData.csv and Data/TestData.csv**.

The following are template examples for Data/TrainData.csv and Data/TestData.csv.

<img src="/Users/demac/Library/Application Support/typora-user-images/image-20230415120136808.png" alt="image-20230415120136808" style="zoom:33%;" />

tips: The length of the string needs to be consistent with the example.