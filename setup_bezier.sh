# install torch+cuda 1.9: need compile apex
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install apex
git clone https://github.com/NVIDIA/apex.gitcd apex
python setup.py install
cd ..

# bezier 
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu

python setup_bezier.py build develop
