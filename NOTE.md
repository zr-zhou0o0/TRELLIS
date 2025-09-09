# **Dataset Preprocess**
-> DATASET.md

python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE

<!-- ABO 154G 即使--world_size 2000 并不改变download大小 -->
python dataset_toolkits/download.py ABO --output_dir datasets/ABO --rank 1 --world_size 2000
python dataset_toolkits/download.py Toys4k --output_dir datasets/Toys4k --world_size 500
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 40000
python dataset_toolkits/download.py 3D-FUTURE --output_dir datasets/3D-FUTURE --world_size 2000

python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab

<!-- render -->
<!-- 没有sudo权限 但是仍然在tmp下安装上了blender -->
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab

. ./setup.sh --new-env --basic

pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

python dataset_toolkits/voxelize.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab

python dataset_toolkits/extract_feature.py --output_dir datasets/ObjaverseXL_sketchfab

python dataset_toolkits/encode_ss_latent.py --output_dir datasets/ObjaverseXL_sketchfab

cd TRELLIS
git submodule init
git submodule update

. ./setup.sh --kaolin

export CURL_CA_BUNDLE="" && export REQUESTS_CA_BUNDLE=""

python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context"

export HF_ENDPOINT=https://hf-mirror.com

. ./setup.sh --flash-attn

<!-- ERROR: Failed building wheel for flash-attn -->

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

python dataset_toolkits/encode_latent.py --output_dir datasets/ObjaverseXL_sketchfab --world_size 10

python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab

python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab




# **Generate Partial Pointcloud**

python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 40000


python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab 


<!-- 渲染深度图 -->
<!-- dataset_toolkits/blender_script/render.py 中的 'init_nodes' 需要删除 'View Layer' 中间的空格 才能成功运行 -->
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab 



