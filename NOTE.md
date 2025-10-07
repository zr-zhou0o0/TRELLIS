# **网络连接**
- http连不上github 用ssh下载
- 可以ping通huggingface，但是代码里连接不上，暂时用镜像站
- requests.exceptions.ConnectionError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api/datasets/JeffreyXiang/TRELLIS-500K (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7fd0457c6450>: Failed to establish a new connection: [Errno 101] Network is unreachable'))"), '(Request ID: e6165af7-c129-4866-9929-a2eead03d131)')
    - 找不到代理配置文件，找不到有代理的地址和端口
    - export HF_ENDPOINT=https://hf-mirror.com


# **Dataset Preprocess**
-> DATASET.md

<!-- 配环境 -->
install: . ./dataset_toolkits/setup.sh

python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k
python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE


<!-- ObjaverseXL不报错但是3DFUTURE报错 -->
手动下载csv，读取本地文件。看来确实是读到了镜像站的错误页面。


<!-- ABO 154G 即使--world_size 2000 并不改变download大小 -->
python dataset_toolkits/download.py ABO --output_dir datasets/ABO --rank 1 --world_size 2000
python dataset_toolkits/download.py Toys4k --output_dir datasets/Toys4k --world_size 500
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 40000
python dataset_toolkits/download.py 3D-FUTURE --output_dir datasets/3D-FUTURE --world_size 2000

python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab


<!-- render -->
<!-- 没有sudo权限 但是仍然在tmp下安装上了blender -->
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/render.py 3D-FUTURE --output_dir datasets/3D-FUTURE 
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE


. ./setup.sh --new-env --basic
. ./setup.sh --basic

pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

python dataset_toolkits/voxelize.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/voxelize.py 3D-FUTURE --output_dir datasets/3D-FUTURE
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE

python dataset_toolkits/extract_feature.py --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/extract_feature.py --output_dir datasets/3D-FUTURE
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE

python dataset_toolkits/encode_ss_latent.py --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/3D-FUTURE
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE

cd TRELLIS
git submodule init
git submodule update
或者手动下载并复制上去，无需安装。
考虑到不能联网，所以直接不要init submodule。
如果已经init了：
git submodule deinit -f trellis/representations/mesh/flexicubes
git add .gitmodules
git rm --cached trellis/representations/mesh/flexicubes

. ./setup.sh --kaolin

export CURL_CA_BUNDLE="" && export REQUESTS_CA_BUNDLE=""

python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context"

export HF_ENDPOINT=https://hf-mirror.com

. ./setup.sh --flash-attn

<!-- ERROR: Failed building wheel for flash-attn -->

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

python dataset_toolkits/encode_latent.py --output_dir datasets/ObjaverseXL_sketchfab --world_size 10

python dataset_toolkits/render_cond.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/render_cond.py 3D-FUTURE --output_dir datasets/3D-FUTURE

python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE




# **Generate Partial Pointcloud**

python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab
python dataset_toolkits/download.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --world_size 40000

python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir datasets/ObjaverseXL_sketchfab 

<!-- 渲染depth和mask -->
<!-- dataset_toolkits/blender_script/render.py 中的 'init_nodes' 需要删除 'View Layer' 中间的空格 才能成功运行 -->
python dataset_toolkits/render.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --save_depth --save_mask

<!-- 创建.blend文件，向物体中随机添加 geometry primitive -->
python dataset_toolkits/add_geometry.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --num_primitives 2,5 --primitive_types cube,cylinder,sphere,cone --use_time_seed

<!-- 渲染添加了 geometry primitive 后的depth和mask -->
# TODO 这个scale应该对准物体而非整个场景的中心
python dataset_toolkits/render_geo.py ObjaverseXL \
    --output_dir datasets/ObjaverseXL_sketchfab \
    --num_views 150 \
    --save_depth \
    --save_mask \
    --scale 1.0

<!-- 通过 depth 和 mask 生成 partial pointcloud -->
<!-- 实质上是把添加 geo primitive 后的 normalized scene 转换成 partial pointcloud，而非 raw model （未经normalize的） -->
python dataset_toolkits/create_partial_pointcloud.py ObjaverseXL \
    --output_dir datasets/ObjaverseXL_sketchfab

# TODO
<!-- 导入glb会自动旋转90度，也就是说，物体的方向没有对上 -->

<!-- 确保 meta data 可以记录是否添加 geometry primitive -->
# TODO 在metadata里添加统计 geometry primitive / render geo / create pc 的列

<!-- 记得及时push到GitHub上 -->




# **Generate Partial Pointcloud 3DFUTURE**

- 移动zip到 datasets/3D-FUTURE/raw 下
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE
python dataset_toolkits/download.py 3D-FUTURE --output_dir datasets/3D-FUTURE --world_size 200
python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE

<!-- 记得改gpu序号！ -->
pip install openexr
<!-- python dataset_toolkits/render.py 3D-FUTURE --output_dir datasets/3D-FUTURE --save_depth --save_normal --save_mask --num_views 60  -->

<!-- 先加primitive 每个保存到两个glb里面 分别再渲染-找pc 或者在渲染完整pc的时候先删geo -->
python dataset_toolkits/add_geometry.py 3D-FUTURE --output_dir datasets/3D-FUTURE --num_primitives 4,7 --primitive_types cube,cylinder,sphere,cone --use_time_seed

<!-- 生成无遮挡的pc -->
python dataset_toolkits/render_no-geo.py 3D-FUTURE --output_dir datasets/3D-FUTURE --num_views 60 --save_depth --save_normal --save_mask --scale 1.0
python dataset_toolkits/create_obj_pointcloud.py 3D-FUTURE --output_dir datasets/3D-FUTURE

<!-- 生成有遮挡的pc -->
python dataset_toolkits/render_geo.py 3D-FUTURE --output_dir datasets/3D-FUTURE --num_views 60 --save_depth --save_normal --save_mask --scale 1.0
python dataset_toolkits/exr_to_png.py 3D-FUTURE
python dataset_toolkits/create_partial_pointcloud.py 3D-FUTURE --output_dir datasets/3D-FUTURE

<!-- python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE -->




# **git**
<!-- 只改name和email还是用不了ssh -->
git config user.name "Your Name"
git config user.email "your.email@example.com"
<!-- git认为太草率，不认可这个private key -->
GIT_SSH_COMMAND="ssh -i /home/junfeng/project-zirui/ssh/id_rsa" git push
<!-- 切换回https-->
git remote set-url origin https://github.com/zr-zhou0o0/TRELLIS.git
<!-- fatal: unable to access 'https://github.com/zr-zhou0o0/TRELLIS.git/': Error in the HTTP2 framing layer -->
<!-- 换一个版本，第一次不行，第二次莫名其妙成功了 -->
git config http.version HTTP/1.1
git config http.version HTTP/2.0



TODO
1. 正态



# **Depth to pointcloud camera**

Pose = c2w
Extrinsic = w2c = Pose^{-1}
Intrinsic = [[f_x, 0, W/2], [0, f_y, H/2], [0, 0, 1]]
其中 f_x f_y 是焦距，W H 是图像宽高，W/2 H/2 是主点坐标（通常在图像中心）也是原点的平移距离

pixel_coord = [u, v, 1]^T
Z_c * pixel_coord = Intrinsic @ cam_coord


**1. Render时候的相机C2W W2C Intrinsic；Depth到rgb的计算方式；场景Normalize的变换和逆变换**

## transforms.json

- 遵循Blender的坐标系约定（Y向上，Z向后）
- 并且保存的是相机到世界（c2w）的变换矩阵
- 格式是标准的 4x4 变换矩阵	[[R, t], [0, 0, 0, 1]]

- 产生于 normalize scene：
    "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    "scale": scale,
    "offset": [offset.x, offset.y, offset.z],

- aabb: Axis-Aligned Bounding Box（轴对齐包围盒）。在 normalize_scene() 函数中，场景被标准化了。

- *内参*
- fov = [40 / 180 * np.pi] * num_views # fov全部都设置为40度，再转换为弧度

- blender中的焦距设置的是物理焦距而非像素焦距
- cam.data.lens = 16 / np.tan(view['fov'] / 2) # 利用 fov 计算焦距
- focal_length = (sensor_width / 2) / tan(FOV / 2)

- 但内参矩阵需要的是像素焦距
    H = W = 512 # 图像分辨率
    focal_length = W / (2 * np.tan(fov / 2))
    
- 转换为像素焦距
    focal_length_px_x = focal_length_mm * resolution / sensor_width_mm
    focal_length_px_y = focal_length_mm * resolution / sensor_height_mm
- 主点坐标（图像中心）
    cx = resolution / 2
    cy = resolution / 2
- 内参矩阵
    intrinsic_matrix = [
        [focal_length_px_x, 0, cx],
        [0, focal_length_px_y, cy],
        [0, 0, 1]
    ]
    

- *相机坐标系*
通常遵循右手坐标系
在Blender/OpenGL中，相机坐标系通常是：
X向右，Y向上，Z向后（与观察方向相反）
而在许多计算机视觉库（如OpenCV）和NeRF实现中，通常使用：
X向右，Y向下，Z向前（观察方向）


- *变换矩阵：世界坐标系到相机坐标系*
[ R00 R01 R02 Tx ]
[ R10 R11 R12 Ty ]
[ R20 R21 R22 Tz ]
[ 0   0   0   1  ]


**2. 从 Pixel coord 到 Cam 到 World；从Depth得到z；Denormalize到原位置**
- 像素坐标系到相机坐标系：令 z = 1，则 pixel_coord = Intrinsic @ rays_d, 则 rays_d = Intrinsic^{-1} @ pixel_coord； 其中pixel_coord = [u, v, 1]^T, 这个 rays_d 就是相机射线 
- 相机坐标系到世界坐标系： world_coord = c2w @ rays_d； 这个 world_coord 就是相机射线在世界坐标系下的方向
- 由针孔相机模型，射线起点就是 T
- 由深度图得到 z： z = distance = depth[u, v] * (far - near) + near
- 则点云坐标 world_point = T + z * world_coord



- *注意*
- 第一次normalize是raw到render/add_geometry并存储glb，先normalize再add，所以add之后其实还有可能又不normalized了
- 第二次normalize是加了 geometry primitive 后 render_geo 这里还有一次normalize，但是没有保存 3d asset，所以需要反变换回去。

- blender渲染深度的distance也是z轴距离，而不是沿射线的距离。所以只要像素坐标系中设定z=1即可，不需要normalize向量。




# **Add Geometric Primitives**

```
python dataset_toolkits/add_geometry.py <SUBSET> --output_dir <OUTPUT_DIR> [--num_primitives <NUM_PRIMITIVES>] [--primitive_types <TYPES>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `SUBSET`: The subset of the dataset to process. Options are `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, and `Toys4k`.
- `OUTPUT_DIR`: The directory to save the data.
- `NUM_PRIMITIVES`: The number of random primitives to add to each object. Default is 16.
- `PRIMITIVE_TYPES`: Comma-separated list of primitive types. Options are `cube`, `cylinder`, `sphere`. Default is `cube,cylinder,sphere`.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to add 16 random geometric primitives to the ObjaverseXL (sketchfab) subset:

```
python dataset_toolkits/add_geo_primitive.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab --num_primitives 16 --primitive_types cube,cylinder,sphere
```

This will create new GLB files in `geo/glbs/000-016/` directory with added geometric primitives. The primitives are organized under an empty object named "random_geometry" for easy identification and manipulation.

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab
```

