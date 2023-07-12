import torch
import mcubes
from termcolor import colored, cprint

import numpy as np

import pytorch3d
import pytorch3d.io
import pytorch3d.ops
import pytorch3d.renderer
from pytorch3d.renderer import (
    AlphaCompositor,
    DirectionalLights,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    NormWeightedCompositor,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
    Textures,
)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import RotateAxisAngle


def init_mesh_renderer(image_size=512, dist=3.5, elev=90, azim=90, camera="0", device="cuda:0"):
    # Initialize a camera
    # With world coordinates +Y up, +X left, and +Z in, the front of the cow is facing the -Z direction
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.

    if camera == "0":
        # for vox orientation
        # dist, elev, azim = 1.7, 20, 20 # shapenet
        # dist, elev, azim = 3.5, 90, 90 # front view

        # dist, elev, azim = 3.5, 0, 135 # front view
        camera_cls = FoVPerspectiveCameras
    else:
        # dist, elev, azim = 5, 45, 135 # shapenet
        camera_cls = FoVOrthographicCameras

    R, T = look_at_view_transform(dist, elev, azim)
    cameras = camera_cls(device=device, R=R, T=T)
    # print(f'[*] renderer device: {device}')

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[1.0, 1.0, 0.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras,
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftPhongShader(
    #         device=device,
    #         cameras=cameras,
    #         lights=lights
    #     )
    # )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras),
    )
    return renderer


############################# END: renderer #######################################


def sdf_to_mesh(sdf, level=0.02, color=None, render_all=False):
    # device='cuda'
    device = sdf.device
    
    # extract meshes from sdf
    n_cell = sdf.shape[-1]
    bs, nc = sdf.shape[:2]
    
    assert nc == 1
    
    nimg_to_render = bs
    if not render_all:
        if bs > 16:
            cprint("Warning! Will not return all meshes", "red")
        nimg_to_render = min(bs, 16)    # no need to render that much
    
    verts = []
    faces = []
    verts_rgb = []
    
    for i in range(nimg_to_render):
        sdf_i = sdf[i, 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mcubes.marching_cubes(sdf_i, level)
        verts_i = verts_i / n_cell - .5
        
        verts_i = torch.from_numpy(verts_i).float().to(device)
        faces_i = torch.from_numpy(faces_i.astype(np.int64)).to(device)
        text_i = torch.ones_like(verts_i).to(device)
        if color is not None:
            for i in range(3):
                text_i[:, i] = color[i]
        
        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)
        
    try:
        p3d_mesh = Meshes(verts, faces, textures=Textures(verts_rgb=verts_rgb))
    except:
        p3d_mesh = None
    
    return p3d_mesh
    
def voxel_to_mesh(voxel, color=None):
    vox_mesh = pytorch3d.ops.cubify(voxel, thresh=0.5)
    verts = vox_mesh.verts_list()
    verts_rgb_list = []
    for i in range(len(verts)):
        verts_rgb = torch.ones_like(verts[i])
        if color is not None:
            for i in range(3):
                verts_rgb[:, i] = color[i]
        verts_rgb_list.append(verts_rgb)
        
    vox_mesh.textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
    return vox_mesh

####################################################################################################

##################################### rendering #########################################
def render_mesh(renderer: MeshRenderer, mesh: Meshes, color=None, norm=True):
    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, C, H, W
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
            # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture

    images = renderer(mesh)
    return images.permute(0, 3, 1, 2)


def render_voxel(mesh_renderer, voxel, render_all=False):
    bs = voxel.shape[0]
    if not render_all:
        nimg_to_render = min(bs, 16)
        # nimg_to_render = min(bs, 16) # no need to render that much..
        voxel = voxel[:nimg_to_render]
    else:
        nimg_to_render = bs

    # render voxel
    meshes = pytorch3d.ops.cubify(voxel, thresh=0.5)
    # verts = meshes.verts_list()[0][None, ...]
    verts_list = meshes.verts_list()
    norm_verts_list = []
    verts_rgb_list = []
    for verts in verts_list:
        # print(f'verts: {verts.min()}, {verts.max()}, {verts.mean()}')
        try:
            verts = (verts - verts.min()) / (verts.max() - verts.min())
        except:
            # quick fix
            images = torch.zeros(nimg_to_render, 4, 256, 256).to(voxel)
            return images

        verts = verts * 2 - 1
        norm_verts_list.append(verts)
        verts_rgb_list.append(torch.ones_like(verts))
    
    meshes.textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
    try:
        images = mesh_renderer(meshes)
        images = images.permute(0, 3, 1, 2)
    except:
        images = torch.zeros(nimg_to_render, 4, 256, 256).to(voxel)
        print('here')
    
    return images

# rendering
def add_mesh_textures(mesh: Meshes):
    verts = mesh.verts_list()
    faces = mesh.faces_list()
    
    bs = len(verts)
    verts_rgb = []
    for i in range(bs):
        verts_rgb.append(torch.ones_like(verts[i]))
    # mesh = Meshes(verts=verts, faces=faces, textures=pytorch3d.renderer.mesh.TexturesVertex(verts_features=verts_rgb))
    mesh.textures = pytorch3d.renderer.mesh.TexturesVertex(verts_rgb)
    return mesh

def render_sdf(mesh_renderer, sdf, level=0.02, color=None, render_imsize=256, render_all=False):
    """ 
        shape of sdf:
        - bs, 1, nC, nC, nC 

        return a tensor of image rendered according to self.renderer
        shape of image:
        - bs, rendered_imsize, rendered_imsize, 4

        ref: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/base_3d.py
    """
    # device = 'cuda'
    device = sdf.device
    bs = sdf.shape[0]
    
    if not render_all:
        nimg_to_render = min(bs, 16)
    
    p3d_mesh = sdf_to_mesh(sdf, level)
    # todo
    
    