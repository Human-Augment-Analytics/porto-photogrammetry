"""Shared-SfM leakage prevention and calibrated textured rendering."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.eval.meshroom import dense_graph, filter_sfmdata
from augenblick.eval.mesh_render import Renderer, calibration, image_size
from augenblick.eval.nvs import find_test_dir, render_iteration


def sfm_fixture():
    records, views, poses, intrinsics = [], [], [], []
    for index, name in enumerate(('train_a', 'train_b', 'held_out'), 1):
        center = [float(index), 0, 0]
        records.append(dict(img_name=name, position=center, rotation=np.eye(3).tolist(),
                            fx=50, fy=50, cx=32, cy=24))
        views.append(dict(viewId=index, poseId=index, intrinsicId=index, path='/all/' + name + '.jpg',
                          width=64, height=48))
        poses.append(dict(poseId=index, pose=dict(transform=dict(center=list(center),
                          rotation=np.eye(3).ravel().tolist()))))
        intrinsics.append(dict(intrinsicId=index, type='pinhole', focalLength=50,
                               sensorWidth=64, pixelRatio=1, principalPoint=[0, 0], distortionParams=[]))
    observations = lambda ids: [dict(observationId=i, featureId=i, x=[10, 20]) for i in ids]
    data = dict(views=views, poses=poses, intrinsics=intrinsics,
                featuresFolders=['/all/features'], matchesFolders=['/all/matches'],
                structure=[dict(landmarkId=1, observations=observations([1, 2, 3])),
                           dict(landmarkId=2, observations=observations([1, 3])),
                           dict(landmarkId=3, observations=observations([3]))])
    return data, records


def test_dense_sfm_removes_heldout_views_poses_and_observations():
    data, records = sfm_fixture()
    before = copy.deepcopy(data)
    filtered = filter_sfmdata(data, records, {'train_a', 'train_b'}, Path('/private/images'))
    assert data == before
    assert [v['viewId'] for v in filtered['views']] == [1, 2]
    assert [p['poseId'] for p in filtered['poses']] == [1, 2]
    assert [i['intrinsicId'] for i in filtered['intrinsics']] == [1, 2]
    assert len(filtered['structure']) == 1
    assert [o['observationId'] for o in filtered['structure'][0]['observations']] == [1, 2]
    assert filtered['featuresFolders'] == filtered['matchesFolders'] == []
    assert all(v['path'].startswith('/private/images/') for v in filtered['views'])


def test_wrong_sfm_gauge_is_rejected_before_dense_reconstruction():
    data, records = sfm_fixture()
    data['poses'][0]['pose']['transform']['center'][0] += .1
    with pytest.raises(SceneError, match='centers disagree'):
        filter_sfmdata(data, records, {'train_a', 'train_b'}, Path('/images'))


def graph_fixture():
    edges = {
        'PrepareDenseScene_1': {},
        'DepthMap_1': {'input':'{PrepareDenseScene_1.input}','imagesFolder':'{PrepareDenseScene_1.output}'},
        'DepthMapFilter_1': {'input':'{DepthMap_1.input}','depthMapsFolder':'{DepthMap_1.output}'},
        'Meshing_1': {'input':'{DepthMapFilter_1.input}','depthMapsFolder':'{DepthMapFilter_1.output}'},
        'MeshFiltering_1': {'inputMesh':'{Meshing_1.outputMesh}'},
        'Texturing_1': {'input':'{Meshing_1.output}','imagesFolder':'{DepthMap_1.imagesFolder}', 'inputMesh':'{MeshFiltering_1.outputMesh}'},
        'CameraInit_1': {}, 'StructureFromMotion_1': {}, 'Publish_1': {'output':'/teammate/results'}}
    return {'graph':{n:{'inputs':v} for n,v in edges.items()}}


def test_dense_graph_cannot_publish_into_shared_allview_outputs():
    source = graph_fixture()
    filtered = dense_graph(source, Path('/private/train.sfm'), Path('/private/masks'))
    assert 'Publish_1' not in filtered['graph'] and 'CameraInit_1' not in filtered['graph']
    assert filtered['graph']['PrepareDenseScene_1']['inputs']['input'] == '/private/train.sfm'
    assert source['graph']['Publish_1']['inputs']['output'] == '/teammate/results'


@pytest.mark.parametrize('node,key', [('Texturing_1','input'),('Texturing_1','imagesFolder'),('Texturing_1','inputMesh'),('Texturing_1','inputRefMesh'),('DepthMap_1','input')])
def test_dense_graph_rejects_allview_bypass(node,key):
    graph=graph_fixture();graph['graph'][node]['inputs'][key]='/allview/leak'
    with pytest.raises(SceneError):dense_graph(graph,Path('/private/train.sfm'),Path('/private/masks'))


def test_dense_graph_rejects_alternate_sfm_node():
    graph=graph_fixture();graph['graph']['StructureFromMotion_2']={'inputs':{}}
    with pytest.raises(SceneError):dense_graph(graph,Path('/private/train.sfm'),Path('/private/masks'))


def test_mesh_exports_do_not_invent_a_training_iteration(tmp_path):
    (tmp_path / 'mesh').mkdir()
    assert find_test_dir(tmp_path) == tmp_path / 'mesh'
    assert render_iteration('mesh') is None


def camera():
    return dict(width=64, height=64, fx=40., fy=40., cx=32., cy=32.,
                position=[0., 0., 0.], rotation=np.eye(3).tolist(), camera_model='PINHOLE')


def test_scaled_camera_retains_offcenter_principal_point():
    c = camera()
    c.update(width=128, height=64, fx=80, fy=80, cx=60, cy=30, position=[1, 2, 3])
    k, pose = calibration(c, image_size(c, 2))
    assert np.allclose(k, [[40, 0, 30], [0, 40, 15], [0, 0, 1]])
    assert np.allclose(pose[:3, 3], [-1, -2, -3])


@pytest.mark.parametrize("mesh_frame,udim_tile", [("colmap", 0), ("alicevision_obj", 0), ("alicevision_obj", 1)])
def test_raycaster_renders_texture_orientation_and_background(tmp_path, mesh_frame, udim_tile):
    pytest.importorskip('open3d')
    pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / 'texture.png')
    (tmp_path / 'mesh.mtl').write_text('newmtl surface\nmap_Kd texture.png\n')
    obj = 'mtllib mesh.mtl\nusemtl surface\nv -1 -1 3\nv 1 -1 3\nv 1 1 3\nv -1 1 3\n'
    obj += 'vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n'
    if mesh_frame == 'alicevision_obj':
        lines = []
        for line in obj.splitlines():
            if line.startswith('v '):
                _, x, y, z = line.split()
                line = f'v {x} {-float(y)} {-float(z)}'
            elif line.startswith('vt '):
                _, u, v = line.split()
                line = f'vt {float(u)+udim_tile} {float(v)+udim_tile}'
            lines.append(line)
        obj = '\n'.join(lines) + '\n'
    (tmp_path / 'mesh.obj').write_text(obj)
    rendered = Renderer(tmp_path / 'mesh.obj', mesh_frame).render(camera(), (64, 64))
    assert tuple(rendered[0, 0]) == (0, 0, 0)
    assert rendered[20, 20, 0] > 200 and rendered[20, 20, 1] < 40
    assert rendered[20, 43, 1] > 200 and rendered[20, 43, 0] < 40
    assert rendered[43, 20, 2] > 200 and rendered[43, 20, 0] < 40


def test_raycaster_keeps_two_udim_materials_separate(tmp_path):
    pytest.importorskip('open3d')
    Image.new('RGB',(2,2),(255,0,0)).save(tmp_path/'red.png')
    Image.new('RGB',(2,2),(0,0,255)).save(tmp_path/'blue.png')
    (tmp_path/'mesh.mtl').write_text('newmtl red\nmap_Kd red.png\nnewmtl blue\nmap_Kd blue.png\n')
    # AliceVision OBJ basis: z=-3, with two distinct material/UDIM tiles.
    obj='mtllib mesh.mtl\nv -2 1 -3\nv 0 1 -3\nv 0 -1 -3\nv -2 -1 -3\nv 0 1 -3\nv 2 1 -3\nv 2 -1 -3\nv 0 -1 -3\n'
    obj+='vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nvt 1 2\nvt 2 2\nvt 2 1\nvt 1 1\n'
    obj+='usemtl red\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\nusemtl blue\nf 5/5 6/6 7/7\nf 5/5 7/7 8/8\n'
    (tmp_path/'mesh.obj').write_text(obj)
    image=Renderer(tmp_path/'mesh.obj','alicevision_obj').render(camera(),(64,64))
    assert image[32,16,0]>240 and image[32,16,2]<10
    assert image[32,48,2]>240 and image[32,48,0]<10
