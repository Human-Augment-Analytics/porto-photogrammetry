"""Execute the actual native neighbor loop with uneven/zero availability."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_neighbor_cap_is_per_camera_and_independent_of_camera_order():
    path=Path(__file__).parents[1]/"src/libs/geosvr/src/dataloader/data_pack.py"
    module=ast.parse(path.read_text());helper=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=="select_neighbor_indices")
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=="DataPack")
    loop=next(n for n in ast.walk(cls) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
              and [getattr(x,"id",None) for x in n.target.elts]==["id","cur_cam"]
              and any(isinstance(c,ast.Call) and getattr(c.func,"id",None)=="select_neighbor_indices" for c in ast.walk(n)))
    base={"np":np};exec(compile(ast.Module(body=[helper],type_ignores=[]),str(path),"exec"),base)
    expected=np.array([8,2,8,0,8,8,8,8,8,8]);n=len(expected)
    angles=np.full((n,n),90.0);distances=np.ones((n,n));np.fill_diagonal(distances,0)
    for i,k in enumerate(expected):angles[i,[j for j in range(n) if j!=i][:k]]=0
    for perm in (np.arange(n),np.array([3,4,1,0,9,8,7,6,5,2])):
        cams=[SimpleNamespace(image_name=str(i),nearest_id=[],nearest_names=[]) for i in perm]
        ns={**base,"self":SimpleNamespace(_cameras={"train":{1.0:cams}}),"dataset_downscale":1.0,
            "angles":angles[np.ix_(perm,perm)],"diss":distances[np.ix_(perm,perm)],
            "multi_view_num":8,"multi_view_max_angle":30,"multi_view_min_dis":.01,"multi_view_max_dis":1.5}
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(path),"exec"),ns)
        assert {int(c.image_name):len(c.nearest_id)for c in cams}==dict(enumerate(expected.tolist()))
        assert ns["multi_view_num"]==8
