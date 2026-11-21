"""Exercise native training/evaluation resolution against the real workflow/scorer."""
import ast
import shlex
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from augenblick.reconstruction.geosvr import GeosvrBackend,GeosvrConfig
from augenblick.eval import nvs


def workflow_args(tmp_path, shared, extra=None):
    source=(Path(__file__).parents[1]/"workflow/Snakefile").read_text();prefix=source[:source.index("rule all:")]
    nodes=ast.parse(prefix).body
    function=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=="backend_arguments")
    assignment=next(n for n in nodes if isinstance(n,ast.Assign) and any(getattr(x,"id",None)=="EVALUATION_RESOLUTION"for x in n.targets))
    conf={"nvs_resolution":shared,"backend_args":{"geosvr":["--resolution","2",*(extra or [])]}}
    ns={"CONF":conf,"TREE":tmp_path,"SPLIT":"manual","shlex":shlex}
    exec(compile(ast.Module(body=[assignment,function],type_ignores=[]),"workflow_prefix","exec"),ns)
    return shlex.split(ns["backend_arguments"](SimpleNamespace(backend="geosvr",specimen="view"))),ns["EVALUATION_RESOLUTION"]


@pytest.mark.parametrize("shared,expected",[(-1,(64,32)),(2,(32,16))])
def test_training_and_shared_evaluation_resolution_contract(tmp_path,monkeypatch,shared,expected):
    argv,score_resolution=workflow_args(tmp_path,shared)
    c=GeosvrConfig(eval=True,resolution=int(argv[argv.index("--resolution")+1]),
                   evaluation_resolution=int(argv[argv.index("--evaluation_resolution")+1]))
    stages=GeosvrBackend(c).stages(SimpleNamespace(root=tmp_path),tmp_path/"out")
    assert stages[0].cmd[stages[0].cmd.index("--res_downscale")+1]=="2.0"
    evaluation=int(stages[1].cmd[stages[1].cmd.index("--canonical_test_resolution")+1]);assert evaluation==score_resolution
    path=Path(__file__).parents[1]/"src/libs/geosvr/render.py"
    helper=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="canonical_test_view")
    ns={};exec(compile(ast.Module(body=[helper],type_ignores=[]),str(path),"exec"),ns)
    class View:
        image_name="view";image_width=32;image_height=16
        def clone_mini(self):return View()
    images=tmp_path/"images";masks=tmp_path/"masks"
    for d in (images,masks,tmp_path/"renders",tmp_path/"gt"):d.mkdir()
    photograph=images/"view.png";Image.new("RGB",(64,32),(180,170,160)).save(photograph)
    camera,truth=ns["canonical_test_view"](View(),photograph,evaluation)
    assert (camera.image_width,camera.image_height)==truth.size==expected
    for d in ("renders","gt"):truth.save(tmp_path/d/"view.png")
    Image.new("L",(64,32),255).save(masks/"view.png")
    class Distance:
        def __call__(self,a,b,mask=None):return float(((a-b)**2).mean())
    monkeypatch.setattr(nvs.metrics,"Lpips",lambda:Distance());monkeypatch.setattr(nvs.metrics,"Dists",lambda:Distance());monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    result=nvs.score(tmp_path,masks,["view"],tmp_path/"scores.json",canonical_images=images,
                     resolution=score_resolution,require_native_raster=True,require_complete=True)
    assert tuple(result["views"][0]["scored_size"])==expected and not result["views"][0]["render_resampled"]


def test_workflow_prevents_conflicting_backend_evaluation_override(tmp_path):
    with pytest.raises(ValueError,match="workflow owns"):
        workflow_args(tmp_path,-1,["--evaluation_resolution","2"])
