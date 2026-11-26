"""Actual native camera partitions consumed by reconstruction subprocesses."""
import hashlib,json,os,sys
from pathlib import Path
from augenblick.core.errors import SceneError

def record_selection(scene_root, output_root, train, test, scale=1):
    scene_root,output_root=Path(scene_root),Path(output_root)
    split=scene_root/'split.json'
    if not split.is_file(): return
    wanted=json.loads(split.read_text())
    def entry(camera):
        name=Path(str(getattr(camera,'image_name',getattr(camera,'img_name','')))).stem
        if not name: raise SceneError('camera selection has no image name')
        width=getattr(camera,'image_width',getattr(camera,'width',None))
        height=getattr(camera,'image_height',getattr(camera,'height',None))
        mask=scene_root/'masks'/(name+'.png')
        return {'name':name,'width':int(width) if width is not None else None,
                'height':int(height) if height is not None else None,'mask':str(mask),'mask_exists':mask.is_file()}
    a,b=[entry(c) for c in train],[entry(c) for c in test]
    an,bn=[r['name'] for r in a],[r['name'] for r in b]
    if len(set(an))!=len(an) or len(set(bn))!=len(bn) or set(an)&set(bn) or set(an)!=set(wanted['train']) or set(bn)!=set(wanted['test']):
        raise SceneError('native camera partition differs from explicit split')
    out=output_root/'native_selections';out.mkdir(parents=True,exist_ok=True)
    data={'source':'actual native reader cameras','script':str(sys.argv[0]),'resolution_scale':scale,
          'split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),'train':a,'test':b}
    path=out/(Path(sys.argv[0]).stem+'_'+str(scale)+'.json')
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
