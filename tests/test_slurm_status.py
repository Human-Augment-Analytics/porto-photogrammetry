import os, subprocess, tempfile
from pathlib import Path

def test_slurm_status_survives_accounting_outages():
    helper = Path(__file__).resolve().parents[1] / "workflow/profiles/hipergator/status-sacct.sh"
    cases=[('RUNNING',0,'',1,'running'),('',0,'123|COMPLETED|',0,'success'),('',0,'123|CANCELLED by 456|',0,'failed'),('',1,'123|OUT_OF_MEMORY|',0,'failed'),('',1,'',1,'running'),('',0,'',0,'running'),('',0,'123.batch|COMPLETED|',0,'running'),('',0,'123|PENDING|',0,'running')]
    with tempfile.TemporaryDirectory() as d:
     for name in ['squeue','sacct']:
      p=Path(d)/name;p.write_text('#!/bin/bash\nprintf "%s\\n" "$'+name.upper()+'_TEXT"\nexit "$'+name.upper()+'_EXIT"\n');p.chmod(0o755)
     for q,qe,a,ae,want in cases:
      env=dict(os.environ,PATH=d+':'+os.environ['PATH'],SQUEUE_TEXT=q,SQUEUE_EXIT=str(qe),SACCT_TEXT=a,SACCT_EXIT=str(ae))
      p=subprocess.run(['bash',str(helper),'123'],env=env,capture_output=True,text=True,check=True)
      assert p.stdout.strip()==want,(p.stdout,want)
