"""Compare the JS scoring ports with the real Python client on random inputs."""
import json, random, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace as NS

SITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / "generated"))
import numpy as np
from evaluation import _auroc
from extract_labels import GroundTruthGazeComparator

random.seed(7)
tmp = Path(tempfile.mkdtemp())
cases = []
for c in range(400):
    rows, labels = ["start_time,end_time,label_type"], []
    for _ in range(random.randint(1, 8)):
        s = round(random.uniform(0, 8), 2); e = round(s + random.uniform(0.1, 4), 2)
        src = random.randint(1, 3); tgt = random.choice([None, 1, 2, 3])
        rows.append(f"{s},{e},P{src}Gaze{'Robot' if tgt is None else 'P' + str(tgt)}")
        labels.append({"start": s, "end": e, "src": src, "target": tgt})
    path = tmp / f"{c}.csv"; path.write_text("\n".join(rows) + "\n")
    gt = GroundTruthGazeComparator(path)
    t = round(random.uniform(0, 12), 2)
    edges = []
    for src in (1, 2, 3):
        look = random.choice([None, *[d for d in (1, 2, 3) if d != src]])
        for d in (1, 2, 3):
            if d != src:
                edges.append({"from": src, "to": d, "isLooking": d == look, "probability": round(random.random(), 3)})
    update = NS(playback_timestamp_ns=int(t * 1e9), interactions=[NS(from_person_id=e["from"], to_person_id=e["to"], is_looking=e["isLooking"], probability=e["probability"]) for e in edges])
    out = []
    for actual, expected, _p in gt.compare_update(update):
        pair = actual or expected
        kind = "false_negative" if actual is None else ("wrong_target" if expected else "false_positive")
        out.append([pair[0], pair[1], kind])
    cases.append({"labels": labels, "t": t, "edges": edges, "python": sorted(out)})

aucs = []
for _ in range(100):
    n = random.randint(4, 60)
    p = [round(random.random(), 1) for _ in range(n)]; y = [random.random() < 0.4 for _ in range(n)]
    aucs.append({"p": p, "y": y, "python": _auroc(np.array(p), np.array(y))})
(tmp / "cases.json").write_text(json.dumps({"cases": cases, "aucs": aucs}))
js = """
const M=require(process.argv[1]);const d=JSON.parse(require('fs').readFileSync(process.argv[2]));let bad=0;
for(const c of d.cases){const r=M.compareEdges(c.labels,c.t,c.edges).map(m=>[m.from,m.to,m.kind]).sort();
 const sort=a=>JSON.stringify(a.slice().sort((x,y)=>(x+'').localeCompare(y+'')));if(sort(r)!==sort(c.python)){bad++;if(bad<4)console.log(JSON.stringify(c),r)}}
let auc=0;for(const a of d.aucs){const v=M.auroc(a.p,a.y);if((v===null)!==(a.python===null)||(v!==null&&Math.abs(v-a.python)>1e-9))auc++}
console.log('compare mismatches',bad,'of',d.cases.length,'| auroc mismatches',auc,'of',d.aucs.length);process.exit(bad||auc?1:0)
"""
sys.exit(subprocess.call(["node", "-e", js, str(SITE / "assets/js/model.js"), str(tmp / "cases.json")]))
