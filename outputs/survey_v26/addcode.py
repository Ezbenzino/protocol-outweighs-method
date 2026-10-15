import json,sys,glob,re,os
DB='codes.json'
def norm(s): return ' '.join(s.split())
def add(entries):
    db=json.load(open(DB)) if os.path.exists(DB) else {}
    for e in entries:
        n=e['id']; fn=glob.glob(f'txt/{n}_*.txt')
        t=norm(open(fn[0]).read()) if fn else ''
        for k in ('fus_ev','thr_ev','crop_ev','ext_ev','dice_ev'):
            for q in e.get(k,[]):
                if norm(q) not in t: print('MISSING QUOTE',n,k,q[:80])
        db[n]=e
    json.dump(db,open(DB,'w'),indent=1,ensure_ascii=False)
    print('codes:',len(db))
if __name__=='__main__':
    add(json.load(sys.stdin))
