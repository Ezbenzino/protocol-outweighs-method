import re,glob,sys
P={'FUS':r'(?:radiologist|\breaders?\b|annotator|expert)s?\b.{0,120}(?:annotat|mask|contour|label|ground|segment|delineat|consensus|agree|mark)|consensus|\bunion\b|intersection of|majority vot|STAPLE|\b50 ?%|pylidc|at least (?:two|three|one|2|3)|ground[- ]truth (?:mask|label|annotation|was|is|were)',
   'THR':r'threshold|binari[sz]|argmax|\b0\.5\b',
   'CROP':r'(?:\bpatch(?:es)?\b|\bcrop|\bROIs?\b|region of interest|\bVOIs?\b|volume of interest|bounding box|sub-?image|\bresiz).{0,140}(?:\d+ ?[×x*] ?\d+|nodule|cent(?:er|re)|size|pixel|voxel|slice)|cent(?:er|re)d? (?:on|at|around) (?:the )?(?:nodule|lesion)|\d{2,3} ?[×x*] ?\d{2,3}(?: ?[×x*] ?\d{2,3})? (?:pixel|voxel|patch|crop|ROI|region|sub|image|input)',
   'EVAL':r'(?:Dice|DSC|IoU)\b.{0,160}(?:comput|calculat|evaluat|slice|volume|per[- ]|each|whole|patch|image|case|nodule)'}
SKIP=re.compile(r'intro|related|background|literature|discussion|conclusion|limitation|future|acknowledg|reference|funding|author|conflict|availability|abbreviation|ethic',re.I)
def sents(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z(\[])',t) if s.strip()]
def run(n,maxn=34,width=300):
    fn=glob.glob(f'txt/{n}_*.txt')[0]; L=open(fn).read().split('\n')
    print('#####',n,L[0][7:150])
    sec='';out=[];seen=set()
    for line in L[1:]:
        if line.startswith('## '): sec=line[3:]; continue
        if line.startswith('ABSTRACT'): sec='abstract'
        if SKIP.search(sec): continue
        for s in sents(line):
            tags=[k for k,p in P.items() if re.search(p,s,re.I if k!='EVAL' else 0)]
            if tags and s not in seen:
                seen.add(s); out.append(f"  [{','.join(tags)}|{sec[:18]}] {s[:width]}")
    print('\n'.join(out[:maxn]))
for n in sys.argv[1:]: run(n)
