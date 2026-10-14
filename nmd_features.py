import pandas as pd
from Bio.Seq import Seq
import warnings
from Bio import BiopythonWarning
import numpy as np

# Подавить все предупреждения от Biopython
warnings.simplefilter('ignore', BiopythonWarning)

def check_mapping(seq,start,stop,tid):
    # Первый stop при трансляции от start должен совпасть с annotated stop.
    start5=start[0];stop5=stop[0]
    aa=Seq(seq[start5:]).translate(to_stop=False)
    first_stop=aa.find("*")
    assert first_stop>=0,f"{tid}: no in-frame stop found"
    assert first_stop*3==stop5-start5,\
        f"{tid}: annotated stop={stop5-start5} nt, translated first stop={first_stop*3} nt"
    # Дополнительно проверяем сам stop codon.
    assert seq[stop5:stop5+3] in ("TAA","TAG","TGA"),f"{tid}: invalid stop sequence"

def extract_features(gtf,hg38,transcript_ids):
    g=gtf[gtf.transcript_id.isin(transcript_ids)].copy()
    out=[]

    for tid,x in g.groupby("transcript_id",sort=False):
        strand=x.Strand.iloc[0];chrom=x.Chromosome.iloc[0]

        # Экзоны сортируем в направлении 5'->3' зрелой мРНК.
        ex=x[x.Feature=="exon"].sort_values("Start",ascending=(strand=="+")).reset_index(drop=True)
        ex["t0"]=ex["End"].sub(ex["Start"]).cumsum().shift(fill_value=0)

        # Собираем зрелую мРНК.
        exons = [hg38[chrom][int(r.Start):int(r.End)] for r in ex.itertuples()]
        if strand=='-':
            exons = [i.reverse_complement() for i in exons]
        seq=Seq("").join(exons)
        seq=str(seq).upper()

        # genomic interval -> [start,end) в зрелой мРНК.
        def to_tx(row):
            s,e=int(row.Start),int(row.End);hits=[]
            for r in ex.itertuples():
                a,b=int(r.Start),int(r.End);lo,hi=max(s,a),min(e,b)
                if lo>=hi:continue
                if strand=="+":
                    t0=int(r.t0)+(lo-a);t1=int(r.t0)+(hi-a)
                else:
                    t0=int(r.t0)+(b-hi);t1=int(r.t0)+(b-lo)
                hits.append((t0,t1))
            return sorted(hits)

        # Позиции splice junction в зрелой мРНК.
        junctions=[int(r.t0) for r in ex.itertuples()][1:]

        # [start,end) feature в координатах зрелой мРНК.
        # Работает и для split codon, состоящего из двух GTF-записей.
        def feature_pos(feature):
            rows=x[x.Feature==feature]
            p=[z for r in rows.itertuples() for z in to_tx(r)]
            if not p:return None
            return min(a for a,b in p),max(b for a,b in p)

        start=feature_pos("start_codon");stop=feature_pos("stop_codon")
        if start is None or stop is None:continue

        # Проверяем, что перевод зрелой mRNA правильно воспроизводит annotated stop.
        try:
            check_mapping(seq,start,stop,tid)
        except AssertionError as e:
            print(e)
            continue

        start5=start[0];stop5,stop3=stop

        # Экзон, содержащий 5'-конец stop codon.
        erow=next(r for r in ex.itertuples()
                   if int(r.t0)<=stop5<int(r.t0)+int(r.End-r.Start))
        ex5=int(erow.t0);ex3=ex5+int(erow.End-erow.Start)

        # Все расстояния считаются от 5'-конца stop codon.
        d_ex5=stop5-ex5
        d_ex3=ex3-stop5
        d_start=stop5-start5

        # Ближайший AUG после stop в каждой из трех рамок.
        aug=[i for i in range(stop3,len(seq)-2) if seq[i:i+3]=="ATG"]
        best={0:None,1:None,2:None}
        for i in aug:
            frame=(i-start5)%3
            if best[frame] is None:best[frame]=i

        # Проверяем ORF каждого ближайшего AUG.
        valid=[]
        for i in best.values():
            if i is None:continue
            j=i+3;new_stop=None
            while j+2<len(seq):
                if seq[j:j+3] in ("TAA","TAG","TGA"):
                    new_stop=j;break
                j+=3
            if new_stop is None:continue

            # Новый stop не PTC, если нет junction >50 nt downstream.
            if not any(junc-new_stop>50 for junc in junctions):
                valid.append(i)

        d_aug=min(valid)-stop5 if valid else np.nan

        out.append({"transcript_id":tid,
                    "stop_to_exon5":d_ex5,
                    "stop_to_exon3":d_ex3,
                    "stop_to_start":d_start,
                    "stop_to_downstream_AUG":d_aug})

    return pd.DataFrame(out)

