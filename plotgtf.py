from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Arc


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(slots=True)
class Transcript:

    transcript_id: str
    chrom: str
    strand: str

    start: int
    end: int

    exons: list
    cds: list

    collapsed_ids: list = field(default_factory=list)

    @property
    def cds_start(self):
        if not self.cds:
            return None
        return min(x[0] for x in self.cds)

    @property
    def cds_end(self):
        if not self.cds:
            return None
        return max(x[1] for x in self.cds)


# ============================================================
# GTF PARSER
# ============================================================

def transcripts_from_gtf(gtf):

    transcripts = {}

    for tid, tr in gtf.groupby("transcript_id"):

        exons = (
            tr[tr.type == "exon"]
            .sort_values(["start", "end"])
            [["start", "end"]]
            .itertuples(index=False, name=None)
        )

        cds = (
            tr[tr.type == "CDS"]
            .sort_values(["start", "end"])
            [["start", "end"]]
            .itertuples(index=False, name=None)
        )

        transcripts[tid] = Transcript(
            transcript_id=tid,
            chrom=tr.chrn.iloc[0],
            strand=tr.strand.iloc[0],
            start=int(tr.start.min()),
            end=int(tr.end.max()),
            exons=list(exons),
            cds=list(cds),
        )

    return transcripts


# ============================================================
# COLLAPSE
# ============================================================

def transcript_signature(tr):

    return (
        tuple(tr.exons),
        tuple(tr.cds),
        (tr.start,tr.end)
    )


def collapse_transcripts(transcripts):

    groups = defaultdict(list)

    for tr in transcripts.values():
        groups[transcript_signature(tr)].append(tr)

    collapsed = []

    for members in groups.values():

        rep = members[0]

        rep.collapsed_ids = [
            m.transcript_id
            for m in members
        ]

        collapsed.append(rep)

    return collapsed


# ============================================================
# COORDINATE MAPPING
# ============================================================

@dataclass
class CoordinateSegment:

    start: int
    end: int

    display_start: float
    display_end: float

    @property
    def scale(self):

        return (
            self.display_end - self.display_start
        ) / (
            self.end - self.start
        )


@dataclass
class CoordinateMap:

    segments: list

    def transform(self, x):

        for seg in self.segments:

            if seg.start <= x <= seg.end:

                return (
                    seg.display_start
                    + (x - seg.start) * seg.scale
                )

        if x < self.segments[0].start:
            return self.segments[0].display_start + x - self.segments[0].start

        return self.segments[-1].display_end + x - self.segments[-1].end

    def interval(self, start, end):

        return (
            self.transform(start),
            self.transform(end)
        )

    @classmethod
    def from_transcripts(
        cls,
        transcripts,
        intron_size=50,
    ):

        exons = []

        for tr in transcripts:
            exons.extend(tr.exons)

        boundaries = sorted(
            set(
                x
                for exon in exons
                for x in exon
            )
        )

        intervals = []

        for a, b in zip(
            boundaries[:-1],
            boundaries[1:]
        ):
            intervals.append((a, b))

        position = 0

        segments = []

        for start, end in intervals:

            exon_overlap = False

            for ex1, ex2 in exons:

                if ex1 < end and ex2 > start:
                    exon_overlap = True
                    break

            length = end - start

            if exon_overlap:
                display_length = length
            else:
                display_length = min(
                    intron_size,
                    length
                )

            segments.append(
                CoordinateSegment(
                    start,
                    end,
                    position,
                    position + display_length
                )
            )

            position += display_length

        return cls(segments)


# ============================================================
# LAYOUT
# ============================================================

def compute_layout(transcripts):

    return {
        tr.transcript_id: i
        for i, tr in enumerate(
            reversed(transcripts)
        )
    }


# ============================================================
# COLORS
# ============================================================

def make_color_dict(
    highlight=None,
    default="black",
    cmap="tab10",
):

    if highlight is None:
        return defaultdict(lambda: default)

    cm = plt.get_cmap(cmap)

    colors = {}

    for k, v in highlight.items():

        if isinstance(v, int):
            colors[k] = cm(v)
        else:
            colors[k] = v

    return defaultdict(
        lambda: default,
        colors
    )


# ============================================================
# DRAWING
# ============================================================



def draw_introns(
    ax,
    tr,
    y,
    mapper=None,
    color="black",
):
    if mapper:
        start,end = mapper.transform(tr.start),mapper.transform(tr.end)
    else:
        start,end = tr.start,tr.end

    ax.plot(
        [start,end],
        [y, y],
        lw=1,
        color=color,
        zorder=1,
    )


def draw_exons(
    ax,
    tr,
    y,
    mapper=None,
    color="black",
):

    exons = tr.exons

    if mapper:
        exons = [
            mapper.interval(a, b)
            for a, b in exons
        ]

    for a, b in exons:

        ax.add_patch(
            Rectangle(
                (a, y - 0.12),
                b - a,
                0.24,
                color=color,
                zorder=2,
            )
        )


def draw_cds(
    ax,
    tr,
    y,
    mapper=None,
    color="black",
):

    cds = tr.cds

    if mapper:
        cds = [
            mapper.interval(a, b)
            for a, b in cds
        ]

    for a, b in cds:

        ax.add_patch(
            Rectangle(
                (a, y - 0.22),
                b - a,
                0.44,
                color=color,
                zorder=3,
            )
        )


def draw_arrows(
    ax,
    tr,
    y,
    mapper=None,
    n=20,
    s=20,
    fixed_strand=False
):

    start = tr.start
    end = tr.end

    if mapper:
        start = mapper.transform(start)
        end = mapper.transform(end)

    xs = np.linspace(start, end, n)

    marker = ">" if (tr.strand == "+") or fixed_strand else "<"

    ax.scatter(
        xs,
        [y] * len(xs),
        marker=f"${marker}$",
        color="gray",
        s=s,
        zorder=0,
    )


# ============================================================
# COVERAGE
# ============================================================

def draw_coverage(
    ax,
    coverage,
    mapper=None,
    color="gray",
):

    if coverage is None:
        return

    x = coverage[['start','end']].to_numpy().flatten()
    y = coverage[['coverage','coverage']].to_numpy().flatten()

    if mapper:

        x = np.array([
            mapper.transform(i)
            for i in x
        ])

    ax.fill_between(x,y,color=color,lw=0)




# ============================================================
# JUNCTIONS
# ============================================================

def draw_junctions(
    ax,
    junctions,
    mapper=None,
    color="black",
):

    if junctions is None:
        return

    for row in junctions.itertuples():

        s = row.start
        e = row.end

        if mapper:
            s = mapper.transform(s)
            e = mapper.transform(e)

        width = e - s

        patch = Arc(
            ((s + e) / 2, 0),
            width,
            width / 2,
            theta1=0,
            theta2=180,
            lw=2,
            color=color,
        )

        ax.add_patch(patch)


# ============================================================
# MAIN API
# ============================================================

def plot_gene(
    gtf,
    collapse=True,
    shrink=True,
    intron_size='rel',
    rel_length = 0.5,
    highlight=None,
    figsize=(12, 6),
    ax=None,
    arrows=True,
    arrow_kwargs=None,
    max_tr=np.inf,
):
    if intron_size=='rel':
        aa = gtf[gtf.type=='exon']
        intron_size = int((aa.end-aa.start).median()*rel_length)

    transcripts = list(
        transcripts_from_gtf(gtf).values()
    )

    if collapse:

        transcripts = collapse_transcripts(
            {
                t.transcript_id: t
                for t in transcripts
            }
        )

    mapper = None

    if shrink:

        mapper = CoordinateMap.from_transcripts(
            transcripts,
            intron_size=intron_size,
        )

    layout = compute_layout(transcripts)

    colors = make_color_dict(highlight)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize
        )

    for tr in transcripts:

        y = layout[tr.transcript_id]

        color = colors[tr.transcript_id]

        draw_introns(
            ax,
            tr,
            y,
            mapper,
            color,
        )

        draw_exons(
            ax,
            tr,
            y,
            mapper,
            color,
        )

        draw_cds(
            ax,
            tr,
            y,
            mapper,
            color,
        )

        if arrows:
            kws = dict() if arrow_kwargs is None else arrow_kwargs
            draw_arrows(
                ax,
                tr,
                y,
                mapper,
                **kws,
            )

    ax.set_yticks(
        list(layout.values())
    )

    labels = []

    for tr in transcripts:

        if tr.collapsed_ids:
            if max_tr<len(tr.collapsed_ids):
                label = ",".join(tr.collapsed_ids[:max_tr])+f'..+{len(tr.collapsed_ids)-max_tr}'
            else:
                label = ",".join(tr.collapsed_ids)
        else:
            label = tr.transcript_id
        labels.append(
            label
        )

    ax.set_yticklabels(
        reversed(labels)
    )

    ax.set_ylim(
        -1,
        len(layout)
    )

    return ax, mapper



######
# MISC
######
import io
import subprocess

def read_shell(command, **kwargs):
    proc = subprocess.Popen(command, 
                            shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    output, error = proc.communicate()

    if proc.returncode == 0:
        with io.StringIO(output.decode()) as buffer:
            return pd.read_table(buffer, **kwargs)
    else:
        raise Exception(error.decode())