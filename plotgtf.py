import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def compress_introns(df, max_intron=200):
    exons = df[df.Feature.isin(['exon','CDS'])].sort_values(['Start','End']).reset_index(drop=True)
    result = []
    offset = 0
    prev_end = None
    for i, row in exons.iterrows():
        s,e = row.Start,row.End
        if i == 0:
            new_s = s
            new_e = e
        else:
            intron = s - prev_end
            displayed = min(intron, max_intron)
            offset += intron - displayed
            new_s = s - offset
            new_e = e - offset
        result.append((new_s, new_e))
        prev_end = e
    exons[['Start','End']] = pd.DataFrame(result,index=exons.index)
    return exons

def plot_transcripts(
    gtf,
    chrom=None,
    start=None,
    end=None,
    gene_id=None,
    collapse=False,
    highlight=None,
    figsize=(12, 0.6),
    exon_height=0.22,
    cds_height=0.45,
    arrow_spacing=800,
    default_color="steelblue",
    ax=None,
    resize_introns=None,
):
    """
    Parameters
    ----------
    gtf : pandas.DataFrame
        DataFrame из pyranges.read_gtf(...).df

    chrom, start, end : genomic interval

    gene_id : str
        Вместо координат можно указать gene_id.

    collapse : bool
        Объединять одинаковые структуры транскриптов.

    highlight : dict
        {transcript_id: color}

    Returns
    -------
    matplotlib Axes
    """

    if highlight is None:
        highlight = {}

    df = gtf.copy()

    # ----------------------------------------------------
    # Выбор области
    # ----------------------------------------------------
    if gene_id is not None:
        g = df[df["gene_id"] == gene_id]

        if len(g) == 0:
            raise ValueError(f"gene_id '{gene_id}' not found")

        chrom = g.iloc[0]["Chromosome"]
        start = g["Start"].min()
        end = g["End"].max()

    if chrom is None:
        raise ValueError("chrom must be specified")

    region = df[
        (df["Chromosome"] == chrom)
        & (df["End"] >= start)
        & (df["Start"] <= end)
    ].copy()

    if isinstance(resize_introns,int):
        region = compress_introns(region,resize_introns)
        start = region["Start"].min()
        end = region["End"].max()

    transcripts = sorted(region["transcript_id"].dropna().unique())

    # ----------------------------------------------------
    # Собираем структуры транскриптов
    # ----------------------------------------------------
    models = []

    for tx in transcripts:

        tx_df = region[region["transcript_id"] == tx]

        exons = tx_df[tx_df["Feature"] == "exon"][
            ["Start", "End"]
        ].sort_values("Start")

        cds = tx_df[tx_df["Feature"] == "CDS"][
            ["Start", "End"]
        ].sort_values("Start")

        strand = tx_df["Strand"].iloc[0]

        color = highlight.get(tx, default_color)

        exon_tuple = tuple(map(tuple, exons.values))
        cds_tuple = tuple(map(tuple, cds.values))

        models.append(
            dict(
                transcripts=[tx],
                exons=exon_tuple,
                cds=cds_tuple,
                strand=strand,
                color=color,
            )
        )

    # ----------------------------------------------------
    # Collapse
    # ----------------------------------------------------
    if collapse:

        groups = {}

        for m in models:

            key = (
                m["exons"],
                m["cds"],
                m["strand"],
                m["color"],  # важно для highlight
            )

            if key not in groups:
                groups[key] = m.copy()
            else:
                groups[key]["transcripts"].extend(
                    m["transcripts"]
                )

        models = list(groups.values())

    # ----------------------------------------------------
    # Figure
    # ----------------------------------------------------
    if ax is None:
        fig_height = max(1.5, figsize[1] * len(models))
        fig, ax = plt.subplots(
            figsize=(figsize[0], fig_height)
        )

    # ----------------------------------------------------
    # Рисование
    # ----------------------------------------------------
    y_positions = np.arange(len(models))[::-1]

    for y, model in zip(y_positions, models):

        strand = model["strand"]
        color = model["color"]

        exon_centers = [
            (s + e) / 2
            for s, e in model["exons"]
        ]

        if len(exon_centers) > 1:
            ax.plot(
                [min(exon_centers), max(exon_centers)],
                [y, y],
                color=color,
                lw=1,
                zorder=1,
            )

        # exon
        for s, e in model["exons"]:

            left = max(s, start)
            right = min(e, end)

            ax.add_patch(
                Rectangle(
                    (left, y - exon_height / 2),
                    right - left,
                    exon_height,
                    facecolor=color,
                    edgecolor="black",
                    lw=0.5,
                    zorder=2,
                )
            )

        # CDS
        for s, e in model["cds"]:

            left = max(s, start)
            right = min(e, end)

            ax.add_patch(
                Rectangle(
                    (left, y - cds_height / 2),
                    right - left,
                    cds_height,
                    facecolor=color,
                    edgecolor="black",
                    lw=0.5,
                    zorder=3,
                )
            )

        # стрелки направления
        xs = np.arange(start, end, arrow_spacing)

        if strand == "-":
            dx = -arrow_spacing * 0.25
        else:
            dx = arrow_spacing * 0.25

        for x in xs:
            ax.arrow(
                x,
                y,
                dx,
                0,
                width=0.0,
                head_width=0.08,
                head_length=abs(dx) * 0.5,
                length_includes_head=True,
                color=color,
                zorder=0,
            )

        label = ",".join(model["transcripts"])

        ax.text(
            start - (end - start) * 0.01,
            y,
            label,
            ha="right",
            va="center",
            fontsize=8,
        )

    # ----------------------------------------------------
    # оформление
    # ----------------------------------------------------
    ax.set_xlim(start, end)
    ax.set_ylim(-1, len(models))

    ax.set_yticks([])
    ax.set_xlabel(f"{chrom}:{start:,}-{end:,}")

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax