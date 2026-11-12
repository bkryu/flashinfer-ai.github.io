"""Draw reproducible, CPU-only GEMM illustrations of batch invariance."""
import argparse
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.rcParams.update({
    "font.family":"DejaVu Sans", "svg.fonttype":"none",
    "svg.hashsalt":"flashinfer-batch-invariance", "savefig.facecolor":"white",
})

A=((1,0,2),(0,1,1),(2,1,0),(1,2,1))
B=((1,2),(0,1),(2,-1))
ROW_COLORS=("#2563eb","#c77c0a","#8656c6","#078478")
ROW_FILLS=("#e5edfc","#fff0d7","#eee5f9","#dcf2eb")
INK="#24312b"
MUTED="#617069"
PERMUTATION=(2,0,3,1)
PARTITIONS=((0,1),(2,3))


def product(rows):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(2)] for i in rows]


def bf16_bits(value):
    raw=struct.unpack(">I",struct.pack(">f",float(value)))[0]
    # The toy's small integer outputs have exact BF16 representations.
    assert raw & 0xffff == 0
    return f"0x{raw >> 16:04x}"


OUTPUT=tuple(tuple(bf16_bits(x) for x in row) for row in product(range(4)))


def matrix(ax,x,cy,values,ids=None,cell_w=.72,row_h=.53,bit_cells=False):
    rows,cols=len(values),len(values[0])
    h,w=rows*row_h,cols*cell_w
    for r,row in enumerate(values):
        y=cy+h/2-(r+1)*row_h
        fill=ROW_FILLS[ids[r]] if ids is not None else "#f0f2ef"
        edge=ROW_COLORS[ids[r]] if ids is not None else "#9ba59c"
        ax.add_patch(Rectangle((x,y),w,row_h,facecolor=fill,edgecolor="white",lw=1.4))
        if ids is not None:
            ax.add_patch(Rectangle((x,y),.055,row_h,facecolor=edge,edgecolor="none"))
            ax.text(x-.17,y+row_h/2,f"r{ids[r]}",ha="right",va="center",
                    fontsize=12,color=edge,weight="bold")
        for c,value in enumerate(row):
            if c:
                ax.plot([x+c*cell_w,x+c*cell_w],[y+.07,y+row_h-.07],
                        color="#d9dfd6",lw=.7)
            ax.text(x+(c+.5)*cell_w,y+row_h/2,str(value),ha="center",va="center",
                    fontsize=13.5 if bit_cells else 14,color=INK,
                    fontfamily="DejaVu Sans Mono" if bit_cells else "DejaVu Sans")
    ax.add_patch(Rectangle((x,cy-h/2),w,h,fill=False,edgecolor="#a7b0a4",lw=.85))


def lane(ax,cy,rows,a_label,c_label):
    ids=tuple(rows)
    top=cy+max(len(ids),3)*.53/2+.25
    ax.text(2.25,top,a_label,ha="center",va="bottom",fontsize=13,weight="bold",color=INK)
    ax.text(5.5,top,"B · fixed weights",ha="center",va="bottom",fontsize=13,weight="bold",color=INK)
    ax.text(9.8,top,c_label,ha="center",va="bottom",fontsize=13,weight="bold",color=INK)
    matrix(ax,1.17,cy,[A[i] for i in ids],ids,cell_w=.72)
    matrix(ax,4.75,cy,B,cell_w=.75)
    matrix(ax,8.0,cy,[OUTPUT[i] for i in ids],ids,cell_w=1.8,bit_cells=True)
    ax.text(4.02,cy,"×",ha="center",va="center",fontsize=29,color=INK)
    ax.text(6.97,cy,"=",ha="center",va="center",fontsize=27,color=INK)


def footer(ax,y,text):
    ax.add_patch(FancyBboxPatch((.67,y-.28),11.5,.56,boxstyle="round,pad=.07",
                               facecolor="#edf5e8",edgecolor="#cbdac0",lw=.8))
    ax.text(6.42,y,text,ha="center",va="center",fontsize=13,color="#365c20",
            fontfamily="DejaVu Sans Mono")


def save(fig,out,name):
    for ext in ("svg","png"):
        path=out/f"{name}.{ext}"
        fig.savefig(path,bbox_inches="tight",dpi=170,pad_inches=.18)
        if ext=="svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def permutation_figure(out):
    fig,ax=plt.subplots(figsize=(13,7.8))
    ax.set(xlim=(0,12.8),ylim=(0,8.6));ax.axis("off")
    ax.text(.55,8.3,"Permutation invariance",fontsize=19,weight="bold",color=INK)
    ax.text(.55,7.85,"Reorder the token rows. Each token keeps the same output bits.",fontsize=13,color=MUTED)
    lane(ax,5.8,range(4),"A · original order","C · BF16 output bits")
    lane(ax,2.15,PERMUTATION,"P A · permuted order","P C · same row bits")
    for x,label in [(2.25,"P: [2, 0, 3, 1]"),(9.8,"same permutation P")]:
        ax.add_patch(FancyArrowPatch((x,4.5),(x,3.65),arrowstyle="-|>",
                                     mutation_scale=16,lw=1.5,color="#87977d"))
        ax.text(x+.19,4.05,label,ha="left",va="center",fontsize=11,color=MUTED)
    footer(ax,.45,"GEMM(P A, B) = P GEMM(A, B)  [bitwise]")
    fig.subplots_adjust(left=.025,right=.99,top=.99,bottom=.02)
    save(fig,out,"batch-invariance-permutation")


def subbatch_figure(out):
    fig,ax=plt.subplots(figsize=(13,9.3))
    ax.set(xlim=(0,12.8),ylim=(0,10.5));ax.axis("off")
    ax.text(.55,10.2,"Sub-batch invariance",fontsize=19,weight="bold",color=INK)
    ax.text(.55,9.75,"Split token rows into smaller GEMMs. Concatenate identical output bits.",fontsize=13,color=MUTED)
    lane(ax,8.0,range(4),"A · full batch (M=4)","C · full output bits")
    lane(ax,4.8,PARTITIONS[0],"A0 = A[0:2] (M=2)","C0 = C[0:2] · bits")
    lane(ax,2.05,PARTITIONS[1],"A1 = A[2:4] (M=2)","C1 = C[2:4] · bits")
    ax.text(6.42,6.35,"Only M changes; B, N, and K stay fixed.",ha="center",
            fontsize=12,color=MUTED)
    footer(ax,.4,"GEMM(A, B) = concat(C0, C1)  [bitwise]")
    fig.subplots_adjust(left=.025,right=.99,top=.99,bottom=.02)
    save(fig,out,"batch-invariance-subbatch")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    full=product(range(4))
    assert product(PERMUTATION)==[full[i] for i in PERMUTATION]
    assert [row for part in PARTITIONS for row in product(part)]==full
    permutation_figure(args.out)
    subbatch_figure(args.out)
    print(f"Wrote permutation/sub-batch diagrams as SVG and PNG to {args.out}")


if __name__=="__main__":
    main()
