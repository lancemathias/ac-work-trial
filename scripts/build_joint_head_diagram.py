"""Editable Excalidraw-style joint-head diagram, including lexical-prior inputs."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def build(root=ROOT):
    out = root / 'outputs/handoff/figures'
    font = Path('/System/Library/Fonts/Supplemental/Comic Sans MS.ttf')
    family = 'DejaVu Sans'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    fig, ax = plt.subplots(figsize=(15, 13.6))
    ax.set(xlim=(0, 1500), ylim=(1360, 0))
    ax.axis('off')
    fig.patch.set_facecolor('#fffefa')
    elements = []
    ink, blue, green, gray = '#243247', '#e9f2ff', '#e8f5ed', '#f5f5f3'

    def base(kind, x, y, w, h, fill='transparent'):
        return dict(id=f'joint-{len(elements)}', type=kind, x=x, y=y, width=w, height=h,
                    angle=0, strokeColor=ink, backgroundColor=fill, fillStyle='solid',
                    strokeWidth=1.6, strokeStyle='solid', roughness=1, opacity=100,
                    groupIds=[], frameId=None, roundness={'type': 3} if kind=='rectangle' else None,
                    seed=3000+len(elements), version=1, versionNonce=1, isDeleted=False,
                    boundElements=None, updated=0, link=None, locked=False)

    def label(x, y, w, h, value, size=19, bold=False):
        ax.text(x+w/2, y+h/2, value, ha='center', va='center', fontsize=size*.72,
                fontfamily=[family, 'DejaVu Sans'], weight='bold' if bold else 'normal',
                color=ink, linespacing=1.5, zorder=4)
        e=base('text', x+8, y+8, w-16, h-16)
        e.update(text=value, originalText=value, fontSize=size, fontFamily=1, textAlign='center',
                 verticalAlign='middle', containerId=None, autoResize=False, lineHeight=1.25)
        elements.append(e)

    def box(x, y, w, h, value, fill=green):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
                        fc=fill,ec=ink,lw=1.5,zorder=2)
        p.set_sketch_params(scale=.65,length=120,randomness=1.6)
        ax.add_patch(p)
        elements.append(base('rectangle',x,y,w,h,fill))
        label(x,y,w,h,value)

    def arrow(points):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color='#647b94',lw=1.5,zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                    mutation_scale=15,lw=1.5,color='#647b94',zorder=1))
        x,y=points[0]
        e=base('arrow',x,y,max(p[0] for p in points)-min(p[0] for p in points),
               max(p[1] for p in points)-min(p[1] for p in points))
        e.update(points=[[px-x,py-y] for px,py in points],startBinding=None,endBinding=None,
                 lastCommittedPoint=None,startArrowhead=None,endArrowhead='arrow',elbowed=False)
        elements.append(e)

    label(80,20,1340,65,'Joint schema head on frozen Qwen3.5-9B',29,True)
    main=[(110,70,'State + question + permitted options',gray),
          (220,80,'Frozen Qwen3.5-9B backbone · BF16',blue),
          (340,70,'Token hidden states · H = 4096',blue),
          (450,90,'Head LayerNorm + six projections\nProjection width = 1024',green),
          (580,80,'2 evidence-routing layers\n16 attention heads · FFN 4096',green),
          (700,80,'Question + option summaries + global state\n+ type embedding',green),
          (820,80,'4 joint field decoder layers\nSelf-attention + cross-attention',green),
          (940,80,'Cosine score + residual MLP\n4096 → 1024 → 1 · GELU',green),
          (1080,80,'Lexical prior + sigmoid(gate) × joint score',green),
          (1200,70,'Per-field softmax over allowed alternatives',gray)]
    for y,h,value,fill in main:box(80,y,700,h,value,fill)
    for a,b in zip(main,main[1:]):arrow([(430,a[0]+a[1]),(430,b[0])])
    label(950,90,450,45,'Lexical prior branch',23,True)
    box(950,250,450,75,'Option JSON token IDs\n(ID + description tokens)',gray)
    box(950,390,450,95,'Frozen Qwen output-embedding table\nLook up rows for those token IDs',blue)
    box(950,550,450,80,'Mean over option-token embeddings\n→ lexical option vector e',blue)
    box(950,690,450,95,'Question-span mean q + final-token g\nFrom head-normalized token states',green)
    box(950,850,450,100,'cosine(e, q + g) × learned positive scale\n→ lexical prior',green)
    arrow([(780,145),(1175,145),(1175,250)])
    arrow([(1175,325),(1175,390)])
    arrow([(1175,485),(1175,550)])
    # The lexical mean also feeds the option-lexical projection.
    arrow([(950,590),(870,590),(870,520),(780,520)])
    # Hidden-state anchor, computed after the shared head LayerNorm.
    arrow([(780,470),(850,470),(850,737),(950,737)])
    arrow([(1175,785),(1175,850)])
    arrow([(1400,590),(1450,590),(1450,900),(1400,900)])
    arrow([(1175,950),(1175,1120),(780,1120)])
    label(80,1302,1340,35,'Blue: frozen backbone inputs   •   Green: trainable head (121,762,820 parameters)   •   Gray: inputs / probabilities',16)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    for ext in ['png','svg']:
        fig.savefig(out/f'joint-head-architecture.{ext}',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    svg=out/'joint-head-architecture.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (out/'joint-head-architecture.excalidraw').write_text(json.dumps({
        'type':'excalidraw','version':2,'source':'ac-worktrial','elements':elements,
        'appState':{'viewBackgroundColor':'#fffefa','gridSize':None},'files':{}},indent=2)+'\n')
    print('Created joint-head PNG/SVG and editable Excalidraw with token→embedding→mean→cosine prior path.')


if __name__=='__main__':
    build()
