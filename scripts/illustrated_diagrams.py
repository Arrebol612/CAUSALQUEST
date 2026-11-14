"""Illustrated white-background diagrams, with locally vendored MIT SVG icons."""
import json
import re
import xml.etree.ElementTree as ET

from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from matplotlib import transforms
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

BLUE = '#0757e8'
GREEN = '#26712c'
PURPLE = '#4a2095'
RED = '#e53935'
BLACK = '#181818'
GRAY = '#636571'


class SvgPen(BasePen):
    def __init__(self):
        super().__init__(None)
        self.points, self.codes, self.start = [], [], None

    def _moveTo(self, p):
        self.start = p
        self.points.append(p); self.codes.append(Path.MOVETO)

    def _lineTo(self, p):
        self.points.append(p); self.codes.append(Path.LINETO)

    def _curveToOne(self, p1, p2, p3):
        self.points.extend((p1, p2, p3)); self.codes.extend((Path.CURVE4,)*3)

    def _qCurveToOne(self, p1, p2):
        self.points.extend((p1, p2)); self.codes.extend((Path.CURVE3,)*2)

    def _closePath(self):
        self.points.append(self.start); self.codes.append(Path.CLOSEPOLY)

    def _endPath(self):
        pass


def icon(f, root, name, x, y, size, recolor=None):
    """Flatten filled paths from the selected Fluent flat icons into vector patches.

    Source SVGs and their MIT notice are retained. Only fills and geometry are used;
    the teacher icon's optional blur effects are omitted in this small schematic.
    """
    tree = ET.parse(root/'docs/figures/icons'/name).getroot()
    vb = [float(v) for v in tree.get('viewBox', '0 0 32 32').split()]
    transform = transforms.Affine2D().translate(-vb[0], -vb[1]).scale(size/vb[2]).translate(x, y)
    palette = recolor or {}

    def walk(node, inherited=None):
        tag = node.tag.rsplit('}', 1)[-1]
        if tag in {'defs', 'clipPath', 'filter'}:
            return
        fill = node.get('fill', inherited or '#000000')
        if tag == 'path' and fill != 'none':
            pen = SvgPen(); parse_path(node.get('d', ''), pen)
            if pen.points:
                f.ax.add_patch(PathPatch(Path(pen.points, pen.codes),
                    facecolor=palette.get(fill, fill), edgecolor='none',
                    alpha=float(node.get('opacity', 1)), transform=transform+f.ax.transData, zorder=5))
        elif tag == 'rect' and fill != 'none':
            rect = Rectangle((float(node.get('x', 0)), float(node.get('y', 0))),
                float(node.get('width', 0)), float(node.get('height', 0)),
                facecolor=palette.get(fill, fill), edgecolor='none',
                transform=transform+f.ax.transData, zorder=5)
            f.ax.add_patch(rect)
        for child in node:
            walk(child, fill)
    walk(tree)


def card(f, x, y, w, h, title, color, size=23):
    f.rect(x, y, w, h, fill='white', edge=color, radius=9, lw=1.7)
    f.text(x+w/2, y+29, title, size=size, color=color, weight='bold', align='center')


def line(f, points, color=BLACK, width=2.1, dashed=False, head=True, z=4):
    patch = f.arrow(points, color=color, width=width, dashed=dashed, head=head)
    patch.set_zorder(z)


def dot(f, x, y, r, color, filled=True):
    f.ax.add_patch(Circle((x, y), r, facecolor=color if filled else 'white',
                         edgecolor=color, linewidth=1.7, zorder=5))


def speech(f, x, y, size, color=PURPLE):
    f.rect(x, y, size*.82, size*.68, fill='white', edge=color, radius=8, lw=3)
    f.ax.add_patch(Polygon([(x+size*.20, y+size*.67), (x+size*.20, y+size*.88),
                           (x+size*.40, y+size*.67)], facecolor='white', edgecolor=color,
                          linewidth=3, joinstyle='round', zorder=4))
    f.text(x+size*.41, y+size*.34, '?', size=size*.56, color=color, weight='bold', align='center')


def shield(f, x, y, size, color=PURPLE):
    vertices=[(x+size*.5,y),(x+size*.92,y+size*.16),(x+size*.87,y+size*.62),
              (x+size*.5,y+size*.98),(x+size*.13,y+size*.62),(x+size*.08,y+size*.16)]
    f.ax.add_patch(Polygon(vertices, facecolor=color, edgecolor=color, linewidth=1.5, zorder=5))
    line(f, [(x+size*.30,y+size*.48),(x+size*.46,y+size*.64),(x+size*.72,y+size*.31)],
         color='white', width=3, head=False, z=6)


def magnifier(f, x, y, size, color=PURPLE):
    f.ax.add_patch(Circle((x+size*.42,y+size*.38),size*.29,
                         facecolor='#fbfaff',edgecolor=color,linewidth=3,zorder=5))
    line(f,[(x+size*.64,y+size*.61),(x+size*.91,y+size*.89)],color=color,width=6,head=False)
    f.text(x+size*.42,y+size*.38,'?',size=size*.30,color=color,align='center')


def monitor(f, x, y, size, color=GREEN, pulse=False):
    f.rect(x,y,size,size*.65,fill='#f7fbf7',edge=color,radius=3,lw=2.3)
    line(f,[(x+size*.50,y+size*.65),(x+size*.50,y+size*.88)],color=color,width=3,head=False)
    line(f,[(x+size*.26,y+size*.88),(x+size*.74,y+size*.88)],color=color,width=3,head=False)
    if pulse:
        points=[(x+size*.10,y+size*.35),(x+size*.28,y+size*.35),(x+size*.39,y+size*.15),
                (x+size*.51,y+size*.52),(x+size*.64,y+size*.30),(x+size*.88,y+size*.30)]
    else:
        points=[(x+size*.13,y+size*.49),(x+size*.35,y+size*.40),
                (x+size*.57,y+size*.43),(x+size*.84,y+size*.18)]
    line(f,points,color=color,width=2.6,head=False)
    if not pulse:
        for px,py in points:dot(f,px,py,3,color)


def database(f, x, y, size, color=GREEN):
    w,h=size,size*.90
    f.ax.add_patch(Rectangle((x,y+h*.15),w,h*.65,facecolor='#f5faf5',edgecolor=color,lw=2,zorder=5))
    for yy in [y+h*.15,y+h*.47,y+h*.80]:
        f.ax.add_patch(Ellipse((x+w/2,yy),w,h*.30,facecolor='#f5faf5',edgecolor=color,lw=2,zorder=5))


def paper(f, x, y, size, color=BLUE):
    f.rect(x,y,size*.75,size,fill='white',edge=color,radius=3,lw=2)
    for j in range(4):
        yy=y+size*(.25+j*.16)
        f.rect(x+size*.12,yy-4,7,7,fill='white',edge=color,radius=0,lw=1.3)
        line(f,[(x+size*.29,yy),(x+size*.62,yy)],color=color,width=1.8,head=False)


def star_badge(f, x, y, size, color=RED):
    from math import cos,sin,pi
    f.ax.add_patch(Circle((x+size/2,y+size/2),size*.44,facecolor='white',edgecolor=color,lw=3,zorder=5))
    pts=[]
    for i in range(10):
        angle=-pi/2+i*pi/5;rad=size*(.30 if i%2==0 else .13)
        pts.append((x+size/2+rad*cos(angle),y+size/2+rad*sin(angle)))
    f.ax.add_patch(Polygon(pts,facecolor=color,edgecolor=color,zorder=6))


def trajectory(f,x,y,w,h,title,color=GREEN,n=4):
    f.rect(x,y,w,h,fill='white',edge=color,radius=5,lw=1.4)
    f.rect(x+8,y+9,34,29,fill=color,edge=color,radius=4,lw=1)
    f.text(x+25,y+24,title,size=18,color='white',weight='bold',align='center')
    xs=[x+15+(w-30)*j/(n-1) for j in range(n)]
    ys=[y+h-21-(h-65)*(j/(n-1)) for j in range(n)]
    line(f,list(zip(xs,ys)),color=color,width=2,head=False)
    for j,(xx,yy) in enumerate(zip(xs,ys)):dot(f,xx,yy,4.4,color,j==0)


def bracket(f,x1,x2,y,color=PURPLE,label=''):
    line(f,[(x1,y-12),(x1+4,y),(x2-4,y),(x2,y-12)],color=color,width=1.5,dashed=True,head=False)
    if label:f.text((x1+x2)/2,y+30,label,size=19,color=color,align='center')


def architecture(out, canvas, root):
    f=canvas(1670,730)
    f.text(24,25,'CAUSALQUEST · 自主调查与策略优化',size=25,color=BLACK,weight='bold')
    # Same interaction semantics as the public runner. No teacher-policy branch.
    line(f,[(129,415),(180,415)],color=BLACK,width=2.4)
    line(f,[(310,240),(310,315)],color=BLUE,width=2.4)
    f.text(343,277,'初始化',size=18,color=BLUE)
    line(f,[(440,385),(485,385)],color=GREEN,width=2.7)
    f.text(463,365,'$a_t$',size=21,color=GREEN,align='center')
    line(f,[(625,515),(625,570),(310,570),(310,515)],color=GREEN,width=2.2)
    f.text(468,570,r'工具观测 $o_{t+1}$',size=19,color=GREEN,align='center',background=True)
    line(f,[(625,315),(625,240)],color=GREEN,width=2.3)
    f.text(648,278,'完整交互记录',size=17,color=GREEN)
    line(f,[(765,150),(815,150)],color=PURPLE,width=2.8)
    line(f,[(1085,285),(1130,285)],color=PURPLE,width=2.8)
    line(f,[(1390,175),(1435,175)],color=PURPLE,width=2.8)
    line(f,[(1535,290),(1535,335)],color=RED,width=2.7)
    line(f,[(1535,515),(1535,638),(155,638),(155,490),(180,490)],color=GREEN,width=2.2,dashed=True)
    f.text(930,638,r'下一轮使用新策略 $\pi_{\theta^{\prime}}$ 重新采样',
           size=21,color=GREEN,align='center',background=True)

    card(f,24,340,105,155,'',PURPLE)
    speech(f,48,362,68)
    f.text(76,450,'任务输入',size=20,color=BLACK,weight='bold',align='center')
    f.text(76,477,'案例 x',size=18,color=BLACK,align='center')
    card(f,180,60,260,180,'专家示范与 SFT',BLUE)
    icon(f,root,'teacher_flat_default.svg',202,105,100)
    paper(f,327,111,75,BLUE)
    f.text(310,220,'仅学习 assistant 目标',size=18,color=BLUE,align='center')
    card(f,180,315,260,200,r'Agent 策略 $\pi_\theta$',GREEN)
    icon(f,root,'technologist_flat_default.svg',260,355,105)
    f.text(310,479,'选工具 · 核对证据',size=18,color=BLACK,align='center')
    f.text(310,502,'更新状态 · 提交报告',size=18,color=BLACK,align='center')
    card(f,485,315,280,200,'环境与工具',GREEN)
    monitor(f,505,371,92,GREEN)
    database(f,570,407,49,GREEN)
    f.text(684,387,'查证据',size=18,color=BLACK,align='center')
    f.text(684,420,'算统计',size=18,color=BLACK,align='center')
    f.text(684,453,'数值辅助',size=18,color=BLACK,align='center')
    f.text(625,494,'环境执行；Agent 决定行动',size=17,color=GREEN,align='center')
    card(f,485,60,280,180,'完整轨迹组',GREEN)
    trajectory(f,500,110,73,100,'1',GREEN,3)
    trajectory(f,581,110,73,100,'2',GREEN,4)
    trajectory(f,680,110,73,100,'G',GREEN,5)
    f.text(667,169,'…',size=22,color=GREEN,align='center')
    f.text(625,225,r'同一案例：$\tau_1,\tau_2,\ldots,\tau_G$',size=17,color=GREEN,align='center')

    card(f,815,60,270,455,'公开证据评审',PURPLE)
    palette={'#9B9B9B':PURPLE,'#CDC4D6':'#b6a5d4','#E19747':'#7958b2','#F3EEF8':'#faf7ff'}
    icon(f,root,'clipboard_flat.svg',839,112,91,palette)
    shield(f,949,115,75)
    magnifier(f,995,168,52)
    f.text(950,256,'人工评审 · 仅用公开观测',size=17,color=BLACK,align='center')
    labels=['有效性','效应估计','异常诊断','证据覆盖','决策一致性','不确定性','格式']
    for j,label in enumerate(labels):
        yy=291+j*26
        dot(f,849,yy,10,PURPLE)
        f.text(849,yy,str(j+1),size=13,color='white',weight='bold',align='center')
        f.text(874,yy,label,size=18,color=BLACK)
        line(f,[(999,yy),(1058,yy)],color=PURPLE,width=1.2,head=False)
    f.text(950,490,'适用项 0–2 分；其余为 null',size=17,color=PURPLE,align='center')
    card(f,1130,60,260,455,'组内排序与优势',PURPLE)
    scale_palette={'#9B9B9B':PURPLE,'#D3D3D3':'#b6a5d4','#F9C23C':'#7958b2'}
    icon(f,root,'balance_scale_flat.svg',1202,117,115,scale_palette)
    f.text(1260,263,'等级优先',size=19,color=BLACK,align='center')
    for j,num in enumerate(['3','2','1','0']):
        xx=1150+j*59
        f.rect(xx,287,43,42,fill=PURPLE if j<2 else 'white',edge=PURPLE,radius=4,lw=1.5)
        f.text(xx+21.5,308,num,size=22,color='white' if j<2 else PURPLE,weight='bold',align='center')
    f.text(1260,362,'同级：语义 Pareto',size=19,color=BLACK,align='center')
    f.text(1260,397,'核心支持门控',size=19,color=GREEN,weight='bold',align='center')
    f.text(1260,442,r'$A_i=\frac{\mathrm{wins}-\mathrm{losses}}{G-1}$',size=24,color=PURPLE,align='center')
    f.text(1260,489,r'无核心支持组：$A_i=0$',size=17,color=BLACK,align='center')
    card(f,1435,60,200,230,'裁剪策略更新',RED,size=22)
    star_badge(f,1492,115,84)
    f.text(1535,225,'当前 / 旧策略比率',size=18,color=BLACK,align='center')
    f.text(1535,260,'固定参考约束',size=18,color=BLACK,align='center')
    card(f,1435,335,200,180,r'新策略 $\pi_{\theta^{\prime}}$',GREEN,size=22)
    icon(f,root,'technologist_flat_default.svg',1490,380,89)
    f.text(1535,494,'保存与独立重载',size=18,color=GREEN,align='center')
    bracket(f,815,1390,550,PURPLE,'公开证据 → 组内偏好 → 策略信号')
    f.text(835,707,'模型自主选择证据与行动；程序执行查询和计算。完整轨迹与模型失败均保留用于评测。',
           size=19,color=BLACK,align='center')
    f.save('framework_architecture',out)


LABELS={'validity':'有效性','issue_coverage':'问题覆盖','power':'功效',
        'localization':'异常定位','trajectory':'调查过程','estimand':'估计对象',
        'statistics':'统计判断','practicality':'业务意义','evidence':'证据',
        'decision':'决策','human_boundary':'人工审批'}


def rubric(out,canvas,root):
    f=canvas(1670,780)
    f.text(24,27,'Rubric Tree · 任务标准与公开证据评分',size=25,color=BLACK,weight='bold')
    # The current RL interface has its own input; task-tree weights are not rewards.
    line(f,[(168,318),(195,318),(195,80),(1075,80)],color=PURPLE,width=2.2,head=False)
    for cx in [385,730,1075]:line(f,[(cx,80),(cx,115)],color=PURPLE,width=2.3)
    f.text(730,59,'按任务类型选择评价结构',size=20,color=PURPLE,align='center',background=True)
    card(f,28,230,140,180,'任务标准',PURPLE,size=22)
    speech(f,59,283,85)
    f.text(98,387,'Rubric Tree',size=17,color=PURPLE,align='center')

    configs=[('design_review.json','实验设计评审',BLUE),
             ('health_diagnosis.json','实验健康诊断',GREEN),
             ('post_experiment.json','实验后决策',PURPLE)]
    for i,(filename,title,color) in enumerate(configs):
        x=230+345*i
        payload=json.loads((root/'examples/rubrics'/filename).read_text(encoding='utf-8'))
        card(f,x,115,310,475,title,color,size=23)
        if i==0:
            paper(f,x+105,170,84,color)
            line(f,[(x+191,228),(x+227,177)],color=color,width=6,head=False)
        elif i==1:
            monitor(f,x+111,174,91,color,pulse=True)
        else:
            icon(f,root,'balance_scale_flat.svg',x+111,166,91,
                 {'#9B9B9B':color,'#D3D3D3':'#b6a5d4','#F9C23C':'#7958b2'})
        f.text(x+155,274,'任务根节点 · 权重 1',size=17,color=color,align='center')
        children=payload['root']['children']
        for j,node in enumerate(children):
            yy=309+j*29
            gate=node['hard_gate']
            if gate:
                shield(f,x+20,yy-9,18,RED)
            else:
                f.rect(x+23,yy-6,12,12,fill='white',edge=color,radius=0,lw=1.2)
            label=LABELS[node['id']]+(' †' if gate else '')
            f.text(x+51,yy,label,size=19,color=RED if gate else BLACK)
            f.text(x+280,yy,f"{node['weight']:.2f}",size=19,color=color,align='right')
        f.text(x+155,566,payload['version'],size=12,color=color,align='center')
    bracket(f,230,1230,620,PURPLE,'三类任务的结构化评价标准')
    f.text(730,700,'† 硬门禁标记；每棵树的同级权重之和为 1。',size=19,color=BLACK,align='center')

    f.rect(1280,45,350,43,fill='white',edge=PURPLE,radius=6,lw=1.5)
    f.text(1455,66,'公开轨迹与实际观测',size=20,color=PURPLE,align='center')
    line(f,[(1455,88),(1455,115)],color=PURPLE,width=2.3)
    card(f,1280,115,350,475,'当前 RL 评审接口',PURPLE,size=23)
    icon(f,root,'clipboard_flat.svg',1330,170,85,
         {'#9B9B9B':PURPLE,'#CDC4D6':'#b6a5d4','#E19747':'#7958b2','#F3EEF8':'#faf7ff'})
    shield(f,1460,176,76,PURPLE)
    magnifier(f,1525,205,52,PURPLE)
    f.text(1455,290,'人工七维公开证据评分',size=21,color=PURPLE,weight='bold',align='center')
    f.text(1455,329,'有效性 · 效应估计 · 异常诊断\n证据 · 决策 · 不确定性 · 格式',
           size=18,color=BLACK,align='center')
    f.text(1455,374,'适用维度 0–2；其余 null',size=17,color=BLACK,align='center')
    for j,num in enumerate(['0','1','2','3']):
        xx=1320+j*70
        f.rect(xx,406,53,43,fill=GREEN if j>=2 else 'white',edge=GREEN,radius=4,lw=1.5)
        f.text(xx+26.5,427,num,size=22,color='white' if j>=2 else GREEN,weight='bold',align='center')
    f.text(1455,478,'等级 → 组内比较 → 优势',size=21,color=PURPLE,weight='bold',align='center')
    f.text(1455,520,r'无核心支持组：$A_i=0$',size=18,color=BLACK,align='center')
    f.text(1455,563,'保留完整轨迹与失败分母',size=18,color=GREEN,align='center')
    line(f,[(1455,590),(1455,624)],color=RED,width=2.5)
    card(f,1280,624,350,78,'组内优势用于策略更新',GREEN,size=21)
    f.text(1455,682,'裁剪目标 + 固定参考',size=17,color=GREEN,align='center')
    f.text(835,753,'任务树描述评价结构；当前七维评审单独定义排序，树权重不直接相加为 RL 奖励。',
           size=19,color=BLACK,align='center')
    f.save('rubric_tree_and_reward',out)
