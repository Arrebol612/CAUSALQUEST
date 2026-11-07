"""Render the README's original, implementation-aligned academic figures.

Install the optional ``figures`` extra. A CJK font is needed only when regenerating;
SVG exports contain glyph outlines and need no reader-side font installation.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INK = '#243347'
MUTED = '#5d6a79'
LINE = '#607083'
BLUE = '#3266a0'
GREEN = '#337b68'
PURPLE = '#79599c'
AMBER = '#a4772d'
RED = '#a75b45'
PALE = {'blue': '#eaf1fa', 'green': '#eaf4ef', 'purple': '#f1edf7',
        'amber': '#faf2e4', 'gray': '#f4f6f8', 'gate': '#fbf0e9'}
STROKE = {'blue': BLUE, 'green': GREEN, 'purple': PURPLE, 'amber': AMBER,
          'gray': '#a2aeba', 'gate': RED}


def choose_font(path=None):
    if path:
        font_manager.fontManager.addfont(str(Path(path).resolve(strict=True)))
        family = font_manager.FontProperties(fname=path).get_name()
    else:
        available = {item.name for item in font_manager.fontManager.ttflist}
        names = ('Noto Sans CJK SC', 'Microsoft YaHei', 'Source Han Sans SC',
                 'WenQuanYi Micro Hei', 'SimHei')
        family = next((name for name in names if name in available), None)
        if family is None:
            raise ValueError('Install a CJK font or pass --font /path/to/font.ttf')
    plt.rcParams.update({'font.family': family, 'font.size': 14,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'causalquest-figures-v1',
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})


class Figure:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width / 100, height / 100), facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
        self.ax.axis('off')
        self.text_bounds = []

    def text(self, x, y, value, size=14, color=INK, weight='normal', align='left',
             valign='center', background=False):
        item = self.ax.text(x, y, value, fontsize=size * .72, color=color,
                            fontweight=weight, ha=align, va=valign, linespacing=1.6,
                            zorder=6, bbox={'facecolor': 'white', 'edgecolor': 'none',
                                           'pad': 2} if background else None)
        self.text_bounds.append(item)
        return item

    def rect(self, x, y, w, h, fill='white', edge='#cbd3dc', radius=5, lw=1.15, z=3):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle=f'round,pad=0,rounding_size={radius}',
                              facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
        self.ax.add_patch(patch)
        return patch

    def panel(self, x, y, w, h, title, theme):
        self.rect(x, y, w, h, fill='#fafbfc', edge='#d4dbe3', radius=8, lw=1.05, z=0)
        self.ax.add_patch(Rectangle((x + 20, y + 24), 4, 27,
                                   color=STROKE[theme], linewidth=0, zorder=1))
        self.text(x + 38, y + 38, title, size=23, weight='bold')

    def node(self, x, y, w, h, title, body='', theme='gray', title_size=22, body_size=18):
        self.rect(x, y, w, h, fill=PALE[theme], edge=STROKE[theme])
        if body:
            self.text(x + w/2, y + 30, title, size=title_size, weight='bold', align='center')
            self.text(x + w/2, y + h*.70, body, size=body_size, color=MUTED, align='center')
        else:
            self.text(x + w/2, y + h/2, title, size=title_size, weight='bold', align='center')

    def arrow(self, points, color=LINE, dashed=False, width=1.5, head=True):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
        patch = FancyArrowPatch(path=path, arrowstyle='-|>' if head else '-',
                                mutation_scale=12, linewidth=width,
                                linestyle=(0, (5, 4)) if dashed else '-',
                                color=color, joinstyle='round', capstyle='round', zorder=2)
        self.ax.add_patch(patch)

    def save(self, stem, out):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        bounds = self.fig.bbox
        outside = []
        for item in self.text_bounds:
            box = item.get_window_extent(renderer)
            if box.x0 < bounds.x0 or box.y0 < bounds.y0 or box.x1 > bounds.x1 or box.y1 > bounds.y1:
                outside.append(item.get_text())
        if outside:
            raise ValueError('Text outside canvas: ' + repr(outside))
        metadata = {'Title': stem.replace('_', ' '), 'Creator': 'CAUSALQUEST figure source',
                    'Date': None, 'Description': 'Original schematic of the implemented framework.'}
        self.fig.savefig(out / (stem + '.svg'), format='svg', metadata=metadata,
                         facecolor='white')
        self.fig.savefig(out / (stem + '.png'), format='png', dpi=180,
                         facecolor='white', metadata={'Software': 'CAUSALQUEST figure source'})
        # An indexed preview keeps README transfers small; SVG remains full vector.
        with Image.open(out / (stem + '.png')) as source:
            preview = source.convert('RGB').quantize(colors=256, dither=Image.Dither.NONE)
            preview.save(out / (stem + '.png'), optimize=True, dpi=(180, 180))
        plt.close(self.fig)


def architecture(out):
    f = Figure(1600, 970)
    f.text(36, 43, 'CAUSALQUEST：证据驱动的 Agent 训练闭环', size=31, weight='bold')
    f.text(36, 90, '同一任务环境连接专家示范、自主交互、公开证据评审与分组策略更新', size=20, color=MUTED)
    f.panel(36, 142, 340, 674, '(a) 数据与任务', 'green')
    f.panel(408, 142, 600, 674, '(b) 自主执行', 'blue')
    f.panel(1040, 142, 524, 674, '(c) 评审与优化', 'purple')

    # Connectors are routed in fixed lanes, outside node interiors.
    f.arrow([(206, 354), (206, 420)], color=GREEN)
    f.arrow([(350, 467), (365, 467), (365, 228), (855, 228), (855, 285)], color=GREEN)
    f.text(660, 213, '任务材料与工具合同', size=17, color=GREEN, background=True)
    f.arrow([(206, 663), (206, 710)], color=BLUE)
    f.arrow([(350, 748), (386, 748), (386, 330), (445, 330)], color=BLUE)
    f.text(260, 685, '仅学习 assistant 目标', size=16, color=MUTED, align='center')
    f.arrow([(660, 365), (745, 365)])
    f.text(702, 345, '$a_t$', size=22, align='center', background=True)
    f.arrow([(855, 445), (855, 500)], color=GREEN)
    f.text(892, 472, '$o_{t+1}$', size=22, color=GREEN, background=True)
    f.arrow([(550, 500), (550, 445)], color=BLUE)
    f.text(584, 472, '$h_t$', size=22, color=BLUE, background=True)
    f.arrow([(705, 588), (705, 650)])
    f.text(726, 618, '按任务分组', size=16, color=MUTED, background=True)
    f.arrow([(963, 712), (1024, 712), (1024, 310), (1070, 310)])
    f.arrow([(1302, 378), (1302, 445)])
    f.arrow([(1302, 553), (1302, 620)])
    f.arrow([(1302, 746), (1302, 850), (398, 850), (398, 405), (445, 405)],
            color=BLUE, dashed=True, width=1.8)
    f.text(860, 850, r'保存新策略 $\pi_{\theta^{\prime}}$；下一轮重新采样',
           size=20, color=BLUE, align='center', background=True)

    f.node(64, 250, 286, 104, 'Case Factory',
           '设计评审 / 健康诊断\n实验后决策', theme='green', body_size=18)
    f.node(64, 420, 286, 95, '公开任务材料',
           '文档 · 只读数据库 · 合同', theme='green', body_size=17)
    f.node(64, 585, 286, 78, '审核后的专家示范', theme='gray', title_size=21)
    f.node(64, 710, 286, 75, 'SFT 冷启动', 'assistant-target CE', theme='blue', body_size=17)
    f.node(445, 285, 215, 160, r'Agent 策略 $\pi_{\theta}$',
           '选择工具与操作数\n更新状态与提交报告', theme='blue', title_size=22, body_size=18)
    f.node(745, 285, 220, 160, r'任务环境 $\mathcal{E}$',
           '查询、统计与工具执行\n五个数值辅助工具', theme='green', title_size=22, body_size=17)
    f.node(448, 500, 515, 88, '公开交互历史 $h_t$',
           '请求 · 实际观测 · 调查状态 · 报告', theme='gray', body_size=18)
    f.node(448, 650, 515, 126, r'完整轨迹组 $\{\tau_1,\ldots,\tau_G\}$',
           '从正常初始请求出发\n原生 token · 采样配置 · 行为 / 旧策略概率', theme='blue', body_size=18)
    f.node(1070, 250, 464, 128, '公开证据评审',
           '人工七维评分 · 核心结论 · 等级 0–3\n任务 Rubric Tree 描述评价结构', theme='purple', body_size=18)
    f.node(1070, 445, 464, 108, '组内比较与优势 $A_i$',
           '等级优先 · 语义 Pareto · 核心支持门控', theme='purple', body_size=18)
    f.node(1070, 620, 464, 126, r'裁剪策略目标 + 固定参考 $\pi_{\mathrm{ref}}$',
           '完整轨迹内归一化\n优化器更新 · 保存 · 独立重载', theme='amber', title_size=21, body_size=19)

    f.arrow([(44, 895), (83, 895)])
    f.text(95, 895, '数据 / 执行流', size=17, color=MUTED)
    f.arrow([(280, 895), (320, 895)], color=BLUE, dashed=True)
    f.text(332, 895, '策略反馈', size=17, color=MUTED)
    for x, theme, label in [(580, 'green', '任务与环境'), (825, 'blue', '模型与轨迹'),
                            (1070, 'purple', '证据评审'), (1315, 'amber', '参数优化')]:
        f.rect(x, 884, 18, 18, fill=PALE[theme], edge=STROKE[theme], radius=2)
        f.text(x + 30, 895, label, size=17, color=MUTED)
    f.text(800, 942, '贯穿全程：来源与评审绑定 · 原生概率记录 · 检查点核验 · 固定分母质量统计',
           size=19, color=MUTED, align='center')
    f.save('framework_architecture', out)


LABELS = {'validity': '有效性', 'issue_coverage': '问题覆盖', 'power': '功效',
          'localization': '异常定位', 'trajectory': '调查过程', 'estimand': '估计对象',
          'statistics': '统计判断', 'practicality': '业务意义', 'evidence': '证据',
          'decision': '决策', 'human_boundary': '人工审批'}


def rubric(out):
    f = Figure(1600, 1040)
    f.text(36, 43, 'Rubric Tree：任务标准与学习信号', size=31, weight='bold')
    f.text(36, 90, '三棵任务树展示实际标准与局部权重；当前 RL 通过独立的七维评审接口产生排序',
           size=20, color=MUTED)
    trees = [('design_review.json', '(a) 设计评审', 'green'),
             ('health_diagnosis.json', '(b) 健康诊断', 'blue'),
             ('post_experiment.json', '(c) 实验后决策', 'purple')]
    for i, (name, title, theme) in enumerate(trees):
        x = 36 + 336*i
        payload = json.loads((ROOT / 'examples/rubrics' / name).read_text(encoding='utf-8'))
        root = payload['root']
        f.panel(x, 142, 316, 734, title, theme)
        f.node(x + 18, 225, 280, 90, '任务根节点 · 权重 1',
               payload['version'], theme=theme, title_size=20, body_size=12.5)
        children = root['children']
        trunk_x, first, gap = x + 31, 372, 63
        f.arrow([(x+158, 315), (x+158, 338), (trunk_x, 338),
                 (trunk_x, first+(len(children)-1)*gap)], head=False, width=1.15)
        for j, node in enumerate(children):
            cy = first + j*gap
            f.arrow([(trunk_x, cy), (x+58, cy)], width=1.15)
            gate = node['hard_gate']
            f.rect(x+58, cy-23, 240, 46, fill=PALE['gate'] if gate else 'white',
                   edge=RED if gate else '#b8c5d1', radius=4, lw=1)
            label = LABELS[node['id']] + (' †' if gate else '')
            f.text(x+72, cy, label, size=19, color=RED if gate else INK)
            f.text(x+282, cy, f"{node['weight']:.2f}", size=19, align='right')

    f.panel(1072, 142, 492, 734, '(d) 当前 RL 评审路径', 'amber')
    f.node(1096, 225, 444, 84, '实际轨迹与公开证据',
           '以被评动作之前的观测为依据', theme='gray', body_size=18)
    f.arrow([(1318, 309), (1318, 350)])
    f.node(1096, 350, 444, 164, '七维公开证据评审',
           '有效性 · 效应估计 · 异常诊断\n证据 · 决策 · 不确定性 · 格式\n适用项 0–2 分；其余为 null',
           theme='purple', body_size=19)
    f.arrow([(1318, 514), (1318, 560)])
    f.node(1096, 560, 444, 104, '等级 0–3 与核心支持',
           '硬违规 / 未提交 · 不支持 · 部分 · 完整', theme='purple', body_size=18)
    f.arrow([(1318, 664), (1318, 710)])
    f.node(1096, 710, 444, 126, '组内比较 → 优势 $A_i$',
           '同级：语义 Pareto，格式仅用于语义打平\n无核心支持的组：零直接策略优势',
           theme='amber', body_size=17.5)

    f.rect(36, 907, 988, 87, fill=PALE['gray'], edge='#cbd3dc', radius=5)
    f.text(60, 932, '† 硬门禁标记；每棵树的同级权重之和为 1。', size=20, weight='bold')
    f.text(60, 969, '树权重不直接求和为 RL 奖励；历史 verifier 保留独立聚合规则。', size=18, color=MUTED)
    f.rect(1072, 907, 492, 87, fill=PALE['gray'], edge='#cbd3dc', radius=5)
    f.text(1318, 932, '评价结构与学习信号分别定义', size=20, weight='bold', align='center')
    f.text(1318, 969, '所有模型失败保留在原始评测分母中', size=18, color=MUTED, align='center')
    f.save('rubric_tree_and_reward', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/figures')
    parser.add_argument('--font', help='Optional local CJK font path; no font is bundled.')
    args = parser.parse_args()
    choose_font(args.font)
    args.output.mkdir(parents=True, exist_ok=True)
    architecture(args.output)
    rubric(args.output)
    print('Rendered two figures as outlined SVG and high-resolution PNG.')
