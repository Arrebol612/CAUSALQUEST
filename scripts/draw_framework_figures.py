"""Render the README's illustrated, implementation-aligned diagrams.

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
from PIL.PngImagePlugin import PngInfo

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
                                mutation_scale=17, linewidth=width,
                                linestyle=(0, (5, 4)) if dashed else '-',
                                color=color, joinstyle='round', capstyle='round', zorder=2)
        self.ax.add_patch(patch)

        return patch

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
        license_text = (ROOT/'docs/figures/icons/LICENSE.txt').read_text(encoding='utf-8')
        credit = 'Illustrations: Microsoft Fluent Emoji, copyright Microsoft Corporation, MIT license.'
        metadata = {'Title': stem.replace('_', ' '), 'Creator': 'CAUSALQUEST figure source',
                    'Date': None, 'Description': credit+'\n'+license_text}
        self.fig.savefig(out / (stem + '.svg'), format='svg', metadata=metadata,
                         facecolor='white')
        self.fig.savefig(out / (stem + '.png'), format='png', dpi=180,
                         facecolor='white', metadata={'Software': 'CAUSALQUEST figure source'})
        # An indexed preview keeps README transfers small; SVG remains full vector.
        with Image.open(out / (stem + '.png')) as source:
            preview = source.convert('RGB').quantize(colors=256, dither=Image.Dither.NONE)
            info = PngInfo()
            info.add_text('Copyright', credit)
            info.add_text('License', license_text)
            info.add_text('Source', 'https://github.com/microsoft/fluentui-emoji')
            preview.save(out / (stem + '.png'), optimize=True, dpi=(180, 180), pnginfo=info)
        plt.close(self.fig)


from illustrated_diagrams import architecture, rubric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/figures')
    parser.add_argument('--font', help='Optional local CJK font path; no font is bundled.')
    args = parser.parse_args()
    choose_font(args.font)
    args.output.mkdir(parents=True, exist_ok=True)
    architecture(args.output, Figure, ROOT)
    rubric(args.output, Figure, ROOT)
    print('Rendered two figures as outlined SVG and high-resolution PNG.')
