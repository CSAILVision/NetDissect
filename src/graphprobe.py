import os
import re
from collections import OrderedDict, defaultdict
import expdir

category_colors = OrderedDict([
    ('scene',    '#3288bd'),
    ('object',   '#99d594'),
    ('part',     '#e6f598'),
    ('material', '#fee08b'),
    ('texture',  '#fc8d59'),
    ('color',    '#d53e4f'),
    ('total',    '#aaaaaa')
])


def loadviz(directory, blob):
    ed = expdir.ExperimentDirectory(directory)
    html_fn = ed.filename(['html', '%s.html' % expdir.fn_safe(blob)])
    with open(html_fn) as f:
        lines = f.readlines()
    result = OrderedDict()
    for line in lines:
        if 'unit ' in line:
            u, v = line.split(':', 1)
            unit = int(re.search(r'unit (\d+)', u).group(1)) - 1
            u_result = []
            for w in v.split(';'):
                m = re.search(r'(\w.*\w)? \((\w+), ([.\d]+)\)', w)
                if not m:
                    print 'On line', v
                    print 'Error with', w
                label = m.group(1)
                category = m.group(2)
                score = float(m.group(3))
                u_result.append((label, category, score))
            result[unit] = u_result
    return result

def summarize(scores, threshold, top_only=True):
    result = defaultdict(float)
    denom = len(scores)
    for k, v in scores.items():
        got = False
        for val in v:
            label, category, score = val
            if score >= threshold:
                result[category] += 1.0 / denom
                got = True
            if top_only:
                break
        if got:
            result['total'] += 1.0 / denom
    return result

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        parser = argparse.ArgumentParser(
            description='Generate visualization for probed activation data.')
        parser.add_argument(
                '--directories',
                nargs='*',
                help='directories to graph')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='blobs to graph')
        parser.add_argument(
                '--threshold',
                type=float, default=0.05,
                help='score above which to count items')
        parser.add_argument(
                '--top_only',
                type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                default=True,
                help='include only the top values')
        parser.add_argument(
                '--include_total',
                type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                default=False,
                help='include total value line')
        parser.add_argument(
                '--labels',
                nargs='*',
                help='tick labels')
        parser.add_argument(
                '--title',
                help='graph title')
        parser.add_argument(
                '--legend',
                default='upper right',
                help='location of legend')
        parser.add_argument(
                '--maxy',
                type=float, default=None,
                help='y axis range to apply')
        parser.add_argument(
                '--out',
                help='output filename for graph')
        args = parser.parse_args()
        data = []
        categories = set(category_colors.keys())
        for directory in args.directories:
            for blob in args.blobs:
                stats = summarize(loadviz(directory, blob), args.threshold,
                        top_only=args.top_only)
                data.append(stats)
                categories.update(stats.keys())
        x = range(1, len(data) + 1)
        maxval = 0
        plt.figure(num=None, figsize=(7,4), dpi=300)
        for cat in category_colors.keys():
            if cat not in categories:
                continue
            if not args.include_total and cat == 'total':
                continue
            dat = [d[cat] for d in data]
            maxval = max(maxval, max(dat))
            plt.plot(x, dat, 'o-' if cat != 'total' else 's--',
                    color=category_colors[cat], label=cat)
        if args.labels:
            plt.xticks(x, args.labels)
        plt.margins(0.1)
        plt.legend(loc=args.legend)
        if args.maxy is not None:
            plt.ylim(-0.01, args.maxy)
        else:
            plt.ylim(-maxval * 0.05, maxval * 1.5)
        ax = plt.gca()
        ax.yaxis.grid(True)
        for side in ['top', 'bottom', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        if args.title:
            plt.title(args.title)
        plt.ylabel('portion of units alinged to a category concept')
        plt.savefig(args.out)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
