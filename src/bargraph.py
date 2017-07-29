import re
import expdir
import itertools
import operator
from xml.etree import ElementTree as et
 
default_category_order = [
    'object',
    'scene',
    'part',
    'material',
    'texture',
    'color'
]

palette = [
    ('#4B4CBF', '#B6B6F2'),
    ('#55B05B', '#B6F2BA'),
    ('#50BDAC', '#A5E5DB'),
    ('#D4CF24', '#F2F1B6'),
    ('#F0883B', '#F2CFB6'),
    ('#D92E2B', '#F2B6B6')
]

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def bar_graph_svg(ed, blob, barheight, barwidth,
        order=None,
        show_labels=True,
        threshold=0.04,
        rendered_order=None,
        save=None):
    records = ed.load_csv(blob=blob, part='result')
    # ['unit', 'category', 'label', 'score']
    # Examine each label
    label_cats = {}
    label_score = {}
    for record in records:
        if float(record['score']) < threshold:
            continue
        label = record['label']
        if label not in label_cats:
            label_cats[label] = []
        label_cats[label].append(record['category'])
        if (label not in label_score
                or label_score[label] < float(record['score'])):
            label_score[label] = float(record['score'])
    # Count each label, and note its cateogry
    label_counts = {}
    for label, cats in label_cats.items():
        label_counts[label] = len(cats)
        label_cats[label] = most_common(cats)
    # Sort labels by frequency and max score
    sorted_labels = sorted(label_counts.keys(),
            key=lambda x: (-label_counts[x], -label_score[x]))
    category_order = order
    if not category_order:
        # Default category order: broden order plus any missing categories
        category_order = list(default_category_order)
        for label in sorted_labels:
            if label_cats[label] not in category_order:
                category_order.append(label_cats[label])
    # Now make a plot
    heights = []
    colors = []
    categories = []
    labels = []
    for cat in category_order:
        filtered = [label for label in sorted_labels
            if label_cats[label] == cat]
        labels.extend(filtered)
        heights.extend([label_counts[label] for label in filtered])
        categories.append((cat, len(filtered)))
    # Sort records in histogram order and output them if requested
    if rendered_order is not None:
        rendered_order.extend(sorted(records, key=lambda record: (
            # Items below score threshold are sorted last, by score
            (len(category_order), 0, 0, -float(record['score']))
                if float(record['score']) < threshold else
            # Others are sorted by category, label count/score, and score
            (category_order.index(label_cats[record['label']]),
                -label_counts[record['label']], -label_score[record['label']],
                    -float(record['score'])))))
    filename = None
    if save:
        if save == True:
            filename = ed.filename('bargraph.svg', blob=blob, directory='html')
        else:
            filename = save
        ed.ensure_dir('html')
    return make_svg_bargraph(labels, heights, categories,
            barheight, barwidth, show_labels, filename)

def make_svg_bargraph(labels, heights, categories,
        barheight=100, barwidth=12, show_labels=True, filename=None):
    unitheight = float(barheight) / max(heights)
    textheight = barheight if show_labels else 0
    labelsize = float(barwidth)
    gap = float(barwidth) / 4
    textsize = barwidth + gap
    rollup = max(heights)
    textmargin = float(labelsize) * 2 / 3
    leftmargin = 32
    rightmargin = 8
    svgwidth = len(heights) * (barwidth + gap) + 2 * leftmargin + rightmargin
    svgheight = barheight + textheight

    # create an SVG XML element
    svg = et.Element('svg', width=str(svgwidth), height=str(svgheight),
            version='1.1', xmlns='http://www.w3.org/2000/svg')
 
    # Draw the bar graph
    basey = svgheight - textheight
    x = leftmargin
    # Add units scale on left
    for h in [1, (max(heights) + 1) // 2, max(heights)]:
        et.SubElement(svg, 'text', x='0', y='0',
            style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
            'alignment-baseline:hanging;' +
            'transform:translate(%dpx, %dpx);') %
            (textsize, x - gap, basey - h * unitheight)).text = str(h)
    et.SubElement(svg, 'text', x='0', y='0',
            style=('font-family:sans-serif;font-size:%dpx;text-anchor:middle;'+
            'transform:translate(%dpx, %dpx) rotate(-90deg)') %
            (textsize, x - gap - 1.2 * textsize, basey - h * unitheight / 2)
            ).text = 'units'
    # Draw big category background rectangles
    for catindex, (cat, catcount) in enumerate(categories):
        if not catcount:
            continue
        et.SubElement(svg, 'rect', x=str(x), y=str(basey - rollup * unitheight),
                width=(str((barwidth + gap) * catcount - gap)),
                height = str(rollup*unitheight),
                fill=palette[catindex % len(palette)][1])
        x += (barwidth + gap) * catcount
    # Draw small bars as well as 45degree text labels
    x = leftmargin
    catindex = -1
    catcount = 0
    for label, height in zip(labels, heights):
        while not catcount and catindex <= len(categories):
            catindex += 1
            catcount = categories[catindex][1]
            color = palette[catindex % len(palette)][0]
        et.SubElement(svg, 'rect', x=str(x), y=str(basey-(height * unitheight)),
                width=str(barwidth), height=str(height * unitheight),
                fill=color)
        x += barwidth
        if show_labels:
            et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
                'transform:translate(%dpx, %dpx) rotate(-45deg);') %
                (labelsize, x, basey + textmargin)).text = fix(label)
        x += gap
        catcount -= 1
    # Text lables for each category
    x = leftmargin
    for cat, catcount in categories:
        if not catcount:
            continue
        et.SubElement(svg, 'text', x='0', y='0',
            style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
            'transform:translate(%dpx, %dpx) rotate(-90deg);') %
            (textsize, x + (barwidth + gap) * catcount - gap,
                basey - rollup * unitheight + gap)).text = '%d %s' % (
                        catcount, fix(cat + ('s' if catcount != 1 else '')))
        x += (barwidth + gap) * catcount
    # Output - this is the bare svg.
    result = et.tostring(svg)
    if filename:
        f = open(filename, 'w')
        # When writing to a file a special header is needed.
        f.write('<?xml version=\"1.0\" standalone=\"no\"?>\n')
        f.write('<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n')
        f.write('\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n')
        f.write(result)
        f.close()
    return result

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s

# Usage:
# bargraph.py --directory dir --blob layer --barheight h --barwidth w 
if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    parser = argparse.ArgumentParser(
        description='Plot graph of unique interpretations.')
    parser.add_argument(
            '--directory',
            help='directory to graph')
    parser.add_argument(
            '--blob',
            nargs='*',
            help='blob to graph')
    parser.add_argument(
            '--barheight',
            type=int, default=100,
            help='graph big color bar height')
    parser.add_argument(
            '--barwidth',
            type=int, default=12,
            help='graph little color bar width')
    args = parser.parse_args()

    bar_graph_svg(args.directory, args.blob, args.barheight, args.barwidth,
            save=True)
