'''
viewprobe creates visualizations for a certain eval.
'''

import os
import re
import numpy
import upsample
import loadseg
from scipy.misc import imread, imresize, imsave
from loadseg import normalize_label
import expdir
import bargraph

def open_dataset(ed):
    return loadseg.SegmentationData(ed.load_info().dataset)

# unit,category,label,score

def generate_html_summary(ed, ds, layer,
        imsize=None, imcount=50, imscale=None, tally_stats=None,
        gridwidth=None, gap=3, limit=None, force=False,
        include_hist=False, threshold=0.04, verbose=False):
    print 'Generating html summary', (
        ed.filename('html/%s.html' % expdir.fn_safe(layer)))
    # Grab tally stats
    # bestcat_pciou, name_pciou, score_pciou, _, _, _, _ = (tally_stats)
    if verbose:
        print 'Sorting units by score.'
        sys.stdout.flush()
    layerprobe = LayerProbe(ed, ds, layer)
    if imsize is None:
        imsize = layerprobe.input_dim[0]
    if imscale is None:
        imscale = imsize
    top = max_act_indexes(layerprobe, count=imcount)
    ed.ensure_dir('html','image')
    html = [html_prefix]
    rendered_order = []
    barfn = 'image/%s-bargraph.svg' % (
            expdir.fn_safe(layer))
    bargraph.bar_graph_svg(ed, layer, barheight=100,
            barwidth=12, threshold=threshold,
            rendered_order=rendered_order,
            save=ed.filename('html/' + barfn))
    html.extend([
        '<div class="histogram">',
        '<img class="img-fluid" src="%s" title="Summary of %s %s">' % (
            barfn, ed.basename(), layer),
        '</div>'
        ])
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    html.append('%d/%d units covering %d concepts with IoU &ge; %.2f' % (
        len([record for record in rendered_order
            if float(record['score']) >= threshold]),
        len(rendered_order),
        len(set(record['label'] for record in rendered_order
            if float(record['score']) >= threshold)),
        threshold))
    html.append('</div>')
    html.append(html_sortheader)
    html.append('</div>')

    if gridwidth is None:
        gridname = ''
        gridwidth = imcount
        gridheight = 1
    else:
        gridname = '-%d' % gridwidth
        gridheight = (imcount + gridwidth - 1) // gridwidth

    html.append('<div class="unitgrid"') # Leave off > to eat spaces
    if limit is not None:
        rendered_order = rendered_order[:limit]
    for i, record in enumerate(
            sorted(rendered_order, key=lambda record: -float(record['score']))):
        record['score-order'] = i
    for label_order, record in enumerate(rendered_order):
        unit = int(record['unit']) - 1 # zero-based unit indexing
        imfn = 'image/%s%s-%04d.jpg' % (
                expdir.fn_safe(layer), gridname, unit)
        if force or not ed.has('html/%s' % imfn):
            if verbose:
                print 'Visualizing %s unit %d' % (layer, unit)
                sys.stdout.flush()
            # Generate the top-patch image
            tiled = numpy.full(
                ((imsize + gap) * gridheight - gap,
                 (imsize + gap) * gridwidth - gap, 3), 255, dtype='uint8')
            for x, index in enumerate(top[unit]):
                row = x // gridwidth
                col = x % gridwidth
                vis = activation_visualization(ds, layerprobe, unit, index)
                if vis.shape[:2] != (imsize, imsize):
                    vis = imresize(vis, (imsize, imsize))
                tiled[row*(imsize+gap):row*(imsize+gap)+imsize,
                      col*(imsize+gap):col*(imsize+gap)+imsize,:] = vis
            imsave(ed.filename('html/' + imfn), tiled)
        # Generate the wrapper HTML
        graytext = ' lowscore' if float(record['score']) < threshold else ''
        html.append('><div class="unit%s" data-order="%d %d %d">' %
                (graytext, label_order, record['score-order'], unit + 1))
        html.append('<div class="unitlabel">%s</div>' % fix(record['label']))
        html.append('<div class="info">' +
            '<span class="layername">%s</span> ' % layer +
            '<span class="unitnum">unit %d</span> ' % (unit + 1) +
            '<span class="category">(%s)</span> ' % record['category'] +
            '<span class="iou">IoU %.2f</span>' % float(record['score']) +
            '</div>')
        html.append(
            '<div class="thumbcrop"><img src="%s" height="%d"></div>' %
            (imfn, imscale))
        html.append('</div') # Leave off > to eat spaces
    html.append('></div>')
    html.extend([html_suffix]);
    with open(ed.filename('html/%s.html' % expdir.fn_safe(layer)), 'w') as f:
        f.write('\n'.join(html))

def instance_data(ds, i, normalize=True):
    record, shape = ds.resolve_segmentation(
            ds.metadata(i), categories=None)
    if normalize:
        default_shape = (1, ) + shape
        record = dict((cat, normalize_label(dat, default_shape))
                for cat, dat in record.items())
    return record, shape

# Generates a mask at the "lp.level" quantile.
def activation_mask(lp, unit, index, shape):
    blobdata = lp.blobdata
    fieldmap = lp.fieldmap
    quantdata = lp.quantdata
    threshold = quantdata[unit, int(round(quantdata.shape[1] * lp.level))]
    up = upsample.upsampleL(
            fieldmap, blobdata[index:index+1, unit],
            shape=lp.input_dim, scaleshape=shape)[0]
    mask = up > threshold
    return mask

# Makes an iamge using the mask
def activation_visualization(ds, layerprobe, unit, index, alpha=0.2):
    image = imread(ds.filename(index))
    mask = activation_mask(layerprobe, unit, index, image.shape[:2])
    return (mask[:, :, numpy.newaxis] * (1 - alpha) + alpha) * image

def max_act_indexes(layerprobe, count=10):
    max_per_image = layerprobe.imgmax
    return max_per_image.argsort(axis=0)[:-1-count:-1,:].transpose()

class LayerProbe:
    def __init__(self, ed, ds, blob):
        info = ed.load_info(blob=blob)
        self.shape = info.shape
        self.fieldmap = info.fieldmap
        net_info = ed.load_info()
        self.input_dim = net_info.input_dim
        # Load the raw activation data
        self.blobdata = ed.open_mmap(blob=blob, shape=self.shape, mode='r')
        # Load the blob quantile data and grab thresholds
        self.quantdata = ed.open_mmap(blob=blob, part='quant-*',
                shape=(self.shape[1], -1), mode='r')
        # And load imgmax
        self.imgmax = ed.open_mmap(blob=blob, part='imgmax',
                shape=(ds.size(), self.shape[1]), mode='r')
        # Figure out tally level that was used.
        self.level = ed.glob_number(
                'tally-*.mmap', blob=blob, decimal=True)

html_prefix = '''
<!doctype html>
<html>
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
<style>
.unitviz, .unitviz .modal-header, .unitviz .modal-body, .unitviz .modal-footer {
  font-family: Arial;
  font-size: 15px;
}
.unitgrid {
  text-align: center;
  border-spacing: 5px;
  border-collapse: separate;
}
.unitgrid .info {
  text-align: left;
}
.unitgrid .layername {
  display: none;
}
.unitlabel {
  font-weight: bold;
  font-size: 150%;
  text-align: center;
  line-height: 1;
}
.lowscore .unitlabel {
   color: silver;
}
.thumbcrop {
  overflow: hidden;
  width: 288px;
  height: 72px;
}
.unit {
  display: inline-block;
  background: white;
  padding: 3px;
  margin: 2px;
  box-shadow: 0 5px 12px grey;
}
.iou {
  display: inline-block;
  float: right;
}
.modal .big-modal {
  width:auto;
  max-width:90%;
  max-height:80%;
}
.modal-title {
  display: inline-block;
}
.footer-caption {
  float: left;
  width: 100%;
}
.histogram {
  text-align: center;
  margin-top: 3px;
}
.img-wrapper {
  text-align: center;
}
.big-modal img {
  max-height: 60vh;
}
.img-scroller {
  overflow-x: scroll;
}
.img-scroller .img-fluid {
  max-width: initial;
}
.gridheader {
  font-size: 12px;
  margin-bottom: 10px;
  margin-left: 30px;
  margin-right: 30px;
}
.gridheader:after {
  content: '';
  display: table;
  clear: both;
}
.sortheader {
  float: right;
  cursor: default;
}
.layerinfo {
  float: left;
}
.sortby {
  text-decoration: underline;
  cursor: pointer;
}
.sortby.currentsort {
  text-decoration: none;
  font-weight: bold;
  cursor: default;
}
</style>
</head>
<body class="unitviz">
<div class="container-fluid">
'''

html_sortheader = '''
<div class="sortheader">
sort by
<span class="sortby currentsort" data-index="0">label</span>
<span class="sortby" data-index="1">score</span>
<span class="sortby" data-index="2">unit</span>
</div>
'''

html_suffix = '''
</div>
<div class="modal" id="lightbox">
  <div class="modal-dialog big-modal" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title"></h5>
        <button type="button" class="close"
             data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <div class="img-wrapper img-scroller">
          <img class="fullsize img-fluid">
        </div>
      </div>
      <div class="modal-footer">
        <div class="footer-caption">
        </div>
      </div>
    </div>
  </div>
</div>
<script>
$('img:not([data-nothumb])[src]').wrap(function() {
  var result = $('<a data-toggle="lightbox">')
  result.attr('href', $(this).attr('src'));
  var caption = $(this).closest('figure').find('figcaption').text();
  if (!caption && $(this).closest('.citation').length) {
    caption = $(this).closest('.citation').text();
  }
  if (caption) {
    result.attr('data-footer', caption);
  }
  var title = $(this).attr('title');
  if (!title) {
    title = $(this).closest('td').find('.unit,.score').map(function() {
      return $(this).text(); }).toArray().join('; ');
  }
  if (title) {
    result.attr('data-title', title);
  }
  return result;
});
$(document).on('click', '[data-toggle=lightbox]', function(event) {
    $('#lightbox img').attr('src', $(this).attr('href'));
    $('#lightbox .modal-title').text($(this).data('title') ||
       $(this).closest('.unit').find('.unitlabel').text());
    $('#lightbox .footer-caption').text($(this).data('footer') ||
       $(this).closest('.unit').find('.info').text());
    event.preventDefault();
    $('#lightbox').modal();
    $('#lightbox img').closest('div').scrollLeft(0);
});
$(document).on('keydown', function(event) {
    $('#lightbox').modal('hide');
});
$(document).on('click', '.sortby', function(event) {
    var sortindex = +$(this).data('index');
    sortBy(sortindex);
    $('.sortby').removeClass('currentsort');
    $(this).addClass('currentsort');
});
function sortBy(index) {
  $('.unitgrid').find('.unit').sort(function (a, b) {
     return +$(a).eq(0).data('order').split(' ')[index] -
            +$(b).eq(0).data('order').split(' ')[index];
  }).appendTo('.unitgrid');
}
</script>
</body>
</html>
'''

replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def fix(s):
    for pattern, subst in replacements:
        s = re.sub(pattern, subst, s)
    return s


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser(
            description='Generate visualization for probed activation data.')
        parser.add_argument(
                '--directory',
                default='.',
                help='output directory for the net probe')
        parser.add_argument(
                '--blobs',
                nargs='*',
                help='network blob names to visualize')
        parser.add_argument(
                '--gridwidth',
                type=int, default=None,
                help='width of visualization grid')
        parser.add_argument(
                '--imsize',
                type=int, default=None,
                help='thumbnail dimensions')
        parser.add_argument(
                '--force',
                type=int, default=0,
                help='set to 1 to force image regeneration')
        parser.add_argument(
                '--imscale',
                type=int, default=72,
                help='thumbnail dimensions')
        parser.add_argument(
                '--imcount',
                type=int, default=50,
                help='number of thumbnails to include')
        parser.add_argument(
                '--threshold',
                type=float, default=0.04,
                help='minimum IoU to count as a detector')
        args = parser.parse_args()
        ed = expdir.ExperimentDirectory(args.directory)
        ds = open_dataset(ed)
        for blob in args.blobs:
            generate_html_summary(ed, ds, blob,
                    imsize=args.imsize, imscale=args.imscale,
                    imcount=args.imcount,
                    gridwidth=args.gridwidth,
                    force=args.force,
                    threshold=args.threshold,
                    verbose=True)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
