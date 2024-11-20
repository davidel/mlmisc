import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu

from . import image_pad_concat as ipc
from . import module_builder as mb
from . import layer_utils as lu


def unet_down(net, num_channels, act, rids, shapes):
  net.batchnorm2d()
  rid = net.conv2d(num_channels, 3, stride=1, bias=False, padding=1)
  rids.append(rid)
  shapes[rid] = net.shape
  net.add(lu.create(act))
  net.conv2d(num_channels, 3, stride=1, bias=False, padding=1)
  net.add(lu.create(act))
  net.add(nn.MaxPool2d(2))


def unet_up(net, num_channels, act, rid, shape):
  in_channels = net.shape[0]

  net.deconv2d(in_channels, 2, stride=2, bias=False)
  net.batchnorm2d()
  net.deconv2d(in_channels, 3, stride=1, bias=False, padding=1)
  net.add(lu.create(act))
  net.add(ipc.ImagePadConcat(),
          input_fn=mb.inputtuple(rid),
          in_shapes=(net.shape, shape))
  net.deconv2d(num_channels, 3, stride=1, bias=False, padding=1)


def build_unet(net, channels, act, out_channels=None):
  out_channels = pyu.value_or(out_channels, net.shape[0])

  channels = (out_channels,) + tuple(channels)

  rids, shapes = [], dict()
  for num_channels in channels[1:]:
    unet_down(net, num_channels, act, rids, shapes)

  for i, num_channels in enumerate(reversed(channels[: -1])):
    rid = rids[- (i + 1)]
    unet_up(net, num_channels, act, rid, shapes[rid])

