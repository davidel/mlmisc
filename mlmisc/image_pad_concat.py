import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def image_pad_concat(*imgs, channels_dim=None):
  channels_dim = pyu.value_or(channels_dim, -3)
  if channels_dim > 0 and imgs:
    channels_dim = -(imgs[0].ndim - channels_dim)

  max_dims = [-1] * abs(channels_dim + 1)
  for img in imgs:
    for i, dim in enumerate(range(-1, channels_dim, -1)):
      max_dims[i] = max(max_dims[i], img.shape[dim])

  padded_imgs = []
  for img in imgs:
    pad = []
    # Enumerate generating the W[,H[,D]] pad sequence for 1,2,3D images.
    for i, dim in enumerate(range(-1, channels_dim, -1)):
      diff = max_dims[i] - img.shape[dim]
      pad.append(diff // 2)
      pad.append(diff - pad[-1])

    padded_imgs.append(nn.functional.pad(img, pad))

  return torch.cat(padded_imgs, dim=channels_dim)


class ImagePadConcat(nn.Module):

  def forward(self, *imgs):
    return image_pad_concat(*imgs)

