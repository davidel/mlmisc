from . import nets_dict as netd


class ArgsBase(netd.NetsDict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

