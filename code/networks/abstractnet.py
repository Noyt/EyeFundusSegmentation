from torch import nn
from utils.io import *
from utils.tensors import *
from os.path import join
import datetime


class AbstractNet(nn.Module):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self._today = datetime.datetime.now().date()
        super(AbstractNet, self).__init__()

    def save_model(self, epoch=0, iteration=0, filename=None, loss=None, optimizers=None, use_datetime=True, **kwargs):
        if optimizers is not None:
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
        else:
            optimizers = []

        if filename is None:
            filename = "epoch_%i_iter_%i" % (epoch, iteration)
            if loss is not None:
                filename += "_loss_%f" % loss
        for k in kwargs:
            filename += '_'+k + '_%f' % kwargs[k]

        filename += '.pth'
        path = self.checkpoint+'/'
        if use_datetime:
            today = str(self._today)
            path = join(path, today + '/')

        create_folder(path)
        path = join(path, filename)
        save_dict = dict(model_state_dict=self.state_dict(), epoch=epoch)
        for i, optim in enumerate(optimizers):
            save_dict['optim_%i' % i] = optim.state_dict()

        torch.save(save_dict, path)

    def load(self, path, ignore_nan=False, load_most_recent=False, strict=False):
        device = torch.device('cpu')
        if load_most_recent:
            path = get_most_recent_file(path)
            print("Loading model", path)

        state_dict = torch.load(path, map_location=device)['model_state_dict']
        if not ignore_nan:
            check_nan(state_dict)
        self.load_state_dict(state_dict, strict=strict)
