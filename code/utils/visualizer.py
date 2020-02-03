import visdom
import numpy as np
import math


class Visualizer:
    def __init__(self, port, env):
        self.vis = visdom.Visdom(port=port)
        self.ncol = 4
        self.env = env
        self.images_windows = []
        self.curves_windows = []
        self.display_id = 0

    def draw_curves(self, Y, win, labels, X=None):
        if win not in self.curves_windows:
            self.curves_windows.append(self.vis.line(Y, X=X, opts={'legend': labels}, win=win, env=self.env))
        else:
            self.vis.line(Y, X=X, opts={'legend': labels}, win=win, update='append', env=self.env)

    def draw_images(self, images, title, integer_prediction=True, bgr=False):

        if images.ndim == 3:
            images = np.expand_dims(images, 1)

        b, c, h, w = images.shape
        images = images[:, :3, :, :]
        if bgr:
            images = images[:, ::-1, :, :]
        if integer_prediction:
            images = (255*images/np.max(images)).astype(np.uint)

        nrows = math.ceil(b / self.ncol)
        width = 200
        factor = width / w
        height = factor * h

        opts = dict(height=int(height * nrows), width=self.ncol * width, title=title)
        if title not in self.images_windows:
            self.images_windows.append(title)
            self.vis.images(images, nrow=self.ncol, padding=2, opts=opts, env=self.env,
                            win=self.display_id)
            self.display_id += 1
        else:
            self.vis.images(images, nrow=self.ncol, padding=2, opts=opts,
                            win=self.images_windows.index(title), env=self.env)
