import torch.optim as optim
import random
import torch
import numpy as np
import sklearn.metrics as sk
from torch import nn
from sys import path
from IPython import display
from utils.visualizer import Visualizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.tensors import MyDataParallel
from utils.tools import crop
import utils.losses as l
import torch.nn.functional as F
from utils.fundus_process import LAB_clahe, switch_side
from networks.bionet import BioNet
from networks.unet import UNet

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)


class DLManager:
    def __init__(self, config):

        self.config = config
        self.multi_gpu = False
        self.set_apex()
        self.set_junno()
        self.set_seed()
        if self.config.extern.gpu == 'none':
            self.device = torch.device('cpu')

        else:
            if not isinstance(self.config.extern.gpu, list):
                self.config.extern.gpu = [self.config.extern.gpu]
            self.multi_gpu = len(self.config.extern.gpu) > 1
            device_ids = ','.join([str(_) for _ in self.config.extern.gpu])
            self.device = torch.device("cuda:" + device_ids if torch.cuda.is_available() else "cpu")
            self.device = "cuda"

    def set_apex(self):
        """
        Apex allows using 16-bits tensor on GPU instead of traditional 32-bits tensors. Therefore, it's more efficient
        memory-wise and faster, but at a cost of a lower precision in computation
        :return:
        """
        if self.config.extern.apex:
            try:
                from apex import amp
                APEX_AVAILABLE = True
                self.amp = amp
            except ModuleNotFoundError:
                APEX_AVAILABLE = False
                print("Didn't succeed to load Apex, staying on 32 bits")

        self.use_apex = self.config.extern.apex and APEX_AVAILABLE and self.config.extern.gpu != 'none'

    def set_junno(self):
        """
        A convenient function to deal with uninstalled JuNNo library
        :return:
        """
        path.append(self.config.lib.path2junno)
        from junno.j_utils.math import ConfMatrix
        from junno.j_utils import log, Process

        self.Process = Process
        self.ConfMatrix = ConfMatrix

        display.display(log)

    def set_seed(self):
        """
        Fixing the seed to have deterministic behavior and reproducible results
        """
        seed = self.config.hp.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def set_model(self, model=None, load_path=None, use_most_recent=True):
        """

        :param model: Can be directly a pytorch Module. Otherwise, will use the default model (Bionet) EDIT: Changed to UNet
        :param load_path: Path indicating from where to load a pretrained set of weights. Can point to a folder containing
        multiples files (see argument below) or directly to a .pth file
        If None, the model is initialized accordingly to the default policy
        :param use_most_recent: If load_path points to a folder and not a file, the loading function will choose the most
        recent file in the folder (usually, the last saved during the training and therefore 'maybe' the best).
        :return:
        """
        if model is None:
            model = UNet(
                           checkpoint=self.config.hp.save_point,
                           config=self.config.model)
        if load_path is not None:
            model.load(load_path, load_most_recent=use_most_recent)

        self.model = model.to(self.device)
        self.setup_optims()

        if self.use_apex:
            self.model, self.optim = self.amp.initialize(self.model,
                                                         self.optim,
                                                         opt_level="O1",
                                                         loss_scale="dynamic")

        if self.multi_gpu:
            self.model = MyDataParallel(self.model, device_ids=self.config.extern.gpu)


class Trainer(DLManager):
    def __init__(self, config):
        super(Trainer, self).__init__(config)

        self.visu = Visualizer(self.config.extern.visdom_port, env="Arnaud's Preprocessing test %s" % self.config.variant.type)
        self.scores = []
        self.first_eval = True
        self.min_valid = float('inf')

    def setup_loss(self, class_counts=None, class_weights=None):
        """
        :param class_counts: N-element array (with N the number of classes). class_counts[i] is the number of pixels
        belonging in the class i in the training set (or at least a representative subset of the training set).
        If not None, it will be used to weight the loss function (give more importance to less represented classes)
        Otherwise, the loss function is simply averaged per pixel.
        """

        # We deactivate MSE weighting in case of MSE loss
        reduction = 'mean'
        if self.config.hp.loss == 'ce':
            if self.config.hp.classBalance == 'count':
                if class_counts is None:
                    self.loss = nn.CrossEntropyLoss(reduction=reduction)

                else:
                     normalized_hist_high = (class_counts / class_counts.sum()).astype(np.float32)

                     self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(1 - normalized_hist_high).to(self.device),
                                                   reduction=reduction)

            elif self.config.hp.classBalance == 'weight':
                self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(self.device),
                                                reduction=reduction)

        elif self.config.hp.loss == 'dice':
                self.loss = l.dice_loss2

        elif self.config.hp.loss == 'ce/dice':
                self.CE_loss = nn.NLLLoss(weight=torch.from_numpy(class_weights).to(self.device))
                self.dice_loss = l.dice_loss



    def set_datasets(self, train, valid):
        """
        :param train: Junno Dataset. Expected columns are 'x' and 'gt'
        :param valid: Junno Dataset. Expected columns are 'x' and 'gt'
        """
        self.train_set = train
        self.valid_set = valid

    def epoch(self, e):
        """
        This is where the training really happens (within the training loop)
        self.valid_set = valid

        self.valid_set = valid

        :param e: Current epoch
        self.valid_set = valid

        self.valid_set = valid

        :return:
        """
        generator = self.train_set.generator(n=self.config.hp.batch_size)
        batch_number = len(generator)
        training_losses = []

        self.total_batch_length = batch_number
        with self.Process('Epoch %i' % (e + 1), total=batch_number) as p_epoch:
            i = 0
            for b in generator:
                self.model.train()
                self.model.zero_grad()
                tensor_imgs = torch.from_numpy(b['x']).to(self.device)
                if tensor_imgs.size(0) == 1:  # Skip batch of size 1, cause they don't work with batchNorm
                    continue
                gt = torch.from_numpy(b['gt']).to(self.device)

                pred = self.model(tensor_imgs)

                if self.config.hp.loss == 'ce/dice':
                    probas = F.softmax(pred, dim=1)
                    logProbas = F.log_softmax(pred, dim=1)
                    loss = self.CE_loss(logProbas, torch.squeeze(gt)) + self.dice_loss(probas, gt)
                else:
                    loss = self.loss(pred, torch.squeeze(gt))

                if self.use_apex:
                    with self.amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optim.step()
                training_losses.append(loss.item())

                if i % self.config.hp.validation_frequency == 0 and i:
                    """
                    Here is where you should implement the validation policy, for example if you want to save the model
                    every time we reach a minimum in validation loss, or any other kind of relevant metric.
                    You can save a model by calling self.model.save(...)
                    """

                    validation_loss = self.validate(e, i,
                                                    training_loss=np.mean(training_losses))

                    if validation_loss < self.min_valid:
                        torch.save(self.model.state_dict(), '/home/clement/Documents/Arnaud/models/uNet_%s_%s.pth' % (self.config.variant.code, self.config.variant.type))
                        self.min_valid = validation_loss
                        #self.model.save_model(e, i, filename='')

                    training_losses = []
                    self.optim_lr_decay.step(validation_loss)

                p_epoch.update(1)
                i += 1

            p_epoch.succeed()

    def validate(self, epoch, index, **training_losses):
        """
        :param epoch: Current epoch
        :param index: Position within current epoch
        :param training_losses: Possible argument if you want to plot also the training loss during training
        :return:
        """

        if epoch % 10 == 0:
            print(50 * "*")
            print("Epoch %i, iteration %s" % (epoch + 1, index))

        global_index = self.total_batch_length * epoch + index

        self.model.eval()
        with torch.no_grad():
            generator_valid_set = self.valid_set.generator(n=self.config.hp.batch_size)

            list_pred = []
            list_proba = []
            losses = []
            'MODIFICATION'
            f_neg = 0
            f_pos = 0
            t_neg = 0
            t_pos = 0
            auc = 0

            with self.Process('Epoch validation', total=len(generator_valid_set)) as p_valid:
                for b in generator_valid_set:
                    tensor_imgs, gt = b.to_torch('x', 'gt', device=self.device)

                    pred = self.model(tensor_imgs)

                    if self.config.hp.loss == 'ce/dice':
                        probas = F.softmax(pred, dim=1)
                        logProbas = F.log_softmax(pred, dim=1)
                        loss = self.CE_loss(logProbas, torch.squeeze(gt)) + self.dice_loss(probas, gt)
                    else:
                        loss = self.loss(pred, torch.squeeze(gt))

                    #loss = self.loss(pred, torch.squeeze(gt))
                    losses.append(loss.item())

                    list_proba.append(1 - softmax(pred)[:, 0].cpu().numpy())

                    list_pred.append(torch.argmax(pred, 1).cpu().numpy())

                    'MODIFICATION'
                    bin_pred = torch.argmax(pred, 1).cpu().numpy()
                    gt_cpu = crop(torch.squeeze(gt).cpu().numpy(), bin_pred.shape)



                    p_valid.update(1)
        if self.first_eval:
            self.visu.draw_images(self.valid_set[:, 'x'], 'Images', bgr=True)
            self.visu.draw_images(self.valid_set[:, 'gt'], 'Groundtruth')
            self.first_eval = False

        self.visu.draw_images(np.squeeze(np.concatenate(list_proba, 0)), 'Probability', integer_prediction=False)

        self.visu.draw_images(np.squeeze(np.concatenate(list_pred, 0)), 'Prediction')
        labels = []
        total_losses = []
        for k in training_losses:
            labels.append(k)
            total_losses.append([training_losses[k]])
            if epoch % 10 == 0:
                print('Training %s %f' % (k, training_losses[k]))

        validation_loss = np.mean(losses)
        total_losses.append([validation_loss])
        labels.append('Validation loss')

        self.visu.draw_curves(Y=np.asarray(total_losses).transpose(),
                              X=[global_index], labels=labels,
                              win='losses')

        return validation_loss

    def setup_optims(self):
        """
        Optimisers used during training. By default, the learning rate is also decayed every time your validation performance
        stays stuck in a plateau. See the Yaml file for more details on the configuration
        """
        lr = self.config.hp.initial_lr

        if self.config.hp.optim == 'adam':

            self.optim = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=self.config.hp.weight_decay,
                                    eps=self.config.hp.adam_eps)
        elif self.config.hp.optim == 'sgd':
            self.optim = optim.ASGD(self.model.parameters(), lr=lr, weight_decay=self.config.hp.weight_decay)

        self.optim_lr_decay = ReduceLROnPlateau(self.optim,
                                                factor=self.config.hp.decay_lr,
                                                verbose=True,
                                                patience=self.config.hp.lr_patience_decay,
                                                min_lr=self.config.hp.minimal_lr)
        self.initial_lr_decay = self.optim_lr_decay.state_dict()
        self.initial_optim = self.optim.state_dict()

    def train(self):
        """
        Call this function to start the training
        """
        for e in range(self.config.hp.n_epochs):
            self.epoch(e)

    def reset_optims(self):
        """
        If you want to reset the state of the optimizers at some point.
        :return:
        """
        self.optim.load_state_dict(self.initial_optim)
        self.optim_lr_decay.load_state_dict(self.initial_lr_decay)
