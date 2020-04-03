from tqdm import tqdm
from models.resnet import resnet18
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import torchvision

class Engine:
    def __init__(self, __C, logger):
        self.__C = __C
        self.logger = logger
        self.init_engine()
        self.init_checkpoint()
        if self.__C.mode in ["test"] or self.__C.init in ['pretrained', 'resume']:
            self.load_checkpoint()


    def init_engine(self):
        self.net = resnet18(num_classes = self.__C.num_classes, pretrained=True)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr = 0.001)
        self.best_accuracy_top1 = -1
        self.best_epoch = 0
        self.cur_epoch = 0
        self.cur_iter = 0

    def init_checkpoint(self):
        self.checkpoint = {
            "net": None,
            "optimizer": None,
            "engine": {"best_accuracy_top1": None,
                       "best_epoch": None,
                       "cur_epoch": None,
                       "cur_iter": None}
        }


    def update_checkpoint(self):
        self.checkpoint['net'] = self.net.state_dict()
        self.checkpoint['optimizer'] = self.optimizer.state_dict()
        self.checkpoint['engine'] = {"best_accuracy_top1": self.best_accuracy_top1,
                       "best_epoch": self.best_epoch,
                       "cur_epoch": self.cur_epoch,
                       "cur_iter": self.cur_iter}

    def save_checkpoint(self, postfix=""):
        torch.save(self.checkpoint, self.__C.ckpt_path+postfix)


    def load_checkpoint(self):
        if self.__C.mode in ["test"]:
            ckpt_path = self.__C.test_ckpt_path
        elif self.__C.init in ['pretrained']:
            ckpt_path = self.__C.pretrained_ckpt_path
        elif self.__C.init in ['resume']:
            ckpt_path = self.__C.resume_ckpt_path
        self.checkpoint = torch.load(ckpt_path)
        self.net.load_state_dict(self.checkpoint['net'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        if self.__C.init != "pretrained":
            for k,v in self.checkpoint['engine'].items():
                setattr(self, k, v)



    def run(self, train_loader, val_loader=None):
        if self.__C.mode == 'test':
            self.eval(val_loader, self.net, self.cur_epoch)
        else:
            self.train(train_loader, val_loader)

    def train(self, train_loader, val_loader=None):
        lossfunc = F.cross_entropy
        net = self.net
        optimizer = self.optimizer

        net.train()

        for epoch in range(self.cur_epoch, self.__C.max_epoch):
            self.cur_epoch += 1
            tq = tqdm(train_loader, desc='{} E{:03d}'.format('train', epoch), ncols=100)
            for batch_idx, item in enumerate(tq):
                self.cur_iter += 1
                data = item['data'].cuda()
                target = item['class_id'].cuda()
                optimizer.zero_grad()
                output = net(data)
                loss = lossfunc(output, target)
                loss.backward()
                optimizer.step()
                tq.set_postfix(loss='{:.4f}'.format(loss.item()), comp='{}'.format(batch_idx * len(data)))


            if self.__C.val_epoch_every > 0 and self.cur_epoch % self.__C.val_epoch_every == 0:
                result = self.eval(val_loader, net, epoch)
                if result['accuracy_top1'] > self.best_accuracy_top1:
                    self.best_accuracy_top1 = result['accuracy_top1']
                    self.best_epoch = epoch
                    self.update_checkpoint()
                    self.save_checkpoint("_best")
                self.update_checkpoint()
                self.save_checkpoint("_E{}".format(self.cur_epoch))




    def eval(self, val_loader, net=None, epoch=0, topk=[1,5]):
        net.eval()
        lossfunc = F.cross_entropy
        tq = tqdm(val_loader, desc='{} E{:03d}'.format('val', epoch), ncols=100)
        test_loss = 0
        correct = {k: 0 for k in topk}
        maxk = max(topk)
        with torch.no_grad():
            for batch_idx, item in enumerate(tq):
                data = item['data'].cuda()
                target = item['class_id'].cuda()
                output = net(data)
                loss = lossfunc(output, target)
                test_loss += loss.item()

                # calculate top-k
                _, pred = output.topk(maxk, 1, True, True)
                batch_correct = pred.eq(target.view(-1, 1).expand_as(pred))
                for k in topk:
                    correct[k] += batch_correct[:, :k].sum().item()
                tq.set_postfix(loss='{:.4f}'.format(loss.item()), comp='{}'.format(batch_idx * len(data)))
            for k in correct:
                self.logger.tensorboard.add_scalar("eval_acc{}_{}".format(k, self.__C.mode), correct[k]/len(val_loader.dataset), self.cur_iter)
            self.logger.tensorboard.add_image("eval_input_images_{}".format(self.__C.mode), data[epoch]*0.5+0.5, self.cur_iter)
            self.logger.tensorboard.add_image("eval_grid_images_{}".format(self.__C.mode), torchvision.utils.make_grid(data*0.5+0.5, nrow=4, normalize=False), self.cur_iter)


        test_loss /= len(val_loader.dataset)
        result = {'epoch': epoch, 'loss': test_loss}
        for k in topk:
            keyname = "accuracy_top{}".format(k)
            result[keyname] = 100. * correct[k] / len(val_loader.dataset)
            log_msg = '\n[Epoch: {}] Val set: Average loss: {:.4f}, Top-{} Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, test_loss, k, correct[k], len(val_loader.dataset),
                100. * correct[k] / len(val_loader.dataset))
            self.logger.filelogger.info(log_msg)



        return result