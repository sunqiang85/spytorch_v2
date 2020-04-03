from torch.utils.tensorboard import SummaryWriter
import logging
class Logger():
    def __init__(self, __C):
        self.__C = __C
        self.tensorboard = SummaryWriter(__C.tensorboard_path)
        self.filelogger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        fh = logging.FileHandler(__C.log_path, mode='w')
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        self.filelogger.addHandler(fh)



