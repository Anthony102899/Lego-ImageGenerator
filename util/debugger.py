import time
import datetime
import os
import sys
import logging
import numpy as np
import random


class MyDebugger():
    pre_fix = "../debug"

    def __init__(self, model_name: str, fix_rand_seed = None, save_print_to_file = False):
        if fix_rand_seed is not None:
            np.random.seed(seed=fix_rand_seed)
            random.seed(fix_rand_seed)
            # torch.manual_seed(fix_rand_seed)
        if isinstance(model_name, str):
            self.model_name = model_name
        else:
            self.model_name = '_'.join(model_name)

        self._debug_dir_name = os.path.join(os.path.dirname(__file__), MyDebugger.pre_fix,
                                            datetime.datetime.fromtimestamp(time.time()).strftime(
                                                f'%Y-%m-%d_%H-%M-%S_{self.model_name}'))
        # self._debug_dir_name = os.path.join(os.path.dirname(__file__), self._debug_dir_name)
        print("=================== Program Start ====================")
        print(f"Output directory: {self._debug_dir_name}")
        self._init_debug_dir()

        ######## redirect the standard output
        if save_print_to_file:
            sys.stdout = open(self.file_path("print.log"), 'w')

    def file_path(self, file_name):
        return os.path.join(self._debug_dir_name, file_name)

    def _init_debug_dir(self):
        # init root debug dir
        os.makedirs(self._debug_dir_name)
        logging.info("Directory %s established" % self._debug_dir_name)


if __name__ == '__main__':
    import logging

    LOG_FILENAME = 'debug\\2019-07-03_17-43-27_circular\\info.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    logging.debug('This message should go to the log file')
