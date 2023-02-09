from nnutils.functional import *
import numpy as np


class LeNet:

    def __init__(self, psum_range_dict):
        self.psum_range = psum_range_dict
        self.weightsDict, self.scalesDict = getAllParms()
        self.psum_record_dict = {}

    def forward(self, x, psum_record=False):
        # TODO
        # You should get the record of partial sums by `x, self.psum_record_dict['c1'] = Conv2d(...)`.
        x = ActQuant(x, self.scalesDict['input_scale'], 0)
        x, self.psum_record_dict['c1'] = Conv2d(self.psum_range['c1'], x, self.weightsDict["c1.conv"], out_channels=6, psum_record=psum_record)
        x = ReLU(x)
        x = ActQuant(x, self.scalesDict['c1_output_scale'])
        x = MaxPool2d(x)
        x, self.psum_record_dict['c3'] = Conv2d(self.psum_range['c3'], x, self.weightsDict["c3.conv"], out_channels=16, psum_record=psum_record)
        x = ReLU(x)
        x = ActQuant(x, self.scalesDict['c3_output_scale'])
        x = MaxPool2d(x)
        x, self.psum_record_dict['c5'] = Conv2d(self.psum_range['c5'], x, self.weightsDict["c5.conv"], out_channels=120, psum_record=psum_record)
        x = ReLU(x)
        x = ActQuant(x, self.scalesDict['c5_output_scale'])
        x = x.reshape(-1, 120)
        x, self.psum_record_dict['f6'] = Linear(self.psum_range['f6'], x, self.weightsDict["f6.fc"], psum_record=psum_record)
        x = ReLU(x)
        x = ActQuant(x, self.scalesDict['f6_output_scale'])
        x, self.psum_record_dict['output'] = Linear(self.psum_range['output'], x, self.weightsDict["output.fc"], self.weightsDict["outputBias"], psum_record=psum_record)
        x = ActQuant(x, self.scalesDict['output_output_scale'])

        return x
