import math
from typing import Iterable, List, Union
import torch
import numpy as np
from torch.types import Number
from lightwood.encoder.base import BaseEncoder
from lightwood.encoder.helpers import MinMaxNormalizer
from lightwood.helpers.log import log
from lightwood.helpers.general import is_none
from lightwood.api.dtype import dtype


class NumericNormalizerEncoder(BaseEncoder):
    """
    The numeric normalizer encoder takes numbers (float or integer) and applies MinMaxScaler and converts it into tensors.

    The ``min`` and ``max`` values are computed in the ``prepare`` method and is just the minimum and maximum of the values of all numbers fed to prepare (all none values become 0)

    ``none`` stands for any number that is an actual python ``None`` value or any sort of non-numeric value (a string, nan, inf)
    """ # noqa

    is_trainable_encoder: bool = True

    def __init__(self, data_type: dtype = None, is_target: bool = False, positive_domain: bool = False):
        """
        :param data_type: The data type of the number (integer, float, quantity)
        :param is_target: Indicates whether the encoder refers to a target column or feature column (True==target)
        :param positive_domain: Forces the encoder to always output positive values
        """
        super().__init__(is_target)
        self._type = data_type
        self.positive_domain = positive_domain
        self.decode_log = False
        self.output_size = 1

    def prepare(self, priming_data: Iterable, dev_priming_data: Iterable):
        """
        Prepare the array encoder for sequence data.
        :param train_priming_data: Training data of sequences
        :param dev_priming_data: Dev data of sequences
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        
        value_type = 'int'
        for number in priming_data:
            if not is_none(number):
                if int(number) != number:
                    value_type = 'float'
                
        self._type = value_type if self._type is None else self._type

        self._normalizer = MinMaxNormalizer()  # maybe turn into numerical encoder?

        # if isinstance(priming_data, pd.Series):
        #     priming_data = priming_data.values

        self._normalizer.prepare(priming_data)
        
        self.is_prepared = True

    def encode(self, data: Iterable) -> torch.Tensor:
        """
        :param data: An iterable data structure containing the numbers to be encoded

        :returns: A torch tensor with the representations of each number
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        # if isinstance(column_data, pd.Series):
        #     column_data = column_data.values

        ret = []
        for real in data:
            try:
                real = float(real)
            except Exception:
                real = None
                
            if self.is_target:
                val = real

            else:
                try:
                    if is_none(real):
                        val = 0
                    else:
                        val = real
                except Exception as e:
                    val = 0
                    log.error(f'Can\'t encode input value: {real}, exception: {e}')

            ret.append(val)

        data = torch.cat([self._normalizer.encode(np.array(ret).reshape(-1,1))], dim=-1)
        data[torch.isnan(data)] = 0.0
        data[torch.isinf(data)] = 0.0

        return data

    def decode(self, data: torch.Tensor) -> List[Iterable]:
        """
        Converts data as a list of arrays.

        :param data: Encoded data prepared by this array encoder
        :returns: A list of iterable sequences in the original data space
        """
        
        decoded = self._normalizer.decode(data.reshape(-1,1))
        return decoded.ravel()