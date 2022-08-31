import json
import torch
import numpy as np
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.nets.pytorch_backend.e2e_asr import pad_list

def data_prep_func(args):

    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    ) 
    
    converter = CustomConverter()
    
    transform_tr = lambda data: converter([load_tr(data)])
    transform_cv = lambda data: converter([load_cv(data)])
    
    return train_json, valid_json, transform_tr, transform_cv

class ASRCustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[:: self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
                device, dtype=self.dtype
            )

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(
                    np.array(y[0][:]) if isinstance(y, tuple) else y
                ).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_pad, ilens, ys_pad
        
class CustomConverter(ASRCustomConverter):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.
        use_source_text (bool): use source transcription.

    """

    def __init__(
        self, subsampling_factor=1, dtype=torch.float32, use_source_text=False
    ):
        """Construct a CustomConverter object."""
        super().__init__(subsampling_factor=subsampling_factor, dtype=dtype)

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        text_tgt_pad: padded with 0 as input of text encoder
        ys_pad      : padded with -1 as input of the decoder
         

        """
        # batch should be located in list
        assert len(batch) == 1
        if len(batch[0]) == 3:
            xs, ys, ys_src = batch[0]
        else:
            xs, ys = batch[0]
            ys_src = ys

        # get batch of lengths of input sequences
        speech_ilens = np.array([x.shape[0] for x in xs])

        text_s_ilens = np.array([x.shape[0] for x in ys_src])
        text_t_ilens = np.array([x.shape[0] for x in ys])
 
        speech_ilens = torch.from_numpy(speech_ilens).to(device)
        text_s_ilens = torch.from_numpy(text_s_ilens).to(device)
        text_t_ilens = torch.from_numpy(text_t_ilens).to(device)
             
        speech_pad = pad_list([torch.from_numpy(x.copy()).float() for x in xs], 0).to(device, dtype=self.dtype)       
        text_src_pad = pad_list([torch.from_numpy(np.array(y, dtype=np.int64)) for y in ys_src], 0).to(device) 
        text_tgt_pad = pad_list([torch.from_numpy(np.array(y, dtype=np.int64)) for y in ys], 0).to(device) 
                           
        ys_pad = pad_list([torch.from_numpy(np.array(y, dtype=np.int64)) for y in ys], self.ignore_id,).to(device)              

        return speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad
 
