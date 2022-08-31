import configargparse
import argparse
import torch

from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES
from espnet.utils.cli_utils import strtobool

def get_parser(parser=None, required=True):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train a speech translation (ST) model on one CPU, "
            "one or multiple GPUs",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
    #yaml file parse
    parser.add("--config", is_config_file=True, help="config file path")
    
    #preprocess
    parser.add_argument("--preprocess-conf", type=str, default=None, nargs="?",help="The configuration file for the pre-processing",)
    
    #trainer parameters
    parser.add_argument("--gpus", default=None, type=int, help="Number of GPUs")
    parser.add_argument("--num-nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--accelerator", default="ddp", type=str, choices=["dp", "ddp"],)
    parser.add_argument("--replace-sampler-ddp", default=False, type=strtobool)
    parser.add_argument("--pin-memory", default=True, type=strtobool)
    parser.add_argument("--num-workers", default=8, type=int, help="number workers when load data used in kaldi_loder")
    
    #optimizer
    parser.add_argument("--accumulate-grad-batches", default=1, type=int, help="Number of gradient accumuration")
    parser.add_argument("--gradient-clip-val", default=5, type=float, help="Gradient norm threshold to clip")
    
    #parser.add_argument("--max-epochs", "-e", default=30, type=int, help="Maximum number of epochs")
    
    #log
    parser.add_argument("--name", default=None, type=str, help='subdir of logs')
    
    #reproduce
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--deterministic", default=True, type=strtobool, help="Random seed")
    
    #early stop
    parser.add_argument("--criterion", default="loss", type=str, choices=["loss", "acc"], help="Criterion to perform epsilon decay",)
    parser.add_argument("--threshold", default=0.0, type=float, help="Threshold to stop iteration")
    parser.add_argument("--patience", default=3, type=int, nargs="?", help="Number of epochs to wait without improvement before stopping the training")
    
    #debug
    parser.add_argument("--minibatches", type=int, default="-1", help="Process only N minibatches (for debug)",)
    parser.add_argument("--verbose", type=int, default="1", help="logging setting",)
    
    # task related
    parser.add_argument("--dict", required=required, help="Dictionary")
    parser.add_argument("--tasks", type=str,  default='speaker', help="speaker or intention",)
    parser.add_argument("--stage", type=str,  default='train', choices=["train", "test"], help="speaker or intention",)
    parser.add_argument("--out-layer", type=int,  default='12',  help="speaker or intention",)
    parser.add_argument("--train-data", type=str,  default=None, help="Filename of train label data (json)",)
    parser.add_argument("--valid-data", type=str, default=None,help="Filename of validation label data (json)",)
    parser.add_argument("--test-data", type=str, default=None,help="Filename of validation label data (json)",)
    parser.add_argument("--test-model-path", type=str, default=None,help="test model path",)
        
    # loss related
    parser.add_argument("--ctc_type", default="warpctc", type=str, choices=["builtin", "warpctc"], help="Type of CTC implementation to calculate loss.",)
    parser.add_argument("--ctc-layer", default=-1, type=int, help="the nth layer ctc is calculate on",)
    parser.add_argument("--ctc-shrink", default=False, type=strtobool)
    parser.add_argument("--mtlalpha", default=0.0, type=float, help="Multitask learning coefficient, alpha: \
                                                                     alpha*ctc_loss + (1-alpha)*att_loss",)
                                                                     
    parser.add_argument("--asr-weight", default=0.0, type=float, help="Multitask learning coefficient for ASR task, weight: "
                                                                      "asr_weight*(alpha*ctc_loss + (1-alpha)*att_loss) + (1-asr_weight-mt_weight)*st_loss",)
                                                                      
    parser.add_argument("--mt-weight", default=0.0, type=float, help="Multitask learning coefficient for MT task, weight: \
                                                                     mt_weight*mt_loss + (1-mt_weight-asr_weight)*st_loss",)
                                                                     
    parser.add_argument( "--lsm-weight", default=0.0, type=float, help="Label smoothing weight")
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size")
    parser.add_argument("--penalty", default=0.0, type=float, help="Incertion penalty")

    parser.add_argument("--lm-weight", default=0.0, type=float, help="RNNLM weight.")
    parser.add_argument("--sym-space", default="<space>", type=str, help="Space symbol")
    parser.add_argument("--sym-blank", default="<blank>", type=str, help="Blank symbol")
    
    # minibatch related
    parser.add_argument("--sortagrad", default=0, type=int, nargs="?", help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",)
    
    parser.add_argument("--batch-count", default="auto", choices=BATCH_COUNT_CHOICES, help="How to count batch_size. ")
    parser.add_argument("--batch-size", "--batch-seqs", "-b", default=0, type=int, help="Maximum seqs in a minibatch (0 to disable)",)
    parser.add_argument("--batch-frames", default=0, type=int, help="Maximum bins in a minibatch (0 to disable)",)
    parser.add_argument("--batch-frames-in", default=0, type=int, help="Maximum input frames in a minibatch (0 to disable)",)
    parser.add_argument("--batch-frames-out", default=0, type=int, help="Maximum output frames in a minibatch (0 to disable)",)
    parser.add_argument("--batch-frames-inout", default=0, type=int, help="Maximum input+output frames in a minibatch (0 to disable)",)
    parser.add_argument("--maxlen-in", "--batch-seq-maxlen-in", default=800, type=int, metavar="ML", help="When --batch-count=seq, batch size is reduced if the input sequence length > ML.")
    parser.add_argument("--maxlen-out", "--batch-seq-maxlen-out", default=150, type=int, metavar="ML", help="When --batch-count=seq,batch size is reduced if the output sequence length > ML")

    
    # optimization related
    parser.add_argument("--opt", default="noam", type=str, choices=["adadelta", "adam", "noam"], help="Optimizer",)
    parser.add_argument("--eps", default=1e-8, type=float, help="Epsilon constant for optimizer")
    parser.add_argument("--eps-decay", default=0.01, type=float, help="Decaying ratio of epsilon")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for optimizer")
    parser.add_argument("--lr-decay", default=1.0, type=float, help="Decaying ratio of learning rate")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight decay ratio")

    parser.add_argument("--grad-noise", type=strtobool, default=False, help="The flag to switch to use noise injection to gradients during training",)
    
    # speech translation related ??? this parameters don't understand
    parser.add_argument("--context-residual", default=False, type=strtobool, nargs="?", help="The flag to switch to use context vector residual in the decoder network",)
    
    # finetuning related
    parser.add_argument("--prob-lr", default=1e-4, type=float, help="finetune-lr",)
    parser.add_argument("--enc-model-prefix", default=None, type=str, help="Pre-trained model prefix, such as: model_asr.model",)
    parser.add_argument("--dec-model-prefix", default=None, type=str, help="Pre-trained model prefix, such as: model_st.model",)
    parser.add_argument("--enc-init", default=None, type=str, nargs="?", help="Pre-trained ASR model to initialize encoder.",)
    parser.add_argument("--enc-init-mods", default="enc.enc.",type=lambda s: [str(mod) for mod in s.split(",") if s != ""],help="List of encoder modules to initialize, separated by a comma.",)
    
    parser.add_argument("--dec-init",default=None, type=str, nargs="?", help="Pre-trained ASR, MT or LM model to initialize decoder.",)
    parser.add_argument("--dec-init-mods", default="att., dec.", type=lambda s: [str(mod) for mod in s.split(",") if s != ""],help="List of decoder modules to initialize, separated by a comma.",)
    
    # multilingual related
    parser.add_argument("--multilingual", default=False, type=strtobool, help="Prepend target language ID to the source sentence. \
                                                                               Both source/target language IDs must be prepend in the pre-processing stage.",)
    parser.add_argument("--replace-sos", default=False, type=strtobool, help="Replace <sos> in the decoder with a target language ID \
                                                                             (the first token in the target sequence)",)
    
    return parser
