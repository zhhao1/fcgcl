MAIN_ROOT=$PWD
KALDI_ROOT=$MAIN_ROOT/src/kaldi-master
SRC_ROOT=$MAIN_ROOT/src
espnet_root=/home/speech/espnet-master
# BEGIN from kaldi path.sh
#[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
#export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD/utils:$PATH
#[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
#. $KALDI_ROOT/tools/config/common_path.sh

#conda deactivate && conda activate espnet

export PATH=$PWD:$MAIN_ROOT/src/tools:$MAIN_ROOT/src/tools/moses/scripts/tokenizer/:$MAIN_ROOT/tools/moses/scripts/generic/:$SRC_ROOT/bins:$PATH
export LC_ALL=C
# END

export PYTHONPATH=$SRC_ROOT:$MAIN_ROOT:$SRC_ROOT/tools/espnet:$SRC_ROOT/tools:$PYTHONPATH

if [ -e $espnet_root/tools/venv/etc/profile.d/conda.sh ]; then
    source $espnet_root/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $espnet_root/tools/venv/bin/activate
fi


export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
