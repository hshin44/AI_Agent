python process_data.py\
    --EMBEDDING_DIM 100\
    --POSITIVE_DATA data/content_EF-morph.pos\
    --NEGATIVE_DATA data/content_EF-morph.neg\
    --WORD_EMBEDDING matrices/matrix-morph-skip.txt\
    --DYNAMIC_VAR 1

THEANO_FLAGS=mode=FAST_RUN,device=cuda*,floatX=float32\
    python conv_net_sentence.py\
        --EMBEDDING_DIM 100\
        --STATIC 0\
        --RANDOM 0\
        --MAX_LENGTH 70
