from utils import *

answer_these_questions('/home/an3729/datasets', '/home/an3729/checkpoints', '/home/an3729/datasets/test/1.jpg', ['Where does the image represent in the body?'], model_type = 'baseline', run_name = 'vqa_384_blstm_baseline_testf',
                           top_k = 218,  use_dropout = True, num_answers = 5, bi_directional = False,  use_lstm = True)
