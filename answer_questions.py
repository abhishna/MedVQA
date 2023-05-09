from utils import *

import argparse

def boolstr(s):
    """
        Argument parser for boolean string.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description='VQA')

    parser.add_argument('--data_dir',               type=str,       help='directory of the preprocesses data', default='/scratch/crg9968/datasets')
    parser.add_argument('--model_dir',              type=str,       help='directory to store model checkpoints (saved as run_name.pth)', default='/scratch/crg9968/checkpoints')
    parser.add_argument('--run_name',               type=str,       help='unique experiment name (used as prefix for all data saved on a run)', default='testrun', required=True)
    parser.add_argument('--model',                  type=str,       help='VQA model choice', choices=['baseline'], default='baseline')
    parser.add_argument('--image_model_type',       type=str,       help='Type of CNN for the Image Encoder', choices=['vgg16', 'resnet152'], default='vgg16')

    parser.add_argument('--top_k_answers',          type=int,       help='Top K answers used to train the model (output classifier size)', default=1000)
    parser.add_argument('--max_length',             type=int,       help='max sequence length of questions', default=14)
    parser.add_argument('--word_embedding_size',    type=int,       help='Word embedding size for the embedding layer', default=300)
    parser.add_argument('--lstm_state_size',        type=int,       help='LSTM hidden state size', default=512)
    parser.add_argument('--use_dropout',            type=boolstr,   help='use dropout', default=True)
    parser.add_argument('--attention_mechanism',    type=str,       help='method of combining image and text embeddings', choices=['element_wise_product', 'sum', 'concat'], default='element_wise_product')
    parser.add_argument('--bi_directional',         type=boolstr,   help='True if lstm is to be bi-directional', choices=[True, False], default=False)
    parser.add_argument('--use_lstm',               type=boolstr,   help='True if lstm is to be used', choices=[True, False], default=True)
    parser.add_argument('--use_glove',              type=boolstr,   help='True if glove embeddings are to be used', choices=[True, False], default=False)
    parser.add_argument('--embedding_file_name',    type=str,       help='glove embedding path file', default='word_embeddings_glove.pkl')
    parser.add_argument('--num_answers',            type=int,       help='return top num_answers for each question asked', default=5)

    parser.add_argument('--image_id',               type=int,       help='image id to answer questions for', required = True)
    parser.add_argument('--image_loc',              type=str,       help='the location of the image id passed, either train, val or test directories', choices=['train', 'val', 'test'], default='val')
    args = parser.parse_args()

    # Get Image to questions map
    imgToQ     = get_image_to_questions(args.data_dir, mode = args.image_loc)
    image_path = get_image_path(args.data_dir, args.image_id, mode = args.image_loc)
    questions  = imgToQ[args.image_id][:3]

    # Answer 3 questions for that image
    answer_these_questions(args.data_dir, args.model_dir, image_path, questions, model_type = args.model, run_name = args.run_name,
                           top_k = args.top_k_answers, max_length = args.max_length, image_model_type = args.image_model_type,
                           word_embedding_size = args.word_embedding_size, lstm_hidden_size = args.lstm_state_size,
                           use_dropout = args.use_dropout, attention_mechanism = args.attention_mechanism, num_answers = args.num_answers,
                           bi_directional = args.bi_directional, use_glove = args.use_glove, use_lstm = args.use_lstm, embedding_file_name = args.embedding_file_name)

if __name__ == '__main__':
    main()
