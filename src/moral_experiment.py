import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser for training script.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--nb_examples', type=int, default=8400, help='Number of training examples')
    parser.add_argument('--n_epoch', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--en_ratio', type=float, default=0.5, help='Ratio EN')

    parser.add_argument('--dataset_name', type=str, choices=['moral', 'jigsaw'], default='moral', help='Dataset name')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.3", help='Model name')
    parser.add_argument('--ref_model', type=str, default='mistral', help='Reference model')
    parser.add_argument('--loss_type', type=str, default='sigmoid_reg_ppl', help='Loss to use')
    parser.add_argument('--gamma', type=float, default=1e-2, help='Gamma')
    parser.add_argument('--beta', type=float, default=1e-1, help='Beta')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1e-1, help='Max grad norm')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Lr scheduler type')
    parser.add_argument('--log_id', type=str, default='')
    args = parser.parse_args()

    if args.nb_examples > 8400:
        print('nb_examples must be lower than or equal to 8400')
        sys.exit(1)

    if args.hf_token is None:
        print('HuggingFace token not provided, please provide it using --hf_token')
        sys.exit(1)

    seed_everything(args.seed)
    source_dataset = load_h_data(args)
    target_dataset_fr = load_data('data/moral/test_fr', args)
    target_dataset_en = load_data('data/moral/test_en', args)

    save_path = 'moral_experiment_' + args.ref_model
    model, tokenizer = qlora_training_reg_gdist(args, source_dataset, save_path)
    print('\n------ End training ------\n')
    evaluate_model(model, tokenizer, {'fr':target_dataset_fr, 'en':target_dataset_en}, args, save_path)
