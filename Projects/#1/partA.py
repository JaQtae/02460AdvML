from src.DeepGenerativeModels.Week1.vae_bernoulli_mine import (
    VAE,
    BernoulliDecoder,
    GaussianEncoder,
    GaussianPrior,
    MoGPrior,
    train as train_vae,
    evaluate,
) 

from src.DeepGenerativeModels.Week2.flow_mine import (
    Flow,
)

from .utils import (
    _get_decoder,
    _get_encoder,
    _get_mask_tranformations,
    _get_binarized_mnist,
)

import torch
import logging
import argparse
import os

logger = logging.getLogger()


if __name__ == "__main__":
    # TODO: Make sure it's working, might need args for the flow part?
    # TODO: Add the if args.mode == 'train': ...
    # TODO: Figure out where the standard prior is // MoG // Flow-based and make seamless integration
    
    # Define the path to the folder where the current script file is.
    dir_name = os.path.dirname(os.path.abspath(__file__)) + '/'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--prior_type', type=str, default='sg', choices=['sg', 'mog', 'flow'], help='choice of prior (choices: %(choices)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    
    # Loading binarized MNIST with given batch_size.
    mnist_train_loader, mnist_test_loader = _get_binarized_mnist(path=dir_name,
                                                              batch_size=args.batch_size)
    # Define prior distribution
    M = args.latent_dim
    
    # Define the encoder and decoder networks
    encoder_net = _get_encoder(M)
    decoder_net = _get_decoder(M)
    
    # Choose which prior to use in the VAE.
    if args.prior_type == 'sg':
        prior = GaussianPrior(M)
        
    elif args.prior_type == 'mog':
        prior = MoGPrior(M, args.batch_size, args.device)
        
    elif args.prior_type == 'flow':
        # Define prior distribution
        base, transformations = _get_mask_tranformations(M)
        prior = Flow(base, transformations).to(device)
        

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)


    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("Starting training model...")
        
        train_vae(model, optimizer, mnist_train_loader, args.epochs, args.device)
        
        logger.info(f"Saving model with name: {args.model}")
        torch.save(model.state_dict(), dir_name+args.model)
        
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        
        logger.info(f"Test loss: {evaluate(model, mnist_test_loader)}")
        