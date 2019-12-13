import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import Discriminator, DiscriminatorT

class DiscOptimizer(object):
    def __init__(self, num_inputs, args):
        self.tau = args.tau
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.disc = Discriminator(num_inputs, args.hidden_size, args.latent_size).to(device=self.device)
        self.disc_optim = Adam(self.disc.parameters(), lr=args.dclr)

        self.disc_target = Discriminator(num_inputs, args.hidden_size, args.latent_size).to(device=self.device)
        hard_update(self.disc_target, self.disc)        

    def pseudo_score(self, context, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        context = torch.FloatTensor(context).to(self.device).unsqueeze(0)
        mu = self.disc_target(state)
        score = - F.mse_loss(mu, context, reduction='sum')
        # score = - torch.abs(context - mu) # No significant improvement
        
        return score.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        context_batch, state_batch, _, _, _, _ = memory.sample(batch_size=batch_size)
        context_batch = torch.FloatTensor(context_batch).to(self.device)
        state_batch = torch.FloatTensor(state_batch).to(self.device)

        prediction = self.disc(state_batch)
        disc_loss = F.mse_loss(prediction, context_batch)
        # disc_loss = F.l1_loss(prediction, context_batch) # No significant improvement

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.disc_target, self.disc, self.tau)

        return disc_loss.item()

    def save_model(self, env_name, suffix="", disc_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if disc_path is None:
            disc_path = "models/disc_{}_{}".format(env_name, suffix) 

        print('Saving discriminator models to {}'.format(disc_path))
        torch.save(self.disc.state_dict(), disc_path)

    def load_model(self, disc_path=None, env_name=None, suffix=""):
        print('Loading discriminator models from {}'.format(disc_path))
        if disc_path is not None:
            self.disc.load_state_dict(torch.load(disc_path))
        elif env_name is not None:
            disc_path = "models/disc_{}_{}".format(env_name, suffix) 
            self.disc.load_state_dict(torch.load(disc_path))