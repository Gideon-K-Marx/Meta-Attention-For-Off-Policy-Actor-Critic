import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Hot_Plug
from TD3_model import Actor, Actor_Feature, Critic, Meta_Critic, Meta_Attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta-Critic of Twin Delayed Deep Deterministic Policy Gradients (TD3_MC)
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        self.args = args
        # Action Net
        self.actor = Actor(action_dim, max_action).to(device)
        self.actor_target = Actor(action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        # Actor Feature Net
        self.actor_feature = Actor_Feature(state_dim).to(device)
        self.actor_feature_target = Actor_Feature(state_dim).to(device)
        self.actor_feature_target.load_state_dict(self.actor_feature.state_dict())
        self.actor_feature_optimizer = torch.optim.Adam(self.actor_feature.parameters(), lr=args.actor_lr)

        # Critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Meta critic in Online Meta Critic Learning For Off-Policy Actor-Critic Methods in NIPS-2020
        self.meta_critic = Meta_Critic(128).to(device)
        self.meta_critic_optim = torch.optim.Adam(self.meta_critic.parameters(), lr=args.aux_lr,
                                                  weight_decay=args.weight_decay)
        # Meta Attention
        self.Meta_Attention = Meta_Attention(256).to(device)
        self.Meta_Attention_optim = torch.optim.Adam(self.Meta_Attention.parameters(), lr=args.aux_lr,
                                                 weight_decay=args.weight_decay)

        feature_net = nn.Sequential(*list(self.actor_feature.children())[:-1])
        self.hotplug = Hot_Plug(feature_net)
        self.lr_actor = args.actor_lr
        self.max_action = max_action
        self.loss_store = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actor_feature = self.actor_feature(state)
        action = self.actor(actor_feature)

        if(self.args.method == 'TD3_MAT4'):
            critic_feature = self.critic.features(state, self.actor(actor_feature))
            source = torch.cat([actor_feature, critic_feature], dim=1)
            attention = self.Meta_Attention(source)
            attention_action = self.actor(torch.mul(actor_feature, attention))

            attention_action_Q = self.critic.Q1(state, attention_action).cpu().data.numpy()
            action_Q = self.critic.Q1(state, action).cpu().data.numpy()

            if attention_action_Q > action_Q :
                return attention_action.cpu().data.numpy().flatten()
            else:
                return action.cpu().data.numpy().flatten()

        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, training_step, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        done = torch.FloatTensor(1 - d).to(device)
        reward = torch.FloatTensor(r).to(device)

        # Sample replay buffer for meta test
        x_val, _, _, _, _ = replay_buffer.sample(batch_size)
        state_val = torch.FloatTensor(x_val).to(device)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_state_feature = self.actor_feature_target(next_state)
        next_action = (self.actor_target(next_state_feature) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * discount * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if training_step % policy_freq == 0:
            state_feature = self.actor_feature(state)
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state_feature)).mean()
            mc_loss_auxiliary = self.meta_critic(state_feature)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.actor_feature_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor_feature.parameters(), self.args.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)

            # Meta Critic Update
            if (self.args.method == 'TD3_MC'):
                self.hotplug.update(self.lr_actor)
                state_feature_val = self.actor_feature(state_val)
                policy_loss_val_mc = self.critic.Q1(state_val, self.actor(state_feature_val))
                policy_loss_val_mc = -policy_loss_val_mc.mean()
                policy_loss_val_mc = policy_loss_val_mc

                mc_loss_auxiliary.backward(create_graph=True)
                nn.utils.clip_grad_norm_(self.actor_feature.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.hotplug.update(self.lr_actor)
                state_feature_val_new = self.actor_feature(state_val)
                policy_loss_val_mc_new = self.critic.Q1(state_val, self.actor(state_feature_val_new))
                policy_loss_val_mc_new = -policy_loss_val_mc_new.mean()
                policy_loss_val_mc_new = policy_loss_val_mc_new

                meta_critic_utility = policy_loss_val_mc - policy_loss_val_mc_new
                meta_critic_utility = torch.tanh(meta_critic_utility)
                loss_meta_critc = -meta_critic_utility

                # Meta optimization of auxilary network
                self.meta_critic_optim.zero_grad()
                grad_omega = torch.autograd.grad(loss_meta_critc, self.meta_critic.parameters(), retain_graph=True)
                nn.utils.clip_grad_norm_(self.meta_critic.parameters(), self.args.max_grad_norm)
                for gradient, variable in zip(grad_omega, self.meta_critic.parameters()):
                    variable.grad.data = gradient
                self.meta_critic_optim.step()

            # Meta Attention Update
            if ((self.args.method == 'TD3_MATT') or (self.args.method == 'TD3_MAT4')):
                self.hotplug.update(self.lr_actor)
                actor_features = self.actor_feature(state_val)
                policy_loss = self.critic.Q1(state_val, self.actor(actor_features))
                policy_loss = -policy_loss.mean()
                policy_loss = policy_loss

                critic_features = self.critic.features(state_val, self.actor(actor_features))
                source = torch.cat([actor_features, critic_features], dim=1)
                attention = self.Meta_Attention(source)
                attention_action = self.actor(torch.mul(actor_features, attention))

                policy_loss_ma = self.critic.Q1(state_val, attention_action)
                policy_loss_ma = -policy_loss_ma.mean()
                policy_loss_ma = policy_loss_ma
                policy_loss_ma.backward(create_graph=True)
                self.hotplug.update(self.lr_actor)

                actor_features_new = self.actor_feature(state_val)
                policy_loss_new = self.critic.Q1(state_val, self.actor(actor_features_new))
                policy_loss_new = -policy_loss_new.mean()
                policy_loss_new = policy_loss_new

                Meta_Attention_utility = policy_loss - policy_loss_new
                Meta_Attention_utility = torch.tanh(Meta_Attention_utility)
                loss_Meta_Attention = -Meta_Attention_utility

                self.Meta_Attention_optim.zero_grad()
                grad_psi = torch.autograd.grad(loss_Meta_Attention, self.Meta_Attention.parameters())
                nn.utils.clip_grad_norm_(self.Meta_Attention.parameters(), self.args.max_grad_norm)
                for gradient, variable in zip(grad_psi, self.Meta_Attention.parameters()):
                    variable.grad.data = gradient
                self.Meta_Attention_optim.step()

            nn.utils.clip_grad_norm_(self.actor_feature.parameters(), self.args.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_feature_optimizer.step()
            self.hotplug.restore()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor_feature.parameters(), self.actor_feature_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor_feature.state_dict(), '%s/%s_actor_feature.pth' % (directory, filename))
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor_feature.load_state_dict(torch.load('%s/%s_actor_feature.pth' % (directory, filename)))
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
