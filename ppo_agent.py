import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, model, n_iot=10, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        """
        Agent PPO pour IoT-MEC
        
        Args:
            model: Modèle ActorCritic
            n_iot: Nombre de dispositifs IoT
            lr: Taux d'apprentissage
            gamma: Facteur de discount
            eps_clip: Paramètre de clipping PPO
            value_coef: Coefficient pour la perte de valeur
            entropy_coef: Coefficient pour l'entropie
        """
        self.model = model
        self.n_iot = n_iot
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Buffer pour stocker les transitions
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """
        Sélectionne une action pour chaque IoT
        
        Args:
            state: État courant (vecteur aplati)
        
        Returns:
            actions: Liste d'actions pour chaque IoT
            log_probs: Log probabilités des actions
            value: Valeur estimée de l'état
        """
        # Convertir en tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Ajouter dimension batch
        else:
            state_tensor = state
        
        # Forward pass
        with torch.no_grad():
            actor_output, value = self.model(state_tensor)
        
        # actor_output shape: (1, n_iot, action_dim)
        actor_output = actor_output.squeeze(0)  # (n_iot, action_dim)
        
        # Sélectionner les actions pour chaque IoT
        actions = []
        log_probs = []
        
        for i in range(self.n_iot):
            # Distribution de probabilités pour cet IoT
            dist = Categorical(logits=actor_output[i])
            
            # Échantillonner une action
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions.append(action.item())
            log_probs.append(log_prob)
        
        # Convertir en tensors
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        log_probs_tensor = torch.stack(log_probs)
        
        return actions_tensor, log_probs_tensor, value.squeeze()
    
    def store_transition(self, state, actions, log_probs, value, reward, done):
        """Stocke une transition dans le buffer"""
        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        """Vide le buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def compute_returns_and_advantages(self, next_value):
        """
        Calcule les retours et avantages
        
        Args:
            next_value: Valeur du prochain état
        
        Returns:
            returns: Retours actualisés
            advantages: Avantages
        """
        returns = []
        advantages = []
        R = next_value
        
        # Calculer les retours (Monte Carlo)
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Calculer les avantages
        values = torch.tensor(self.values, dtype=torch.float32)
        advantages = returns - values
        
        # Normaliser les avantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self):
        """Met à jour le modèle avec PPO"""
        if len(self.states) == 0:
            return 0.0
        
        # Convertir les listes en tensors
        states_tensor = torch.stack([torch.FloatTensor(s) for s in self.states])
        actions_tensor = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        
        # Calculer les retours et avantages
        with torch.no_grad():
            _, next_value = self.model(torch.FloatTensor(self.states[-1]).unsqueeze(0))
            next_value = next_value.squeeze().item()
        
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # Optimisation pour plusieurs époques
        losses = []
        
        for epoch in range(4):  # 4 époques
            # Forward pass
            actor_output, values = self.model(states_tensor)
            
            # Calculer les nouvelles probabilités
            new_log_probs = []
            entropies = []
            
            for i in range(self.n_iot):
                # Distribution pour chaque IoT
                dist = Categorical(logits=actor_output[:, i, :])
                
                # Log probabilités pour les actions prises
                new_log_prob = dist.log_prob(actions_tensor[:, i])
                entropy = dist.entropy()
                
                new_log_probs.append(new_log_prob)
                entropies.append(entropy)
            
            # Stack les résultats
            new_log_probs = torch.stack(new_log_probs, dim=1)  # (batch, n_iot)
            entropies = torch.stack(entropies, dim=1)  # (batch, n_iot)
            
            # Ratio des probabilités
            ratios = torch.exp(new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1).detach())
            
            # Perte PPO
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Perte de valeur
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Perte d'entropie
            entropy_loss = -entropies.mean()
            
            # Perte totale
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip des gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            # Optimisation
            self.optimizer.step()
            
            losses.append(total_loss.item())
        
        # Vider le buffer
        self.clear_memory()
        
        return np.mean(losses)