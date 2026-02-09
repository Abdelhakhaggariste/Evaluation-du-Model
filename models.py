import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, n_iot=10):
        """
        Args:
            input_dim: Dimension de l'entrée (n_iot * state_dim)
            action_dim: Dimension de l'action pour chaque IoT (2: local/MEC)
            n_iot: Nombre de dispositifs IoT
        """
        super(ActorCritic, self).__init__()
        
        self.n_iot = n_iot
        self.input_dim = input_dim
        
        # Réseau partagé pour l'extraction de caractéristiques
        # Calculez les dimensions intermédiaires
        hidden1 = 256
        hidden2 = 128
        
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, 64),
            nn.ReLU()
        )
        
        # Tête d'acteur pour produire les logits pour chaque IoT
        # Sortie: n_iot * action_dim
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_iot * action_dim)
        )
        
        # Tête de critique (valeur d'état)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialise les poids du réseau"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor de forme (batch_size, input_dim)
        
        Returns:
            actor_output: Tensor de forme (batch_size, n_iot, action_dim)
            critic_output: Tensor de forme (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Vérifier la dimension d'entrée
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Dimension d'entrée incorrecte: attendu {self.input_dim}, reçu {x.shape[1]}")
        
        # Passage par le réseau partagé
        features = self.shared_network(x)
        
        # Acteur: produire les logits pour chaque IoT
        actor_output = self.actor(features)
        # Reshape en (batch_size, n_iot, action_dim)
        actor_output = actor_output.view(batch_size, self.n_iot, -1)
        
        # Critique: valeur d'état
        critic_output = self.critic(features)
        
        return actor_output, critic_output

class LSTMPredictor(nn.Module):
    """Prédicteur LSTM pour les séquences temporelles"""
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1, num_layers=2):
        super(LSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor de forme (batch_size, seq_len, input_dim)
        """
        # Passage LSTM
        lstm_out, _ = self.lstm(x)
        
        # Prendre la dernière sortie de la séquence
        last_output = lstm_out[:, -1, :]
        
        # Couche fully connected
        output = self.fc(last_output)
        
        return output