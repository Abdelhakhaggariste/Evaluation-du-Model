import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Importations
from env_iot_mec import IoTMECEnv
from models import ActorCritic
from ppo_agent import PPOAgent

# Configuration
N_IOT = 10
STATE_DIM = 6  # Nombre de caractéristiques par IoT
INPUT_DIM = N_IOT * STATE_DIM  # Dimension totale de l'entrée
ACTION_DIM = 2  # 0=local, 1=MEC
EPISODES = 200
STEPS_PER_EPISODE = 20

print("="*60)
print("ENTRAÎNEMENT DRL POUR IoT-MEC")
print("="*60)
print(f"Configuration:")
print(f"  • IoT: {N_IOT}")
print(f"  • État par IoT: {STATE_DIM} features")
print(f"  • Dimension entrée: {INPUT_DIM}")
print(f"  • Actions: {ACTION_DIM} (0=local, 1=MEC)")
print(f"  • Épisodes: {EPISODES}")
print(f"  • Steps par épisode: {STEPS_PER_EPISODE}")
print("="*60)

# Initialisation
print("\nInitialisation de l'environnement...")
env = IoTMECEnv(n_iot=N_IOT)
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.nvec}")

print("\nInitialisation du modèle...")
model = ActorCritic(
    input_dim=INPUT_DIM,
    action_dim=ACTION_DIM,
    n_iot=N_IOT
)
print(f"  Modèle créé avec succès")
print(f"  Paramètres totaux: {sum(p.numel() for p in model.parameters()):,}")

print("\nInitialisation de l'agent PPO...")
agent = PPOAgent(
    model=model,
    n_iot=N_IOT,
    lr=3e-4,
    gamma=0.99,
    eps_clip=0.2
)

# Historique des métriques
metrics_history = defaultdict(list)

print("\n" + "="*60)
print("DÉBUT DE L'ENTRAÎNEMENT")
print("="*60)

for episode in range(EPISODES):
    # Réinitialiser l'environnement
    state, _ = env.reset()
    episode_rewards = []
    episode_energies = []
    episode_latencies = []
    episode_successes = []
    episode_offloads = []
    
    for step in range(STEPS_PER_EPISODE):
        # Sélectionner les actions
        actions, log_probs, value = agent.select_action(state)
        
        # Exécuter les actions
        next_state, reward, terminated, truncated, info = env.step(actions.numpy())
        
        # Stocker la transition
        agent.store_transition(state, actions, log_probs, value, reward, terminated)
        
        # Collecter les métriques
        episode_rewards.append(reward)
        episode_energies.append(info['energy'])
        episode_latencies.append(info['latency'])
        episode_successes.append(info['success_rate'])
        episode_offloads.append(info['offload_rate'])
        
        # Mettre à jour l'état
        state = next_state
        
        # Si c'est la fin de l'épisode, mettre à jour l'agent
        if terminated or step == STEPS_PER_EPISODE - 1:
            loss = agent.update()
            break
    
    # Calculer les moyennes pour l'épisode
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_energy = np.mean(episode_energies) if episode_energies else 0
    avg_latency = np.mean(episode_latencies) if episode_latencies else 0
    avg_success = np.mean(episode_successes) if episode_successes else 0
    avg_offload = np.mean(episode_offloads) if episode_offloads else 0
    
    # Stocker dans l'historique
    metrics_history['episode'].append(episode)
    metrics_history['total_reward'].append(avg_reward)
    metrics_history['avg_energy'].append(avg_energy)
    metrics_history['avg_latency'].append(avg_latency)
    metrics_history['success_rate'].append(avg_success)
    metrics_history['offload_rate'].append(avg_offload)
    metrics_history['loss'].append(loss if 'loss' in locals() else 0)
    
    # Afficher les progrès
    if (episode + 1) % 20 == 0 or episode == 0:
        print(f"Épisode {episode + 1:3d}/{EPISODES} | "
              f"Récompense: {avg_reward:7.4f} | "
              f"Énergie: {avg_energy:6.3f} J | "
              f"Latence: {avg_latency:6.1f} ms | "
              f"Succès: {avg_success:6.2%} | "
              f"Déchargement: {avg_offload:6.2%}")

print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ")
print("="*60)

# Sauvegarde des métriques
print("\nSauvegarde des métriques...")
df_metrics = pd.DataFrame(metrics_history)
df_metrics.to_csv('training_metrics.csv', index=False)
print(f"  Métriques sauvegardées dans 'training_metrics.csv'")

# Sauvegarde du modèle
print("\nSauvegarde du modèle...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
    'metrics': metrics_history
}, 'ppo_model.pth')
print(f"  Modèle sauvegardé dans 'ppo_model.pth'")

# 1. VISUALISATION PRINCIPALE
print("\nGénération des visualisations...")
plt.figure(figsize=(16, 10))

# 1.1 Récompense
plt.subplot(2, 3, 1)
plt.plot(metrics_history['episode'], metrics_history['total_reward'], 
         'b-', linewidth=2, alpha=0.8)
plt.fill_between(metrics_history['episode'], 
                 np.array(metrics_history['total_reward']) * 0.95,
                 np.array(metrics_history['total_reward']) * 1.05,
                 alpha=0.2, color='blue')
plt.xlabel('Épisodes')
plt.ylabel('Récompense Moyenne')
plt.title('Évolution de la Récompense')
plt.grid(True, alpha=0.3)

# 1.2 Énergie
plt.subplot(2, 3, 2)
plt.plot(metrics_history['episode'], metrics_history['avg_energy'], 
         'r-', linewidth=2, alpha=0.8)
plt.xlabel('Épisodes')
plt.ylabel('Énergie Moyenne (J)')
plt.title('Consommation Énergétique')
plt.grid(True, alpha=0.3)

# 1.3 Latence
plt.subplot(2, 3, 3)
plt.plot(metrics_history['episode'], metrics_history['avg_latency'], 
         'g-', linewidth=2, alpha=0.8)
plt.xlabel('Épisodes')
plt.ylabel('Latence Moyenne (ms)')
plt.title('Performance de Latence')
plt.grid(True, alpha=0.3)

# 1.4 Taux de succès
plt.subplot(2, 3, 4)
plt.plot(metrics_history['episode'], metrics_history['success_rate'], 
         'c-', linewidth=2, alpha=0.8)
plt.xlabel('Épisodes')
plt.ylabel('Taux de Succès')
plt.title('Fiabilité des Tâches')
plt.grid(True, alpha=0.3)

# 1.5 Taux de déchargement
plt.subplot(2, 3, 5)
plt.plot(metrics_history['episode'], metrics_history['offload_rate'], 
         'm-', linewidth=2, alpha=0.8)
plt.xlabel('Épisodes')
plt.ylabel('Taux de Déchargement')
plt.title('Stratégie de Déchargement')
plt.grid(True, alpha=0.3)

# 1.6 Loss
plt.subplot(2, 3, 6)
plt.plot(metrics_history['episode'], metrics_history['loss'], 
         'k-', linewidth=2, alpha=0.8)
plt.xlabel('Épisodes')
plt.ylabel('Valeur de Loss')
plt.title('Convergence de l\'Apprentissage')
plt.grid(True, alpha=0.3)

plt.suptitle('Analyse Complète des Performances DRL IoT-MEC', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('multi_metrics_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ multi_metrics_analysis.png")

# 2. VISUALISATION COMBINÉE
plt.figure(figsize=(14, 8))

# Normaliser les métriques
episodes = metrics_history['episode']
reward_norm = (np.array(metrics_history['total_reward']) - np.min(metrics_history['total_reward'])) / \
              (np.max(metrics_history['total_reward']) - np.min(metrics_history['total_reward']))
energy_norm = 1 - ((np.array(metrics_history['avg_energy']) - np.min(metrics_history['avg_energy'])) / \
                  (np.max(metrics_history['avg_energy']) - np.min(metrics_history['avg_energy'])))
latency_norm = 1 - ((np.array(metrics_history['avg_latency']) - np.min(metrics_history['avg_latency'])) / \
                   (np.max(metrics_history['avg_latency']) - np.min(metrics_history['avg_latency'])))

# Toutes les métriques sur un même graphique
plt.plot(episodes, reward_norm, 'b-', linewidth=2, label='Récompense (normalisée)')
plt.plot(episodes, energy_norm, 'r-', linewidth=2, label='Énergie (inversée)')
plt.plot(episodes, latency_norm, 'g-', linewidth=2, label='Latence (inversée)')
plt.plot(episodes, metrics_history['success_rate'], 'c-', linewidth=2, label='Taux de succès')
plt.plot(episodes, metrics_history['offload_rate'], 'm-', linewidth=2, label='Taux de déchargement')

plt.xlabel('Épisodes d\'Apprentissage')
plt.ylabel('Valeur Normalisée')
plt.title('Évolution Conjointe de Toutes les Métriques IoT-MEC', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', ncol=2)
plt.grid(True, alpha=0.3)

# Zones d'apprentissage
plt.axvspan(0, 50, alpha=0.1, color='red', label='Phase d\'exploration')
plt.axvspan(50, 150, alpha=0.1, color='yellow', label='Phase d\'apprentissage')
plt.axvspan(150, 200, alpha=0.1, color='green', label='Phase de convergence')

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig('combined_metrics_evolution.png', dpi=300, bbox_inches='tight')
print("  ✓ combined_metrics_evolution.png")

# 3. VISUALISATION AVANCÉE - COMPARAISON
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 Énergie vs Latence (trade-off)
scatter = axes[0, 0].scatter(metrics_history['avg_energy'], 
                             metrics_history['avg_latency'],
                             c=metrics_history['episode'], 
                             cmap='viridis', 
                             alpha=0.7,
                             s=50)
axes[0, 0].set_xlabel('Énergie (J)')
axes[0, 0].set_ylabel('Latence (ms)')
axes[0, 0].set_title('Compromis Énergie-Latence')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 0], label='Épisode')

# 3.2 Récompense vs Déchargement
axes[0, 1].scatter(metrics_history['offload_rate'], 
                   metrics_history['total_reward'],
                   c=metrics_history['episode'],
                   cmap='plasma',
                   alpha=0.7,
                   s=50)
axes[0, 1].set_xlabel('Taux de Déchargement')
axes[0, 1].set_ylabel('Récompense')
axes[0, 1].set_title('Impact du Déchargement sur la Performance')
axes[0, 1].grid(True, alpha=0.3)

# 3.3 Succès vs Énergie
axes[1, 0].scatter(metrics_history['avg_energy'], 
                   metrics_history['success_rate'],
                   c=metrics_history['episode'],
                   cmap='cool',
                   alpha=0.7,
                   s=50)
axes[1, 0].set_xlabel('Énergie (J)')
axes[1, 0].set_ylabel('Taux de Succès')
axes[1, 0].set_title('Relation Énergie-Fiabilité')
axes[1, 0].grid(True, alpha=0.3)

# 3.4 Distribution finale
metrics_final = ['Récompense', 'Énergie', 'Latence', 'Succès', 'Déchargement']
values_final = [
    metrics_history['total_reward'][-1],
    metrics_history['avg_energy'][-1],
    metrics_history['avg_latency'][-1],
    metrics_history['success_rate'][-1],
    metrics_history['offload_rate'][-1]
]

# Normaliser pour le radar chart
values_normalized = [
    (metrics_history['total_reward'][-1] - min(metrics_history['total_reward'])) / 
    (max(metrics_history['total_reward']) - min(metrics_history['total_reward'])),
    1 - (metrics_history['avg_energy'][-1] - min(metrics_history['avg_energy'])) / 
    (max(metrics_history['avg_energy']) - min(metrics_history['avg_energy'])),
    1 - (metrics_history['avg_latency'][-1] - min(metrics_history['avg_latency'])) / 
    (max(metrics_history['avg_latency']) - min(metrics_history['avg_latency'])),
    metrics_history['success_rate'][-1],
    metrics_history['offload_rate'][-1]
]

bars = axes[1, 1].bar(metrics_final, values_normalized, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                      alpha=0.8)
axes[1, 1].set_ylabel('Valeur Normalisée')
axes[1, 1].set_title('Performance Finale (Dernier Épisode)')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs
for bar, value, norm in zip(bars, values_final, values_normalized):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{norm:.2f}\n({value:.3f})', 
                   ha='center', va='bottom', fontsize=8)

plt.suptitle('Analyse Comparative et Trade-offs DRL IoT-MEC', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ comparative_analysis.png")

# 4. RAPPORT FINAL
print("\n" + "="*60)
print("RAPPORT FINAL D'ENTRAÎNEMENT")
print("="*60)

# Calcul des améliorations
initial_metrics = {
    'reward': metrics_history['total_reward'][0],
    'energy': metrics_history['avg_energy'][0],
    'latency': metrics_history['avg_latency'][0],
    'success': metrics_history['success_rate'][0],
    'offload': metrics_history['offload_rate'][0]
}

final_metrics = {
    'reward': metrics_history['total_reward'][-1],
    'energy': metrics_history['avg_energy'][-1],
    'latency': metrics_history['avg_latency'][-1],
    'success': metrics_history['success_rate'][-1],
    'offload': metrics_history['offload_rate'][-1]
}

# Calcul des améliorations en pourcentage
improvements = {}
for key in initial_metrics.keys():
    if key == 'energy' or key == 'latency':
        # Pour l'énergie et la latence, on veut une réduction (valeur plus basse)
        improvements[key] = ((initial_metrics[key] - final_metrics[key]) / initial_metrics[key]) * 100
    else:
        # Pour les autres, on veut une augmentation (valeur plus haute)
        improvements[key] = ((final_metrics[key] - initial_metrics[key]) / initial_metrics[key]) * 100

print(f"\n🎯 AMÉLIORATIONS APRÈS {EPISODES} ÉPISODES:")
print("-" * 50)
print(f"📊 RÉCOMPENSE:      {initial_metrics['reward']:.4f} → {final_metrics['reward']:.4f} ({improvements['reward']:+.1f}%)")
print(f"⚡ ÉNERGIE:         {initial_metrics['energy']:.3f} J → {final_metrics['energy']:.3f} J ({improvements['energy']:+.1f}%)")
print(f"⏱️  LATENCE:         {initial_metrics['latency']:.1f} ms → {final_metrics['latency']:.1f} ms ({improvements['latency']:+.1f}%)")
print(f"✅ SUCCÈS:          {initial_metrics['success']:.2%} → {final_metrics['success']:.2%} ({improvements['success']:+.1f}%)")
print(f"🔄 DÉCHARGEMENT:    {initial_metrics['offload']:.2%} → {final_metrics['offload']:.2%} ({improvements['offload']:+.1f}%)")

# Statistiques supplémentaires
print(f"\n📈 STATISTIQUES SUPPLÉMENTAIRES:")
print("-" * 50)
print(f"  • Récompense moyenne finale: {np.mean(metrics_history['total_reward'][-10:]):.4f}")
print(f"  • Énergie moyenne finale: {np.mean(metrics_history['avg_energy'][-10:]):.3f} J")
print(f"  • Latence moyenne finale: {np.mean(metrics_history['avg_latency'][-10:]):.1f} ms")
print(f"  • Succès moyen final: {np.mean(metrics_history['success_rate'][-10:]):.2%}")
print(f"  • Déchargement moyen final: {np.mean(metrics_history['offload_rate'][-10:]):.2%}")

# Corrélations
print(f"\n🔗 CORRÉLATIONS IMPORTANTES:")
print("-" * 50)
correlations = df_metrics[['total_reward', 'avg_energy', 'avg_latency', 'success_rate', 'offload_rate']].corr()
print(f"  • Récompense ↔ Énergie: {correlations.loc['total_reward', 'avg_energy']:.3f}")
print(f"  • Récompense ↔ Latence: {correlations.loc['total_reward', 'avg_latency']:.3f}")
print(f"  • Énergie ↔ Latence: {correlations.loc['avg_energy', 'avg_latency']:.3f}")
print(f"  • Déchargement ↔ Succès: {correlations.loc['offload_rate', 'success_rate']:.3f}")

print("\n" + "="*60)
print("🎉 VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS!")
print("="*60)
print("\n📁 FICHIERS CRÉÉS:")
print("  • training_metrics.csv - Données brutes de l'entraînement")
print("  • ppo_model.pth - Modèle PPO entraîné")
print("  • multi_metrics_analysis.png - 6 métriques individuelles")
print("  • combined_metrics_evolution.png - Toutes les métriques combinées")
print("  • comparative_analysis.png - Analyses comparatives et trade-offs")

print("\n✅ L'ENTRAÎNEMENT ET L'ANALYSE SONT TERMINÉS AVEC SUCCÈS!")
print("="*60)

# Afficher les graphiques
plt.show()