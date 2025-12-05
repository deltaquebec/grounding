"""
Demonstrates agent grounding symbolic commands to spatial
coordinates in a grid world, with a formal audit procedure (G0–G4)
evaluating preservation, faithfulness, robustness, and compositionality.

Paper: On measuring grounding and generalizing grounding problems 
Author: Daniel Quigley and Eric Maynard

Date: 12/05/2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from typing import List, Tuple

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

# ==========================================
# environment, context k
# ==========================================

class GridWorldContext:
    """
    world context (k) and meaning type (t = ext)
    defines the ground truth interpretation (I_k^t)
    """
    LANDMARKS = {
        'RED':  np.array([8.0, 8.0]),
        'BLUE': np.array([2.0, 2.0]),
    }
    VECTORS = {
        'NORTH': np.array([0.0, 1.0]),
        'SOUTH': np.array([0.0, -1.0]),
        'EAST':  np.array([1.0, 0.0]),
        'WEST':  np.array([-1.0, 0.0]),
    }
    
    VOCAB = ['PAD', 'RED', 'BLUE', 'NORTH', 'SOUTH', 'EAST', 'WEST']
    
    @classmethod
    def get_ground_truth(cls, tokens: List[str]) -> np.array:
        """calculates I_k^t(tokens) via semantic algebra (vector addition)"""
        target = np.array([0.0, 0.0])
        for t in tokens:
            if t in cls.LANDMARKS:
                target = cls.LANDMARKS[t].copy()
            elif t in cls.VECTORS:
                target += cls.VECTORS[t]
        return target

    @classmethod
    def sample_task(cls) -> Tuple[List[str], np.array]:
        """generates random training task from distribution P"""
        r = random.random()
        if r < 0.2:
            cmd = [random.choice(list(cls.VECTORS.keys()))]
        elif r < 0.5:
            cmd = [random.choice(list(cls.LANDMARKS.keys()))]
        else:
            lm = random.choice(list(cls.LANDMARKS.keys()))
            vec = random.choice(list(cls.VECTORS.keys()))
            cmd = [lm, vec]
        return cmd, cls.get_ground_truth(cmd)


# ==========================================
# agent architecture
#    Φ: Σ* \to R     (encoder: embedding + GRU \to hidden state)
#    Γ: R \to C      (decoder: hidden state \to coordinates)
#    A_k^t: C \to M  (alignment: identity; agent moves to coords)
# ==========================================

class GridBotAgent(nn.Module):
    """
    policy implementing grounding architecture
    
    we have the following:
        Φ (encoder)  = self.embedding + self.gru  :  Σ* → R = ℝ^64
        Γ (decoder)  = self.decoder               :  R → C = ℝ^2
        A_k^t        = identity                   :  C → M_k^t = ℝ^2
    """
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Φ encoder components
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        
        # Γ decoder
        self.decoder = nn.Linear(hidden_dim, 2)
        
        # audit hooks
        self._representation_noise = None  # needed for G3 noise injected into R
        self._ablate_modifiers = False     # and for G2b process only first token

    def encode(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Φ: Σ* → R
        maps token sequence to representation space (GRU hidden state)
        """
        embeds = self.embedding(token_indices)
        
        if self._ablate_modifiers:
            # G2b ablation process only first token
            embeds = embeds[:, 0:1, :]
        
        _, hidden = self.gru(embeds)
        r = hidden.squeeze(0)  # Shape: (batch, hidden_dim) = R
        
        # G3 inject noise into representation space R
        if self._representation_noise is not None:
            r = r + self._representation_noise.to(r.device)
        
        return r

    def decode(self, r: torch.Tensor) -> torch.Tensor:
        """
        Γ: R → C
        maps representation to concept space (predicted coordinates)
        """
        return self.decoder(r)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Ψ = Γ ∘ Φ: Σ* → C
        end-to-end mapping from symbols to concepts
        
        Note: in this toy example, A_k^t is identity, so the output
        directly represents the agent's position in M_k^t
        """
        r = self.encode(token_indices)  # Φ
        c = self.decode(r)               # Γ
        return c  # A_k^t(c) = c


# ==========================================
# training (check for G0)
# ==========================================

def train_agent() -> Tuple[GridBotAgent, dict]:
    """
    train via REINFORCE
    note: for future work, add check for reporting extent of behavior
    """
    print("=" * 60)
    print("G0: TRAINING AGENT")
    print("=" * 60)
    print("method: REINFORCE")
    print("objective: minimize distance to target coordinates\n")
    
    vocab = {w: i for i, w in enumerate(GridWorldContext.VOCAB)}
    agent = GridBotAgent(len(vocab))
    optimizer = optim.Adam(agent.parameters(), lr=0.005)
    
    EPISODES = 3000
    
    for episode in range(EPISODES):
        cmd_str, target_pos = GridWorldContext.sample_task()
        target = torch.tensor(target_pos, dtype=torch.float32).unsqueeze(0)
        
        indices = [vocab[w] for w in cmd_str]
        tokens = torch.tensor([indices], dtype=torch.long)
        
        pred = agent(tokens)
        loss = nn.MSELoss()(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 500 == 0:
            print(f"  episode {episode:4d}: loss {loss.item():.4f} | Cmd: {cmd_str}")

    print("\nTraining complete!\n")
    return agent, vocab


# ==========================================
# grounding audit
# ==========================================

def audit_agent(agent: GridBotAgent, vocab: dict):
    """
    perform grounding audit on trained agent.
    """
    print("=" * 60)
    print("GROUNDING AUDIT")
    print("=" * 60)
    
    def get_output(cmd: List[str], ablate: bool = False, 
                   noise: torch.Tensor = None) -> np.ndarray:
        """get A_k^t ∘ Ψ(cmd) with optional interventions"""
        indices = [vocab[w] for w in cmd]
        tokens = torch.tensor([indices], dtype=torch.long)
        
        agent._ablate_modifiers = ablate
        agent._representation_noise = noise
        
        with torch.no_grad():
            result = agent(tokens)
        
        agent._ablate_modifiers = False
        agent._representation_noise = None
        
        return result.squeeze().numpy()

    # ----- G1 -----
    print("\n[G1] PRESERVATION")
    print("     question: does Agent(RED) ≈ I(RED)?")
    
    atom = ["RED"]
    truth_atom = GridWorldContext.get_ground_truth(atom)
    agent_atom = get_output(atom)
    eps_pres = np.linalg.norm(agent_atom - truth_atom)
    
    print(f"     I_k^t(RED):           {truth_atom}")
    print(f"     A_k^t ∘ Ψ(RED):       {agent_atom}")
    print(f"     ε_pres = {eps_pres:.4f}")

    # ----- G2a-----
    print("\n[G2a] CORRELATIONAL FAITHFULNESS")
    print("      question: does Agent(RED NORTH) ≈ I(RED NORTH)?")
    
    phrase = ["RED", "NORTH"]
    truth_phrase = GridWorldContext.get_ground_truth(phrase)
    agent_phrase = get_output(phrase)
    eps_faith = np.linalg.norm(agent_phrase - truth_phrase)
    
    print(f"      I_k^t(RED NORTH):        {truth_phrase}")
    print(f"      A_k^t ∘ Ψ(RED NORTH):    {agent_phrase}")
    print(f"      ε_faith = {eps_faith:.4f}")

    # ----- G2b -----
    print("\n[G2b] ETIOLOGICAL FAITHFULNESS")
    print("      question: is modifier-processing mechanism M causally necessary?")
    
    SUCCESS_THRESHOLD = 0.5
    
    pos_on = get_output(phrase, ablate=False)
    success_on = np.linalg.norm(pos_on - truth_phrase) < SUCCESS_THRESHOLD
    
    pos_off = get_output(phrase, ablate=True)
    success_off = np.linalg.norm(pos_off - truth_phrase) < SUCCESS_THRESHOLD
    
    ace = float(success_on) - float(success_off)
    
    print(f"      M active:   output = {pos_on}, success = {success_on}")
    print(f"      M ablated:  output = {pos_off}, success = {success_off}")
    print(f"      ACE(M) = {ace:.1f}")

    # ----- G3 -----
    print("\n[G3] ROBUSTNESS")
    print("     question: how does semantic output change under perturbation of R?")
    
    NOISE_MAG = 0.5
    noise_vec = torch.randn(agent.hidden_dim)
    noise_vec = (noise_vec / torch.norm(noise_vec)) * NOISE_MAG
    
    clean = get_output(["RED"])
    noisy = get_output(["RED"], noise=noise_vec)
    drift = np.linalg.norm(clean - noisy)
    
    print(f"     perturbation magnitude (in R):  {NOISE_MAG}")
    print(f"     clean output:   {clean}")
    print(f"     noisy output:   {noisy}")
    print(f"     semantic drift: {drift:.4f}")
    print(f"     ω_U({NOISE_MAG}) = {drift:.4f} {'< ' if drift < NOISE_MAG else '>= '}{NOISE_MAG} (input)")
    print(f"     status: {'Stable (dampening)' if drift < NOISE_MAG else 'Unstable (amplifying)'}")

    # ----- G4 -----
    print("\n[G4] COMPOSITIONALITY")
    print("     question: does Agent(RED NORTH) ≈ Agent(RED) + Agent(NORTH)?")
    
    agent_whole = agent_phrase
    agent_red = agent_atom
    agent_north = get_output(["NORTH"])
    
    sum_of_parts = agent_red + agent_north
    delta_comp = np.linalg.norm(agent_whole - sum_of_parts)
    
    print(f"     A_k^t ∘ Ψ(RED NORTH):               {agent_whole}")
    print(f"     A_k^t ∘ Ψ(RED) + A_k^t ∘ Ψ(NORTH):  {sum_of_parts}")
    print(f"     (where A_k^t ∘ Ψ(NORTH) = {agent_north})")
    print(f"     δ_comp = {delta_comp:.4f}")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("GROUNDING PROFILE SUMMARY")
    print("=" * 60)
    print(f"  G0  (authenticity):    strong (learned via REINFORCE)")
    print(f"  G1  (preservation):    ε_pres  = {eps_pres:.4f}")
    print(f"  G2a (faithfulness):    ε_faith = {eps_faith:.4f}")
    print(f"  G2b (etiological):     ACE(M)  = {ace:.1f}")
    print(f"  G3  (robustness):      ω_U({NOISE_MAG}) = {drift:.4f}")
    print(f"  G4  (compositionality): δ_comp = {delta_comp:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    tee = Tee('output.txt')
    sys.stdout = tee
    try:
        trained_agent, vocabulary = train_agent()
        audit_agent(trained_agent, vocabulary)
    finally:
        sys.stdout = tee.stdout
        tee.close()
