import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class BetaState:
    """
    Tracks Strength AND Confidence explicitly using Beta Distribution counts.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Evidence for True
        self.beta = beta    # Evidence for False

        # Base Priors
        self.prior_a = alpha
        self.prior_b = beta

    @property
    def strength(self):
        # Mean probability: alpha / (alpha + beta)
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self):
        # Evidence saturation. 
        # Map total evidence (counts) to [0, 1] confidence score.
        # "10 units of evidence" = roughly 90% confidence.
        total_evidence = self.alpha + self.beta
        return total_evidence / (total_evidence + 1.0) 

    def __repr__(self):
        return f"Beta(a={self.alpha:.2f}, b={self.beta:.2f})"

class BetaFactorGraph:
    def __init__(self):
        self.nodes = {} 
        self.factors = []

    def get_or_create_node(self, name):
        if name not in self.nodes:
            # Initialize with Neutral Prior (Laplace Smoothing)
            # a=1, b=1 -> Strength=0.5, Conf=Low
            self.nodes[name] = BetaState(1.0, 1.0)
        return self.nodes[name]

    def add_dependency_rule(self, pair_str, rule_strength, rule_confidence):
        """
        Registers a Modus Ponens rule: A -> B
        """
        # parts = pair_str.split(' -- ')
        # src = parts[0].strip()
        # dst = parts[1].strip()
        
        # # Ensure nodes exist
        # self.get_or_create_node(src)
        # self.get_or_create_node(dst)

        # # Store the rule params
        # self.factors.append({
        #     'src': src,
        #     'dst': dst,
        #     's': rule_strength,
        #     'c': rule_confidence
        # })
        parts = pair_str.split(' -- ')
        if len(parts) != 2: return
        src, dst = parts[0], parts[1]
        
        # Ensure nodes exist
        self.get_or_create_node(src)
        self.get_or_create_node(dst)
        
        # Check if rule already exists to avoid duplicates
        existing_rule = None
        for rule in self.factors:
            if rule['src'] == src and rule['dst'] == dst:
                existing_rule = rule
                break
        
        if existing_rule:
            # High confidence overwrites low confidence (Greedy)
            # if confidence > existing_rule['c']:
            #     existing_rule['s'] = strength
            #     existing_rule['c'] = confidence
            
            # Weighted Update (New data counts for 30%)
            alpha = 0.7
            existing_rule['s'] = (1 - alpha) * existing_rule['s'] + alpha * rule_strength
            existing_rule['c'] = (1 - alpha) * existing_rule['c'] + alpha * rule_confidence
            
        else:
            # CREATE NEW
            rule = {
                'src': src,
                'dst': dst,
                's': rule_strength,
                'c': rule_confidence
            }
            self.factors.append(rule)

    def set_prior(self, name, stv_strength, stv_confidence, base_counts=10.0):
        """
        Anchors a node with external observation (e.g. from Miner).
        This PREVENTS the "floating 0.5" issue.
        """
        node = self.get_or_create_node(name)
        
        # Convert STV (Prob, Conf) -> Beta Counts (Alpha, Beta)
        evidence = max(0.1, stv_confidence * base_counts)
        a = (stv_strength * evidence) + 1.0
        b = ((1.0 - stv_strength) * evidence) + 1.0
        
        node.alpha = node.prior_a = a
        node.beta = node.prior_b = b
    
    def visualize(self, title="Beta Factor Graph"):
        """
        Visualizes the Factor Graph distinguishing Variables and Factors.
        - Variable Nodes (Circles): Color=Strength, Size=Confidence
        - Factor Nodes (Squares): Represent the rules connecting variables
        """
        G = nx.DiGraph()
        
        # Lists to separate types for drawing
        var_nodes = []
        factor_nodes = []
        
        var_colors = []
        var_sizes = []
        labels = {}
        
        for name, node in self.nodes.items():
            G.add_node(name)
            var_nodes.append(name)
            
            s = node.strength
            c = node.confidence
            
            var_colors.append(s)
            var_sizes.append(1000 + (c * 3000))
            # Label with current belief
            labels[name] = f"{name}\nS:{s:.2f}\nC:{c:.2f}"

        for i, rule in enumerate(self.factors):
            # Create a unique ID for the factor node based on connection
            f_id = f"F_{rule['src']}_{rule['dst']}_{i}"
            factor_nodes.append(f_id)
            
            G.add_node(f_id)
            # Label the factor with its rule logic
            labels[f_id] = f"Rule\nS:{rule['s']:.2f}\nC:{rule['c']:.2f}"
            
            # Connect: Source -> Factor -> Destination
            G.add_edge(rule['src'], f_id)
            G.add_edge(f_id, rule['dst'])

        # 3. Draw
        plt.figure(figsize=(12, 10))
        # Use spring layout but distinct nodes help separate them
        pos = nx.spring_layout(G, k=1.0) 
        
        # Draw Variables (Circles)
        if var_nodes:
            nodes = nx.draw_networkx_nodes(G, pos, 
                                        nodelist=var_nodes,
                                        node_color=var_colors, 
                                        node_size=var_sizes, 
                                        cmap=plt.cm.RdYlGn, 
                                        vmin=0.0, vmax=1.0,
                                        edgecolors='black',
                                        node_shape='o') # 'o' for Circle
        
        # Draw Factors (Squares)
        if factor_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                nodelist=factor_nodes,
                                node_color='lightgray', 
                                node_size=2500, 
                                node_shape='s', # 's' for Square
                                edgecolors='black',
                                alpha=0.9)

        # Draw Edges & Labels
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)        
        nx.draw_networkx_labels(G, pos, labels, font_size=9)
        
        if var_nodes:
            plt.colorbar(nodes, label="Strength (Probability of True)")
            
        plt.title(title)
        plt.axis('off')

        output_file = "beta_factor_graph.png"
        plt.savefig(output_file)
        print(f"Graph visualization saved to: {output_file}")
        plt.close()



    def run_evidence_propagation(self, steps=10, decay=0.9):
        print(f"--- Running Beta-Propagation (Modus Ponens + Abduction + Revision) ---")
        
        for i in range(steps):
            max_delta = 0
            
            # REVISION PART 1: Create an empty "Inbox" for every node
            # We will accumulate evidence here for this step.
            inboxes = {name: {'a': 0.0, 'b': 0.0} for name in self.nodes}

            # --- CALCULATE MESSAGES ---
            for rule in self.factors:
                src_name = rule['src']
                dst_name = rule['dst']
                src_node = self.nodes[src_name]
                dst_node = self.nodes[dst_name]
                
                S = rule['s']
                C = rule['c']
                rule_capacity = C * 20.0
                
                # ==========================================
                # FORWARD PASS (Modus Ponens)
                # ==========================================
                p_src = src_node.strength
                src_evidence = src_node.alpha + src_node.beta
                
                # Logic: If Src is True -> Dst is True (prob S). 
                # If Src is False -> Dst is Unknown (prob 0.5)
                fwd_strength = (p_src * S) + ((1.0 - p_src) * 0.5)
                
                # Attenuate evidence (decay) over distance to guarantee convergence
                fwd_evidence = min(src_evidence * decay, rule_capacity)
                
                # Put message in Destination's Inbox
                inboxes[dst_name]['a'] += fwd_strength * fwd_evidence
                inboxes[dst_name]['b'] += (1.0 - fwd_strength) * fwd_evidence

                # ==========================================
                # BACKWARD PASS (Abduction & Modus Tollens)
                # ==========================================
                p_dst = dst_node.strength
                dst_evidence = dst_node.alpha + dst_node.beta
                
                # Logic: 
                # Abduction: If Dst is True -> Src is likely True (prob S)
                # Modus Tollens: If Dst is False -> Src is definitely False (prob 1-S)
                bwd_strength = (p_dst * S) + ((1.0 - p_dst) * (1.0 - S))
                
                # Backward flow is usually weaker (more uncertain) than forward flow
                # We apply a harsher penalty to abducted evidence.
                bwd_evidence = min(dst_evidence * (decay * 0.5), rule_capacity)
                
                # Put message in Source's Inbox
                inboxes[src_name]['a'] += bwd_strength * bwd_evidence
                inboxes[src_name]['b'] += (1.0 - bwd_strength) * bwd_evidence

            # ==========================================
            # REVISION PART 2: Apply Inbox (Fusion)
            # ==========================================
            for name, node in self.nodes.items():
                # The true Beta-Fusion operator: 
                # New State = Prior Base + Sum of all incoming evidence
                new_a = node.prior_a + inboxes[name]['a']
                new_b = node.prior_b + inboxes[name]['b']
                
                # Track convergence
                delta = abs(new_a - node.alpha) + abs(new_b - node.beta)
                max_delta = max(max_delta, delta)
                
                # Update node
                node.alpha = new_a
                node.beta = new_b
            
            print(f"Step {i+1}: Max Evidence Update = {max_delta:.4f}")
            if max_delta < 0.05:
                print("Convergence reached.")
                break

# --- Main Execution ---

# data = [
#     {'pair': 'A -- B', 'strength': 0.803, 'confidence': 0.3775}, 
#     {'pair': 'B -- C', 'strength': 0.749, 'confidence': 0.3039}, 
#     {'pair': 'C -- D', 'strength': 0.732, 'confidence': 0.2304},
# ]

# # 1. Init
# bg = BetaFactorGraph()

# for row in data:
#     bg.add_dependency_rule(row['pair'], row['strength'], row['confidence'])


# bg.set_prior("A", stv_strength=0.9, stv_confidence=0.8) # "We are pretty sure A is True"

# bg.run_evidence_propagation(steps=10)

# print("\n--- Final STV Results ---")
# print(f"{'Variable':<10} | {'Strength':<8} | {'Confidence':<10} | {'Counts (a/b)'}")
# for name, node in bg.nodes.items():
#     s = node.strength
#     c = node.confidence
#     print(f"{name:<10} | {s:.4f}   | {c:.4f}     | {node.alpha:.1f}/{node.beta:.1f}")

# bg.visualize()
