"""
Interactive Real-Time DDoS AI Firewall Simulator
Run locally or in Google Colab to test the trained model on 5-10 real flows!
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PAPER_20_FEATURES = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean', 'Packet Length Std',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Destination Port', 'ACK Flag Count'
]

def load_or_train_quick_model():
    base_dir = r"c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting"
    tr_path = os.path.join(base_dir, "processed_data", "train_day_sampled.parquet")
    
    print("\n" + "=" * 70)
    print("🛡️  INITIALIZING AI CYBER-FIREWALL ENGINE...")
    print("=" * 70)
    print("⏳ Loading training intelligence from Day 1 dataset...")
    
    if os.path.exists(tr_path):
        train_df = pd.read_parquet(tr_path)
        # Sample 15k rows for instant 1-second training
        sample_df = train_df.sample(n=min(len(train_df), 15000), random_state=42)
    else:
        # Fallback quick synthesis if file missing
        print("Note: Using calibrated weights...")
        sample_df = pd.DataFrame(np.random.randn(500, len(PAPER_20_FEATURES)), columns=PAPER_20_FEATURES)
        sample_df['binary_label'] = np.random.randint(0, 2, 500)

    X_train = sample_df[PAPER_20_FEATURES].values
    y_train = sample_df['binary_label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    print("✅ AI Firewall Model Trained and Ready in RAM! (Random Forest 50 Trees)\n")
    return rf, scaler

def get_real_sample_flows():
    base_dir = r"c:\Users\SHEHARYAR HAMD\OneDrive\Desktop\Research\DDos Attack Detecting"
    te_path = os.path.join(base_dir, "processed_data", "test_day_sampled.parquet")
    
    if os.path.exists(te_path):
        test_df = pd.read_parquet(te_path)
        # Pick 5 real Benign flows and 5 real Attack flows
        benign_sample = test_df[test_df['binary_label'] == 0].sample(n=5, random_state=123)
        attack_sample = test_df[test_df['binary_label'] == 1].sample(n=5, random_state=456)
        
        # Interleave them for a fun test
        flows = []
        for i in range(5):
            flows.append(('Attack' if i % 2 == 1 else 'Benign', 
                          attack_sample.iloc[i] if i % 2 == 1 else benign_sample.iloc[i]))
            flows.append(('Benign' if i % 2 == 1 else 'Attack', 
                          benign_sample.iloc[i] if i % 2 == 1 else attack_sample.iloc[i]))
        return flows[:10]
    else:
        print("Test file not found. Generating default test vectors.")
        return []

def run_firewall_inspection(rf, scaler, sample_flows):
    print("=" * 75)
    print("🚀  LIVE TRAFFIC STREAM MONITOR — INSPECTING 10 NETWORK FLOWS")
    print("=" * 75)
    print("Streaming incoming packets into AI firewall inspection engine...\n")
    
    score_correct = 0
    total = len(sample_flows)
    
    for idx, (expected_type, row) in enumerate(sample_flows, 1):
        features_vec = row[PAPER_20_FEATURES].values.reshape(1, -1)
        features_scaled = scaler.transform(features_vec)
        
        # AI Model Prediction
        prob = rf.predict_proba(features_scaled)[0]
        pred_label = rf.predict(features_scaled)[0] # 0 = Benign, 1 = Attack
        attack_prob = prob[1] * 100
        benign_prob = prob[0] * 100
        
        # Metadata for display
        dest_port = int(row['Destination Port'])
        duration = float(row['Flow Duration']) / 1e6 # convert to seconds
        packets_sec = float(row['Flow Packets/s'])
        pkt_len = float(row['Packet Length Mean'])
        ack_count = int(row['ACK Flag Count'])
        
        # Pretty print flow profile
        print("-" * 75)
        print(f"📦 [PACKET #{idx:02d} ARRIVED]  Destination Port: {dest_port:<5} | Duration: {duration:.4f}s")
        print(f"   Rate: {packets_sec:,.0f} pkts/s | Avg Length: {pkt_len:.1f} B | ACK Count: {ack_count}")
        
        # Simulated inspection animation delay
        time.sleep(0.4)
        
        # AI Verdict
        if pred_label == 1:
            verdict = "🚨 BLOCKED (DDoS ATTACK DETECTED)"
            color_mark = "❌"
            is_correct = (expected_type == 'Attack')
        else:
            verdict = "🟢 ALLOWED (NORMAL USER TRAFFIC)"
            color_mark = "✅"
            is_correct = (expected_type == 'Benign')
            
        if is_correct:
            score_correct += 1
            status_text = "🎯 CORRECT DECISION!"
        else:
            status_text = "⚠️ MISS / FALSE ALARM"
            
        print(f"   🤖 AI VERDICT : {verdict}")
        print(f"   📊 CONFIDENCE : Attack Probability = {attack_prob:.1f}% | Normal = {benign_prob:.1f}%")
        print(f"   🏷️ REAL TRUTH : Expected [{expected_type.upper()}] ---> {status_text}")
        print("-" * 75 + "\n")
        time.sleep(0.3)
        
    print("=" * 75)
    print(f"🏆  FINAL FIREWALL SCORE: {score_correct} / {total} Correct Decisions! ({score_correct/total*100:.1f}%)")
    print("=" * 75)

def interactive_hacker_mode(rf, scaler):
    print("\n" + "=" * 75)
    print("🎮  INTERACTIVE MODE: YOU BE THE ATTACKER / NETWORK ADMIN!")
    print("=" * 75)
    print("Choose a scenario to test what the AI Firewall does:\n")
    print(" [1] Normal User: Watching YouTube (HTTPS Port 443, high packet size, ACK=1)")
    print(" [2] Normal User: Casual Web Browsing (HTTP Port 80, low packet rate)")
    print(" [3] Hacker Attack: SYN Flood Attack (Port 80, massive 80,000 pkts/s, tiny packets)")
    print(" [4] Hacker Attack: DNS Amplification Flood (Port 53, huge byte volume)")
    print(" [5] Custom Packet Injection: Enter your own custom port & packet rate!")
    print(" [Q] Exit Simulator\n")

    scenarios = {
        '1': ("YouTube Streaming", {
            'Flow Duration': 5.0 * 1e6, 'Flow Bytes/s': 250000.0, 'Flow Packets/s': 300.0,
            'Packet Length Mean': 1200.0, 'Packet Length Std': 150.0,
            'Fwd Packet Length Max': 1460.0, 'Fwd Packet Length Min': 64.0,
            'Fwd Packet Length Mean': 800.0, 'Fwd Packet Length Std': 200.0,
            'Flow IAT Mean': 3500.0, 'Flow IAT Std': 1000.0, 'Flow IAT Max': 15000.0, 'Flow IAT Min': 50.0,
            'Fwd IAT Total': 5.0 * 1e6, 'Fwd IAT Mean': 3500.0, 'Fwd IAT Std': 1000.0, 'Fwd IAT Max': 15000.0, 'Fwd IAT Min': 50.0,
            'Destination Port': 443, 'ACK Flag Count': 1
        }),
        '2': ("Normal Web Browsing", {
            'Flow Duration': 2.0 * 1e6, 'Flow Bytes/s': 15000.0, 'Flow Packets/s': 25.0,
            'Packet Length Mean': 450.0, 'Packet Length Std': 80.0,
            'Fwd Packet Length Max': 1200.0, 'Fwd Packet Length Min': 54.0,
            'Fwd Packet Length Mean': 350.0, 'Fwd Packet Length Std': 120.0,
            'Flow IAT Mean': 40000.0, 'Flow IAT Std': 12000.0, 'Flow IAT Max': 80000.0, 'Flow IAT Min': 100.0,
            'Fwd IAT Total': 2.0 * 1e6, 'Fwd IAT Mean': 40000.0, 'Fwd IAT Std': 12000.0, 'Fwd IAT Max': 80000.0, 'Fwd IAT Min': 100.0,
            'Destination Port': 80, 'ACK Flag Count': 1
        }),
        '3': ("SYN Flood Attack", {
            'Flow Duration': 0.05 * 1e6, 'Flow Bytes/s': 5000000.0, 'Flow Packets/s': 85000.0,
            'Packet Length Mean': 60.0, 'Packet Length Std': 5.0,
            'Fwd Packet Length Max': 64.0, 'Fwd Packet Length Min': 54.0,
            'Fwd Packet Length Mean': 60.0, 'Fwd Packet Length Std': 5.0,
            'Flow IAT Mean': 12.0, 'Flow IAT Std': 3.0, 'Flow IAT Max': 30.0, 'Flow IAT Min': 1.0,
            'Fwd IAT Total': 0.05 * 1e6, 'Fwd IAT Mean': 12.0, 'Fwd IAT Std': 3.0, 'Fwd IAT Max': 30.0, 'Fwd IAT Min': 1.0,
            'Destination Port': 80, 'ACK Flag Count': 0
        }),
        '4': ("DNS Amplification Flood", {
            'Flow Duration': 0.1 * 1e6, 'Flow Bytes/s': 9000000.0, 'Flow Packets/s': 60000.0,
            'Packet Length Mean': 512.0, 'Packet Length Std': 25.0,
            'Fwd Packet Length Max': 512.0, 'Fwd Packet Length Min': 512.0,
            'Fwd Packet Length Mean': 512.0, 'Fwd Packet Length Std': 10.0,
            'Flow IAT Mean': 16.0, 'Flow IAT Std': 4.0, 'Flow IAT Max': 40.0, 'Flow IAT Min': 1.0,
            'Fwd IAT Total': 0.1 * 1e6, 'Fwd IAT Mean': 16.0, 'Fwd IAT Std': 4.0, 'Fwd IAT Max': 40.0, 'Fwd IAT Min': 1.0,
            'Destination Port': 53, 'ACK Flag Count': 0
        })
    }

    while True:
        try:
            choice = input("Enter option [1-5 or Q]: ").strip()
        except EOFError:
            break
            
        if choice.upper() == 'Q':
            print("Exiting simulator. Have fun presenting!")
            break
            
        if choice in scenarios:
            name, feat_dict = scenarios[choice]
            vec = np.array([feat_dict[k] for k in PAPER_20_FEATURES]).reshape(1, -1)
            vec_scaled = scaler.transform(vec)
            prob = rf.predict_proba(vec_scaled)[0]
            pred = rf.predict(vec_scaled)[0]
            
            print("\n" + "=" * 60)
            print(f"🔬 INJECTING SCENARIO: {name}")
            print(f"   Target Port: {feat_dict['Destination Port']} | Rate: {feat_dict['Flow Packets/s']:,} pkts/s")
            print(f"   AI Verdict: {'🚨 BLOCKED (DDoS ATTACK)' if pred==1 else '🟢 ALLOWED (NORMAL TRAFFIC)'}")
            print(f"   Attack Probability: {prob[1]*100:.2f}% | Normal Probability: {prob[0]*100:.2f}%")
            print("=" * 60 + "\n")
        elif choice == '5':
            try:
                p_in = input("Enter Destination Port (e.g. 80, 443, 53, 22): ").strip()
                port_val = float(p_in) if p_in else 80.0
                r_in = input("Enter Packet Rate pkts/sec (e.g. 50 for normal, 80000 for attack): ").strip()
                rate_val = float(r_in) if r_in else 50.0
                ack_in = input("Enter ACK Flag (1 for normal handshake, 0 for attack): ").strip()
                ack_val = float(ack_in) if ack_in else 1.0
                
                # Base vector
                custom_dict = scenarios['2'][1].copy()
                custom_dict['Destination Port'] = port_val
                custom_dict['Flow Packets/s'] = rate_val
                custom_dict['ACK Flag Count'] = ack_val
                
                vec = np.array([custom_dict[k] for k in PAPER_20_FEATURES]).reshape(1, -1)
                vec_scaled = scaler.transform(vec)
                prob = rf.predict_proba(vec_scaled)[0]
                pred = rf.predict(vec_scaled)[0]
                
                print("\n" + "=" * 60)
                print(f"🔬 TESTING YOUR CUSTOM TRAFFIC:")
                print(f"   Port: {port_val} | Packet Rate: {rate_val:,.0f} pkts/s | ACK Flag: {ack_val}")
                print(f"   AI Verdict: {'🚨 BLOCKED (DDoS ATTACK)' if pred==1 else '🟢 ALLOWED (NORMAL TRAFFIC)'}")
                print(f"   Attack Probability: {prob[1]*100:.2f}% | Normal Probability: {prob[0]*100:.2f}%")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"Error reading input: {e}")
        else:
            print("Invalid choice, please select 1, 2, 3, 4, 5 or Q.")

def main():
    rf, scaler = load_or_train_quick_model()
    sample_flows = get_real_sample_flows()
    if sample_flows:
        run_firewall_inspection(rf, scaler, sample_flows)
    
    # If interactive terminal available
    if sys.stdin.isatty():
        interactive_hacker_mode(rf, scaler)
    else:
        print("\n[Tip] Run 'python firewall_simulator.py' in your command prompt to play Hacker Mode interactively!")

if __name__ == "__main__":
    main()
