from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP, ICMP
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import logging

# Set up logging
log_filename = datetime.now().strftime("app_log_%Y_%m_%d_%H%M%S.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Global dictionary to store flow start and end times
flow_times = {}

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected features (use exact names as in training)
expected_features = [
    "TCP_FLAGS",
    "DST_TO_SRC_SECOND_BYTES",
    "L4_DST_PORT",
    "L7_PROTO",
    "IN_BYTES",
    "TCP_WIN_MAX_IN",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "SRC_TO_DST_SECOND_BYTES",
    "NUM_PKTS_UP_TO_128_BYTES",
    "IPV4_SRC_ADDR",
    "FLOW_DURATION_MILLISECONDS",
    "LONGEST_FLOW_PKT",
    "MAX_IP_PKT_LEN",
    "IPV4_DST_ADDR",
    "DURATION_IN",
    "SHORTEST_FLOW_PKT",
]

attack_labels = {
    0: "Analysis",
    1: "Backdoor",
    2: "Benign",
    3: "Bot",
    4: "Brute Force",
    5: "DDoS",
    6: "DoS",
    7: "Exploits",
    8: "Fuzzers",
    9: "Generic",
    10: "Infilteration",
    11: "Reconnaissance",
    12: "Shellcode",
    13: "Theft",
    14: "Worms",
    15: "Injection",
    16: "MITM",
    17: "Password",
    18: "Ransomware",
    19: "Scanning",
    20: "XSS",
}

app = Flask(__name__)

# Initialize a DataFrame to store packet features
packets_df = pd.DataFrame(columns=expected_features)


def extract_features(packet):
    """Extract relevant features from the packet."""
    features = {key: np.nan for key in expected_features}
    try:
        if IP in packet:
            ip_packet = packet[IP]
            features["IPV4_SRC_ADDR"] = ip_packet.src
            features["IPV4_DST_ADDR"] = ip_packet.dst
            features["PROTOCOL"] = ip_packet.proto
            features["FLOW_DURATION_MILLISECONDS"] = float(packet.time)

            # Create a unique identifier for the flow
            flow_id = (
                ip_packet.src,
                ip_packet.dst,
                ip_packet.proto,
                (
                    packet[TCP].sport
                    if TCP in packet
                    else (packet[UDP].sport if UDP in packet else None)
                ),
                (
                    packet[TCP].dport
                    if TCP in packet
                    else (packet[UDP].dport if UDP in packet else None)
                ),
            )

            if flow_id not in flow_times:
                flow_times[flow_id] = {
                    "start": packet.time,
                    "end": packet.time,
                    "min_pkt_len": float("inf"),
                    "max_pkt_len": 0,
                    "total_bytes": 0,
                    "num_pkts": 0,
                }
            else:
                flow_times[flow_id]["end"] = packet.time

            # Update flow stats
            flow_stats = flow_times[flow_id]
            packet_length = len(packet)
            flow_stats["total_bytes"] += packet_length
            flow_stats["num_pkts"] += 1
            flow_stats["min_pkt_len"] = min(flow_stats["min_pkt_len"], packet_length)
            flow_stats["max_pkt_len"] = max(flow_stats["max_pkt_len"], packet_length)

            if TCP in packet:
                tcp_packet = packet[TCP]
                features["L4_SRC_PORT"] = tcp_packet.sport
                features["L4_DST_PORT"] = tcp_packet.dport
                features["TCP_FLAGS"] = int(tcp_packet.flags)
                features["TCP_WIN_MAX_IN"] = getattr(tcp_packet, "window", np.nan)

                payload_len = len(tcp_packet.payload) if tcp_packet.payload else 0
                features["RETRANSMITTED_IN_BYTES"] = payload_len
                features["RETRANSMITTED_OUT_PKTS"] = payload_len
                features["RETRANSMITTED_OUT_BYTES"] = payload_len
                features["RETRANSMITTED_IN_PKTS"] = payload_len

            elif UDP in packet:
                udp_packet = packet[UDP]
                features["L4_SRC_PORT"] = udp_packet.sport
                features["L4_DST_PORT"] = udp_packet.dport

            features["IN_BYTES"] = packet_length
            features["DST_TO_SRC_SECOND_BYTES"] = (
                packet_length / float(packet.time) if packet.time else np.nan
            )
            features["SRC_TO_DST_AVG_THROUGHPUT"] = (
                float(packet.time) / packet_length if packet_length else np.nan
            )
            features["NUM_PKTS_UP_TO_128_BYTES"] = 1 if packet_length <= 128 else 0
            features["NUM_PKTS_128_TO_256_BYTES"] = (
                1 if 128 < packet_length <= 256 else 0
            )
            features["NUM_PKTS_256_TO_512_BYTES"] = (
                1 if 256 < packet_length <= 512 else 0
            )
            features["NUM_PKTS_512_TO_1024_BYTES"] = (
                1 if 512 < packet_length <= 1024 else 0
            )
            features["NUM_PKTS_1024_TO_1514_BYTES"] = (
                1 if 1024 < packet_length <= 1514 else 0
            )

            # Calculate DURATION_IN
            if flow_id in flow_times:
                start_time = flow_times[flow_id]["start"]
                end_time = packet.time
                features["DURATION_IN"] = (
                    end_time - start_time
                ) * 1000  # Convert to milliseconds

                features["SHORTEST_FLOW_PKT"] = flow_times[flow_id]["min_pkt_len"]
                features["LONGEST_FLOW_PKT"] = flow_times[flow_id]["max_pkt_len"]
                features["MAX_IP_PKT_LEN"] = flow_times[flow_id]["max_pkt_len"]
                if flow_times[flow_id]["num_pkts"] > 0:
                    features["SRC_TO_DST_SECOND_BYTES"] = (
                        (flow_times[flow_id]["total_bytes"] / (end_time - start_time))
                        if (end_time - start_time) > 0
                        else np.nan
                    )

            # Placeholder for Layer 7 Protocol
            if TCP in packet:
                features["L7_PROTO"] = 6  # HTTP
            elif UDP in packet:
                features["L7_PROTO"] = 17  # DNS, DHCP
            elif ICMP in packet:
                features["L7_PROTO"] = 1  # ICMP
            else:
                features["L7_PROTO"] = np.nan

    except Exception as e:
        logger.error(f"Error extracting features: {e} - Packet Info: {packet}")

    return features


def preprocess_live_data(live_data):
    """Apply the same preprocessing steps as the training data."""
    df = pd.DataFrame(
        [live_data], columns=expected_features
    )  # Wrap in list for single-row DataFrame

    # Encode categorical features
    for column in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Feature scaling
    df_scaled = scaler.transform(df)
    return df_scaled


def predict_anomaly(packet):
    features = extract_features(packet)
    if features:
        preprocessed_data = preprocess_live_data(features)
        prediction = model.predict(preprocessed_data)
        attack_type = attack_labels.get(prediction[0], "Unknown")
        return attack_type, features  # Return both attack type and features
    return None, None


def save_packet_to_csv(packet_features):
    """Save packet features to a CSV file, replacing NaNs with zeros."""
    global packets_df
    # Convert features to a DataFrame and replace NaNs with zeros
    packet_df = pd.DataFrame([packet_features], columns=expected_features).fillna(0)
    packets_df = pd.concat(
        [packets_df, packet_df],
        ignore_index=True,
    )
    packets_df.to_csv(csv_filename, index=False)


def initialize_csv_file():
    """Initialize CSV file with timestamped filename."""
    global csv_filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    csv_filename = f"captured_packets_{timestamp}.csv"
    logger.info(f"Saving to file: {csv_filename}")
 

@app.route("/predict_live", methods=["GET"]) 
def predict_live():
    try:
        sniff(
            prn=lambda x: handle_packet(x),
            filter="ip",  # Adjusted filter to include ICMP packets
        )
        return jsonify({"status": "Scanning complete"}), 200
    except Exception as e:
        logger.error(f"Error during sniffing: {e}")
        return jsonify({"error": str(e)}), 500


def handle_packet(packet):
    # Extract and predict anomaly
    attack_type, features = predict_anomaly(packet)

    # Get basic packet information
    src_ip = packet[IP].src if IP in packet else "N/A"
    src_port = (
        packet[TCP].sport
        if TCP in packet
        else (packet[UDP].sport if UDP in packet else "N/A")
    )
    dst_ip = packet[IP].dst if IP in packet else "N/A"
    dst_port = (
        packet[TCP].dport
        if TCP in packet
        else (packet[UDP].dport if UDP in packet else "N/A")
    )
    protocol = packet[IP].proto if IP in packet else "N/A"

    # Format packet details
    packet_info = (
        f"Packet Info: {src_ip}:{src_port} > {dst_ip}:{dst_port} "
        f"Protocol: {protocol} Attack Type: {attack_type}"
    )

    # Save features to the timestamped CSV file
    if features:
        save_packet_to_csv(features)

    # Print detailed packet information
    print(packet_info)
    logger.info(packet_info)

    # Print detailed scapy packet info (Ethernet layer included if present)
    if packet.haslayer(IP):
        detailed_info = f"Handling packet: {packet.summary()}"
    else:
        detailed_info = f"Handling non-IP packet: {packet.summary()}"

    print(detailed_info)
    logger.info(detailed_info)


if __name__ == "__main__":
    initialize_csv_file()  # Initialize CSV file before starting the app
    app.run(port=5000)