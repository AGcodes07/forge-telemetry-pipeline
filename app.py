import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time

app = FastAPI(title="AG3 FORGE Telemetry Engine Prototype")

class TelemetryPacket(BaseModel):
    drone_id: str
    timestamp: int
    metrics: List[float]

class DroneClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # [0: NOMINAL, 1: CONTESTED_ANOMALY]
        )
        self.eval()
    def forward(self, x): return self.net(x)

model = DroneClassifier(input_dim=6)
MEANS = np.array([0.0, 0.0, 1.0, 15.0, -50.0, 4000.0])
STDS = np.array([0.2, 0.2, 0.5, 1.0, 10.0, 500.0])

@app.post("/api/v1/telemetry")
async def process_telemetry(packet: TelemetryPacket):
    start_time = time.time()
    try:
        if len(packet.metrics) != 6:
            raise ValueError("Expected 6 telemetry metrics.")
        
        # Normalize and run edge inference
        norm_metrics = (np.array(packet.metrics) - MEANS) / (STDS + 1e-6)
        tensor_input = torch.tensor(norm_metrics, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = int(torch.argmax(model(tensor_input), dim=1).numpy()[0])
        
        status_map = {0: "NOMINAL", 1: "CONTESTED_ANOMALY"}
        return {
            "drone_id": packet.drone_id,
            "system_status": status_map[prediction],
            "timestamp": packet.timestamp,
            "latency_ms": round((time.time() - start_time) * 1000, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
