from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import DQN
import numpy as np

# Load your trained model
model = DQN.load("models/dqn_pricing_model")

# Define the observation data model using Pydantic
class Observation(BaseModel):
    data: list

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(observation: Observation):
    try:
        # Convert the input observation to a numpy array
        obs_array = np.array(observation.data, dtype=np.float32)

        # Ensure the observation is reshaped correctly for the model
        obs_array = obs_array.reshape((1, -1))

        # Use the model to predict the action
        action, _ = model.predict(obs_array)

        # Return the predicted action
        return {"action": int(action[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict/")
async def predict_get():
    
    test_data = [-0.9865911037541286,-0.2663335277556618,0.88634081692978,1.0624870972327003,-0.6224555115317711,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,213.44325079724317]
    obs_array = np.array(test_data, dtype=np.float32).reshape((1, -1))
    action, _ = model.predict(obs_array)
    return {"action": int(action[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
