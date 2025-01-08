# **Dynamic Pricing with Reinforcement Learning for Ride Sharing**

This project implements a dynamic pricing model for ride-sharing services using **Reinforcement Learning** with **Deep Q-Networks (DQN)**. The system dynamically adjusts fares in real-time by leveraging historical ride data and key features to optimize pricing strategies. Deployed via **FastAPI**, the project focuses on reducing latency and enhancing revenue optimization.

---

## **Features**
- **Reinforcement Learning**: Implemented using DQN with **Stable Baselines3**.
- **Real-Time Pricing**: Dynamically adjusts fares based on demand, ride history, and other features.
- **Web Deployment**: Deployed through **FastAPI** for real-time fare predictions.
- **Optimization**: Reduced latency by 14%, improving accuracy and revenue optimization.

---

## **Tech Stack**
- **Programming Language**: Python
- **Libraries**:
  - Reinforcement Learning: Stable Baselines3
  - Data Processing: pandas, NumPy
  - Deployment: FastAPI
- **Deployment**: REST API for real-time integration

---

## **How to Run**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dynamic-pricing-ride-sharing.git
   cd dynamic-pricing-ride-sharing
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the API**:
   Open your browser and navigate to `http://127.0.0.1:8000/docs` for API documentation.

---

## **Workflow**
1. **Data Preparation**:
   - Preprocessed historical ride data for training.
   - Engineered features critical to pricing strategies.

2. **Model Training**:
   - Implemented a **Deep Q-Network (DQN)** using **Stable Baselines3**.
   - Tuned hyperparameters to optimize learning performance.

3. **Deployment**:
   - Integrated the trained model with **FastAPI** for real-time pricing.

---

## **Usage**
- Use the API to input ride details and retrieve optimized fare predictions in real-time.

---

## **Results**
- Achieved a **14% reduction in latency**, enabling faster fare predictions.
- Enhanced revenue optimization by providing accurate fare adjustments in real-time.

---

## **Future Enhancements**
- Extend the model to incorporate multi-agent reinforcement learning for optimizing driver and rider interactions.
- Add a dashboard to visualize fare trends and pricing strategies.

---

## **Screenshots**
_Screenshots showcasing the API endpoints or example outputs._

---

## **Contributors**
- [Madhumitha Venkatesan](https://github.com/vmadhuuu)

---

## **License**
This project is licensed under the MIT License.
