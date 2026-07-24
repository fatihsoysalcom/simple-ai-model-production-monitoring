import time
import random
import statistics

# --- Simulated AI Model ---
# This function simulates a very simple sentiment analysis model.
# In a real scenario, this would be a loaded ML model (e.g., from TensorFlow, PyTorch, scikit-learn).
def predict_sentiment(text: str) -> float:
    """
    Simulates an AI model predicting sentiment score for a given text.
    Returns a score between -1.0 (very negative) and 1.0 (very positive).
    """
    positive_keywords = ["harika", "mükemmel", "iyi", "başarılı", "mutlu", "beğendim"]
    negative_keywords = ["kötü", "berbat", "sorunlu", "hayal kırıklığı", "üzücü", "sevmedim"]

    score = 0.0
    text_lower = text.lower()

    for keyword in positive_keywords:
        if keyword in text_lower:
            score += 0.3
    for keyword in negative_keywords:
        if keyword in text_lower:
            score -= 0.3

    # Add some randomness to simulate real model variability
    score += random.uniform(-0.1, 0.1)
    return max(-1.0, min(1.0, score)) # Clamp score between -1 and 1

# --- Monitoring System ---
prediction_log = []
MONITORING_WINDOW_SIZE = 10 # Number of recent predictions to consider for metrics
ERROR_THRESHOLD = 0.4      # Threshold for Mean Absolute Error to flag potential performance degradation
AVG_SENTIMENT_THRESHOLD_CHANGE = 0.3 # Threshold for average sentiment difference to flag concept drift

def simulate_actual_sentiment(text: str, current_phase: str) -> float:
    """
    Simulates the 'ground truth' sentiment for a given text.
    This would typically come from user feedback or labeled data in a real system.
    Introduces a 'drift' in the 'actual' sentiment during the 'drift_phase'.
    """
    base_score = predict_sentiment(text) # Use the model's base logic for consistency
    if current_phase == "drift_phase":
        # Simulate a scenario where the actual sentiment shifts, 
        # perhaps due to new slang, changing user preferences, or seasonal effects.
        return base_score - 0.5 # Actual sentiment is consistently lower than model predicts
    return base_score

def log_prediction(input_text: str, prediction_score: float, latency_ms: float, actual_score: float):
    """
    Logs each prediction and its associated metadata for monitoring.
    """
    prediction_log.append({
        "timestamp": time.time(),
        "input": input_text,
        "prediction": prediction_score,
        "latency_ms": latency_ms,
        "actual": actual_score
    })
    # Keep the log size manageable
    if len(prediction_log) > MONITORING_WINDOW_SIZE * 2:
        prediction_log.pop(0)

def analyze_metrics():
    """
    Analyzes recent predictions to calculate performance metrics and detect potential issues.
    This function simulates a monitoring dashboard or alerting system.
    """
    if len(prediction_log) < MONITORING_WINDOW_SIZE:
        print(f"  [MONITORING] Not enough data for full analysis ({len(prediction_log)}/{MONITORING_WINDOW_SIZE})")
        return

    recent_logs = prediction_log[-MONITORING_WINDOW_SIZE:]

    # Calculate latency
    latencies = [log["latency_ms"] for log in recent_logs]
    avg_latency = statistics.mean(latencies)

    # Calculate prediction error (Mean Absolute Error)
    errors = [abs(log["prediction"] - log["actual"]) for log in recent_logs]
    mean_abs_error = statistics.mean(errors)

    # Calculate average predicted sentiment
    avg_predicted_sentiment = statistics.mean([log["prediction"] for log in recent_logs])

    # Calculate average actual sentiment
    avg_actual_sentiment = statistics.mean([log["actual"] for log in recent_logs])

    print(f"\n--- Monitoring Report (Last {MONITORING_WINDOW_SIZE} predictions) ---")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Mean Absolute Error (MAE): {mean_abs_error:.2f}") # Illustrates model performance monitoring
    print(f"  Average Predicted Sentiment: {avg_predicted_sentiment:.2f}")
    print(f"  Average Actual Sentiment: {avg_actual_sentiment:.2f}")

    # --- Basic Drift Detection ---
    # This is a simplified example of how drift might be detected.
    # In a real system, this would involve statistical tests,
    # distribution comparisons, or more sophisticated algorithms.

    # Model Performance Drift (e.g., accuracy degradation)
    if mean_abs_error > ERROR_THRESHOLD:
        print(f"  [ALERT] High MAE detected! Model performance might be degrading. (MAE: {mean_abs_error:.2f} > {ERROR_THRESHOLD:.2f})")
        # In a real system, this would trigger an alert (email, PagerDuty, etc.) for maintenance.

    # Data/Concept Drift (e.g., input data distribution or target concept changes)
    # Here, we check if the average predicted sentiment significantly deviates from the average actual sentiment.
    sentiment_difference = abs(avg_predicted_sentiment - avg_actual_sentiment)
    if sentiment_difference > AVG_SENTIMENT_THRESHOLD_CHANGE:
        print(f"  [ALERT] Significant sentiment concept drift detected! "
              f"Avg Predicted ({avg_predicted_sentiment:.2f}) vs Avg Actual ({avg_actual_sentiment:.2f}). "
              f"Difference: {sentiment_difference:.2f} > {AVG_SENTIMENT_THRESHOLD_CHANGE:.2f}")
        # This could indicate that the model's understanding of "positive" or "negative"
        # no longer aligns with the real-world sentiment, requiring model retraining or update.

    print("--------------------------------------------------")

# --- Main Simulation Loop ---
if __name__ == "__main__":
    print("Starting AI Model Production Monitoring Simulation...")
    print("Simulating requests to a sentiment analysis model.")
    print("Watch for monitoring reports and potential alerts.\n")

    sample_texts = [
        "Bu ürün harika, çok beğendim!", # Positive
        "Servis kötüydü, hayal kırıklığına uğradım.", # Negative
        "Fena değil, idare eder.", # Neutral
        "Çok başarılı bir çalışma olmuş.", # Positive
        "Kesinlikle tavsiye etmem, sorunlu.", # Negative
        "Genel olarak iyi bir deneyimdi.", # Positive
        "Beklentilerimi karşılamadı.", # Negative
        "Muhteşem bir ürün, tekrar alırım.", # Positive
        "Vasatın altında bir performans.", # Negative
        "Her şey yolunda gitti.", # Positive
        "Bu yeni özellik çok kafa karıştırıcı.", # Negative (new)
        "Eski versiyon daha iyiydi.", # Negative (new)
        "Harika bir güncelleme!", # Positive
        "Çok yavaş çalışıyor.", # Negative
        "Beklediğimden daha iyi çıktı." # Positive
    ]

    current_phase = "normal_operation"
    for i in range(1, len(sample_texts) * 2 + 1): # Simulate more requests than available texts
        text_index = (i - 1) % len(sample_texts)
        input_text = sample_texts[text_index]

        # Introduce a "drift phase" after some initial normal operation
        if i == len(sample_texts) + 1:
            current_phase = "drift_phase"
            print("\n--- Entering DRIFT PHASE: Simulating a change in real-world sentiment or data distribution ---")
            print("  (e.g., new trends, user base changes, model becoming outdated)\n")

        start_time = time.perf_counter()
        prediction = predict_sentiment(input_text) # Model makes a prediction
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        actual_sentiment = simulate_actual_sentiment(input_text, current_phase) # Get simulated ground truth

        log_prediction(input_text, prediction, latency_ms, actual_sentiment) # Log for monitoring

        print(f"Request {i:2d}: '{input_text[:30]}...' -> Pred: {prediction:.2f}, Actual: {actual_sentiment:.2f}, Latency: {latency_ms:.2f}ms")

        if i % (MONITORING_WINDOW_SIZE // 2) == 0: # Analyze metrics periodically
            analyze_metrics()

        time.sleep(0.1) # Simulate some delay between requests

    print("\nSimulation finished.")
