# 📰 Fake News Detection & Summarizer

## 📘 About This Project
This project is a Python-based application that uses machine learning to detect fake news and summarize news articles. It includes a training script using MiniLM embeddings and logistic regression, and a user-friendly Tkinter GUI for real-time analysis and interaction.

---

📌 How to Run the Fake News Detection + Summarizer Project

IMPORTANT: 📁 Always open the **entire project folder** in your code editor (e.g., VS Code) using the **Open Folder** option.
This ensures the project runs with the correct file path, allowing it to find files like "Fake.csv" and "True.csv" without errors.
❌ Do NOT just double-click individual files to open — it may cause file-not-found issues when running the code.

1. Prepare your data:
   - Make sure the following files are placed in the **same folder** as your code:
     • `Fake.csv` (fake news dataset)
     • `True.csv` or `Real.csv` (real news dataset)

2. Install required Python packages:
   Run this command in your terminal:
	pip install pandas scikit-learn joblib sentence-transformers transformers torch requests
(Note: You do **not** need to install NumPy for this project.)

3. Run the model training script:
- Ensure you are **connected to the internet** — the script will download the MiniLM model.
- Then run:
  ```
  python model_train.py
  ```
- This will:
  • Load and embed the news data
  • Train a Logistic Regression classifier
  • Display the model's accuracy
  • Save the trained model and embedder locally

4. Launch the GUI:
- After training is complete, run:
  ```
  python news_app.py
  ```
- This opens a Tkinter GUI where you can:
  • Fetch live news articles
  • Summarize them
  • Detect if the news is real or fake
  • Optionally save results to a text file

✅ All set! You can now explore and analyze news effortlessly through a simple interface.
