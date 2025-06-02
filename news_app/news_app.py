from tkinter import Tk, Label, Button, Text, WORD, END, messagebox, ttk, Frame
import requests, random, joblib, datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class NewsAnalyzerApp:
    """
    A GUI-based application that fetches news from NewsAPI, summarizes it,
    and predicts whether the article is real or fake using a trained ML model.
    """
    
    def __init__(self, root):
        """
        Initializes the NewsAnalyzerApp with GUI setup and ML model loading.

        Args:
            root (Tk): The root Tkinter window object.
        """
        self.root = root
        self.root.title("📰 News Analyzer (MiniLM)")
        self.root.geometry("800x650")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        # Load ML components
        self.model = joblib.load("minilm_fake_news_model.pkl")
        self.embedder = SentenceTransformer("minilm_embedder")
        self.summarizer = pipeline("summarization")
        self.api_key = "01e247576e92420bb6955e1a23faaff8"

        self.news_articles = []
        self.titles = []

        self.build_gui()
        self.refresh_news()

    def fetch_news(self):
        """
        Fetches random top US news articles from NewsAPI.

        Returns:
            list: A list of news articles with title and content.
        """
        try:
            page = random.randint(1, 5)
            url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=20&page={page}&apiKey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            raw = response.json().get("articles", [])
            articles = [a for a in raw if a.get("title") and a.get("content")]
            random.shuffle(articles)
            return articles[:10]
        except Exception as e:
            print("Error fetching news:", e)
            return []

    def build_gui(self):
        """
        Builds the graphical user interface with dropdown, buttons, summary box,
        and prediction label.
        """
        Label(self.root, text="Select a News Article", font=("Segoe UI", 14, "bold"),
              bg="#1e1e1e", fg="#e0e0e0").pack(pady=15)

        dropdown_frame = Frame(self.root, bg="#2c2c2c", padx=10, pady=10)
        dropdown_frame.pack(pady=5)

        self.article_dropdown = ttk.Combobox(dropdown_frame, width=85, state="readonly", font=("Segoe UI", 10))
        self.article_dropdown.pack()

        btn_frame = Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        Button(btn_frame, text="🔍 Analyze", font=("Segoe UI", 11), bg="#007bff", fg="white",
               padx=10, command=self.analyze).grid(row=0, column=0, padx=10)
        Button(btn_frame, text="🔄 Refresh News", font=("Segoe UI", 11), bg="#28a745", fg="white",
               padx=10, command=self.refresh_news).grid(row=0, column=1, padx=10)
        Button(btn_frame, text="💾 Save Result", font=("Segoe UI", 11), bg="#6f42c1", fg="white",
               padx=10, command=self.save_result).grid(row=0, column=2, padx=10)

        Label(self.root, text="Summary:", font=("Segoe UI", 12, "bold"),
              bg="#1e1e1e", fg="#e0e0e0").pack()
        self.summary_text = Text(self.root, height=7, width=90, wrap=WORD,
                                 font=("Segoe UI", 10), bg="#2c2c2c", fg="#e0e0e0", insertbackground="white")
        self.summary_text.pack(pady=5)

        self.prediction_label = Label(self.root, text="Prediction:", font=("Segoe UI", 16, "bold"),
                                      fg="#f5c518", bg="#1e1e1e")
        self.prediction_label.pack(pady=20)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TCombobox",
                        fieldbackground="white",
                        background="white",
                        foreground="black",
                        selectbackground="#e0e0e0",
                        selectforeground="black")

    def refresh_news(self):
        """
        Fetches a new set of news articles and updates the dropdown menu.
        Also resets the summary and prediction display.
        """
        self.news_articles = self.fetch_news()
        self.titles = [a['title'] for a in self.news_articles] or ["❌ Failed to load news."]
        self.article_dropdown["values"] = self.titles
        if self.titles:
            self.article_dropdown.current(0)
        self.summary_text.delete("1.0", END)
        self.prediction_label.config(text="Prediction:")

    def analyze(self):
        """
        Summarizes the selected news article and predicts whether it's real or fake.
        Updates the GUI with the result.
        """
        if not self.news_articles:
            messagebox.showerror("Error", "No news articles available.")
            return

        index = self.article_dropdown.current()
        if index < 0 or index >= len(self.news_articles):
            messagebox.showwarning("Warning", "Select a valid article.")
            return

        article = self.news_articles[index]
        full_text = article["title"] + ": " + article["content"]

        summary = self.summarizer(full_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        self.summary_text.delete("1.0", END)
        self.summary_text.insert(END, summary)

        embedding = self.embedder.encode([full_text])
        prediction = self.model.predict(embedding)
        self.result = "REAL ✅" if prediction[0] == 1 else "FAKE ❌"
        self.prediction_label.config(text=f"Prediction: {self.result}")

    def save_result(self):
        """
        Saves the current article's title, summary, prediction, and timestamp to a text file.
        Displays a success or error message based on the outcome.
        """
        try:
            index = self.article_dropdown.current()
            if not self.news_articles or index < 0:
                messagebox.showwarning("Warning", "Nothing to save.")
                return

            article = self.news_articles[index]
            summary = self.summary_text.get("1.0", END).strip()
            prediction = self.prediction_label["text"].replace("Prediction: ", "")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open("news_results.txt", "a", encoding="utf-8") as f:
                f.write(f"⏰ {timestamp}\n")
                f.write(f"📰 Title: {article['title']}\n")
                f.write(f"🔍 Summary: {summary}\n")
                f.write(f"🧠 Prediction: {prediction}\n")
                f.write("="*80 + "\n")

            messagebox.showinfo("Saved", "Result saved to news_results.txt ✅")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save result.\n{e}")

# Run the app
if __name__ == "__main__":
    root = Tk()
    app = NewsAnalyzerApp(root)
    root.mainloop()
