import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class CSVQABot:
    def __init__(self):
        self.csv_data = None
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def load_csv(self, file):
        """Load CSV file and prepare TF-IDF vectors"""
        try:
            self.csv_data = pd.read_csv(file.name)

            # Check for text columns
            text_columns = [col for col in self.csv_data.columns if self.csv_data[col].dtype == 'object']
            if not text_columns:
                return "Error: CSV must contain at least one text column", None

            # Combine text columns
            self.csv_data['combined_text'] = self.csv_data[text_columns].apply(
                lambda x: ' '.join(x.dropna().astype(str)), axis=1)

            # Create TF-IDF vectors
            self.tfidf_matrix = self.vectorizer.fit_transform(self.csv_data['combined_text'])

            return f"CSV loaded with {len(self.csv_data)} rows", self.csv_data.head(3).to_string()
        except Exception as e:
            return f"Error loading CSV: {str(e)}", None

    def find_relevant_info(self, question, top_k=3):
        """Find relevant information using TF-IDF cosine similarity"""
        if self.csv_data is None or self.tfidf_matrix is None:
            return "No CSV data loaded"

        # Vectorize the question
        question_vec = self.vectorizer.transform([question])

        # Calculate similarities
        similarities = cosine_similarity(question_vec, self.tfidf_matrix)[0]

        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.csv_data.iloc[idx]['combined_text'] for idx in top_indices]

    def generate_response(self, question):
        """Generate response based on CSV content"""
        relevant_info = self.find_relevant_info(question)

        if isinstance(relevant_info, str):
            return relevant_info

        response = "Here's what I found in the CSV:\n\n"
        for i, info in enumerate(relevant_info, 1):
            response += f"{i}. {info}\n\n"

        return response


# Initialize the bot
bot = CSVQABot()

# Create Gradio interface with modern message format
with gr.Blocks() as demo:
    gr.Markdown("# CSV Q&A Bot")
    gr.Markdown("Upload a CSV file and ask questions about its content")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            load_btn = gr.Button("Load CSV")
            csv_preview = gr.Textbox(label="CSV Preview", interactive=False)

        with gr.Column():
            # Updated to use the new messages format
            chatbot = gr.Chatbot(height=400, type="messages")
            msg = gr.Textbox(label="Your Question")
            clear = gr.Button("Clear Chat")


    def user(user_message, history):
        # Return user message with user role
        return "", history + [{"role": "user", "content": user_message}]


    def bot_response(history):
        user_message = history[-1]["content"]

        if user_message.lower() == "quit":
            bot_message = "Goodbye!"
        else:
            bot_message = bot.generate_response(user_message)

        # Return assistant response with assistant role
        return history + [{"role": "assistant", "content": bot_message}]


    load_btn.click(
        bot.load_csv,
        inputs=file_input,
        outputs=[gr.Textbox(label="Status"), csv_preview]
    )

    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        bot_response, chatbot, chatbot
    )
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()