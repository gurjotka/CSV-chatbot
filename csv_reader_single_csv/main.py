import pandas as pd
import ollama
import gradio as gr


def analyze_data(df, question):
    # First get basic stats about your data
    data_info = f"""
    The dataset contains {len(df)} rows with these columns: {list(df.columns)}.
    Here are the first 3 rows: {df.head(3).to_dict()},
    Here is the data {df}
    """

    full_prompt = f"{data_info}\n\nUser question: {question}"

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': full_prompt}]
    )
    return response['message']['content']


def process_csv_and_query(csv_file, question):
    try:
        df = pd.read_csv(csv_file.name)
        return analyze_data(df, question)
    except Exception as e:
        return f"Error processing file: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# Csv reader and chatbot ")
    gr.Markdown("Upload a CSV file containing student mental health data and ask questions about it.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            question_input = gr.Textbox(label="Your Question",
                                        placeholder="Ask about the data...")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Analysis Results", interactive=False)

    submit_btn.click(
        fn=process_csv_and_query,
        inputs=[file_input, question_input],
        outputs=output
    )

    # Example questions for user guidance
    gr.Examples(
        examples=[
            ["What is the average stress level among students?"],
            ["What is the data about?"],
            ["How does water level calculated?"]
        ],
        inputs=[question_input],
        outputs=output,
        cache_examples=False,
        label="Example Questions"
    )

if __name__ == "__main__":
    demo.launch()