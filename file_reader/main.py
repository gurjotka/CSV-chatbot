import pandas as pd
import ollama
import gradio as gr


def df_to_dict_string(df, max_rows=3):
    """Convert DataFrame to a readable dictionary string with limited rows"""
    return str(df.head(max_rows).to_dict(orient='records'))


def analyze_data(df, question):
    # Convert data to a readable format for the LLM
    data_info = f"""
    Dataset contains {len(df)} rows with columns: {list(df.columns)}
    First {min(3, len(df))} rows as dictionary:
    {df}
    """

    full_prompt = f"{data_info}\n\nQuestion: {question}\n\nProvide a detailed analysis:"

    try:
        response = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


def process_input(uploaded_file, question):
    try:
        # Get the file extension
        file_extension = uploaded_file.split('.')[-1].lower()

        # Read file based on extension
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, header = None)
        else:
            return "Unsupported file format. Please upload a CSV or Excel file."

        return analyze_data(df, question)
    except Exception as e:
        return f"Error processing file: {str(e)}"


with gr.Blocks(title="File Analyzer") as demo:
    gr.Markdown("## 📊 File Analysis Tool")
    gr.Markdown("Upload a CSV or Excel file and ask questions about student mental health data")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload CSV or Excel File",
                file_types=[".csv", ".xlsx", ".xls"],  # All supported extensions
                type="filepath"  # Changed from "file" to "filepath"
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="E.g., What's the average stress level by gender?",
                lines=3
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Analysis Results",
                interactive=False,
                lines=10,
                show_copy_button=True
            )

    submit_btn.click(
        fn=process_input,
        inputs=[file_input, question_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()