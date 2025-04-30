import pandas as pd
import ollama
import gradio as gr
from typing import List, Dict


def df_to_dict_string(df: pd.DataFrame, max_rows: int = 3) -> str:
    """Convert DataFrame to a readable dictionary string with limited rows"""
    return str(df.head(max_rows).to_dict(orient='records'))


def get_dataset_info(dfs: Dict[str, pd.DataFrame]) -> str:
    """Generate information about all uploaded datasets"""
    info = []
    for name, df in dfs.items():
        # Extract just the filename without path or extension
        clean_name = name.split('/')[-1].split('.')[0]
        info.append(f"Dataset '{clean_name}' contains {len(df)} rows with columns: {list(df.columns)}")
        info.append(f"First {min(3, len(df))} rows:\n{df_to_dict_string(df)}\n")
    return "\n".join(info)


def analyze_data(dfs: Dict[str, pd.DataFrame], question: str, chat_history: List[Dict]) -> str:
    """Analyze data using the LLM with context from multiple datasets"""
    data_info = get_dataset_info(dfs)

    # Include chat history in context (last 3 exchanges)
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]])

    full_prompt = (
        f"Available datasets:\n{data_info}\n\n"
        f"Conversation context:\n{context}\n\n"
        f"Question: {question}\n\n"
        # "Provide a detailed analysis considering all available datasets:"
    )

    try:
        response = ollama.chat(
            model='deepseek-coder',
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


def process_input(uploaded_files: List[str], question: str, chat_history: List[Dict]) -> tuple:
    """Process user question with multiple uploaded files"""
    try:
        if not uploaded_files:
            return "Please upload at least one file first.", chat_history

        dfs = {}
        for file in uploaded_files:
            file_extension = file.split('.')[-1].lower()
            name = file.split('/')[-1].split('.')[0]

            if file_extension == 'csv':
                dfs[name] = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                dfs[name] = pd.read_excel(file, header=None)
            else:
                return f"Unsupported file format for {file}. Please upload CSV or Excel files.", chat_history

        analysis = analyze_data(dfs, question, chat_history)
        chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": analysis}
        ])
        return "", chat_history
    except Exception as e:
        error_msg = f"Error processing files: {str(e)}"
        chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_msg}
        ])
        return "", chat_history


def handle_file_upload(uploaded_files: List[str]) -> tuple:
    """Process uploaded files and return preview"""
    try:
        if not uploaded_files:
            return None, "No files uploaded"

        dfs = {}
        previews = []
        for file in uploaded_files:
            file_extension = file.split('.')[-1].lower()
            name = file.split('/')[-1].split('.')[0]

            if file_extension == 'csv':
                dfs[name] = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                dfs[name] = pd.read_excel(file, header=None)
            else:
                return None, f"Unsupported file format for {file}. Please upload CSV or Excel files."

            previews.append(f"📄 {name} preview:\n{dfs[name].head(3).to_markdown()}\n")

        return dfs, "Files uploaded successfully!\n\n" + "\n".join(previews)
    except Exception as e:
        return None, f"Error reading files: {str(e)}"


with gr.Blocks(title="Multi-File Data Chatbot") as demo:
    gr.Markdown("## 🧠📂 Multi-File Data Chatbot")
    gr.Markdown("Upload multiple CSV/Excel files and chat with your combined data")

    # Store DataFrames in session state
    dfs_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload CSV or Excel Files",
                file_types=[".csv", ".xlsx", ".xls"],
                type="filepath",
                file_count="multiple"
            )
            file_output = gr.Textbox(label="Files Info", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400, label="Chat with your data", type="messages")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="E.g., Compare data between these datasets...",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    # File upload handling
    file_input.upload(
        fn=handle_file_upload,
        inputs=file_input,
        outputs=[dfs_state, file_output]
    )

    # Chat handling
    submit_btn.click(
        fn=process_input,
        inputs=[file_input, question_input, chatbot],
        outputs=[question_input, chatbot]
    )
    question_input.submit(
        fn=process_input,
        inputs=[file_input, question_input, chatbot],
        outputs=[question_input, chatbot]
    )

    # Clear chat button
    clear_btn.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False
    )

if __name__ == "__main__":
    demo.launch()