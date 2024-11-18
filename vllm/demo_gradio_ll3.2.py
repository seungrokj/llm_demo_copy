import argparse

import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8011/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=False,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8022)
args = parser.parse_args()

openai_api_key = "EMPTY"
openai_api_base = args.model_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# context one-shot
#SYSTEM_COMMAND = {"role": "user", "content": "Context: date: November 2024; location: India Mumbai; running on: 8 AMD Instinct MI300 GPU; model name: Llama 3.2 90B multimodal. Only provide these information if asked. You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly and politely."}
SYSTEM_COMMAND = {"role": "user", "content": "Context: You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly and politely."}

def predict(message, history, temperature, max_tokens):
    image_url = ""
    if len(message["files"]) != 0:
        image_url = message["files"][0]["url"]

    history_openai_format = [SYSTEM_COMMAND]
    history_openai_format.append({
        "role": "user",
        "content": "You are a great ai assistant."
    })

    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({
            "role": "assistant", "content": assistant
        })

    history_openai_format.append({"role": "user", "content": message["text"]})

    #https://platform.openai.com/docs/api-reference/completions/create
    if image_url is not None:
        messages=[{
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message["text"]
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }]
    else:
            messages=history_openai_format

    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        extra_body={
            'repetition_penalty':
            1,
            'stop_token_ids': [
                int(id.strip()) for id in args.stop_token_ids.split(',')
                if id.strip()
            ] if args.stop_token_ids else []
        })

    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message

gr.set_static_paths(paths=["assets/"])
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
            """
            # Welcome to the Multi-modal Chatbot powered by AMD MI300X GPU and Llama-3.2-90B-Vision 🌟
            """
            )

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.9,
            step=0.1,
            interactive=True,
            label="Temperature",
            visible=True,
        )
        max_tokens = gr.Slider(
            minimum=10,
            maximum=4096,
            value=256,
            step=256,
            interactive=True,
            label="Max output tokens",
            visible=True,
        )

    chatbot = gr.Chatbot(
            placeholder="Nice to meet you today!",
            avatar_images=["assets/avatar_user.jpg","assets/avatar_amd.jpg"],
            height=1200,
            )
    gr.ChatInterface(
        fn=predict,
        multimodal=True,
        chatbot=chatbot,
        additional_inputs=[temperature, max_tokens],
    )

demo.queue().launch(server_name=args.host,
                    server_port=args.port,
                    share=True)
