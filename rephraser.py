from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()

# Define selectable options for the dropdown
OPTIONS = [
    "Formal",
    "Informal",
    "Professional",
    "Friendly",
    "Polite",
    "Casual",
    "Sarcastic",
    "Serious",
    "Humorous",
    "Optimistic",
    "Pessimistic",
    "Respectful",
    "Sympathetic",
    "Empathetic",
    "Assertive",
    "Confident",
    "Encouraging",
    "Apologetic",
    "Urgent",
    "Calm",
    "Neutral",
    "Excited",
    "Curious",
    "Cautious",
    "Passionate",
    "Playful",
    "Inspirational",
    "Direct",
    "Tactful",
    "Commanding",
    "Reflective",
    "Witty",
]

llm_client = OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai"
)


def query_llm(prompt: str, selected_options: list[str]):
    try:
        # Construct system prompt from selected options
        system_instructions = f"""
            Rephrase this user input --> {prompt} to be grammatically correct and sound in a \
            {", ".join(selected_options) if selected_options else "neutral and helpful"} tone.
            Only return the rephrased text without any additional commentary.
            """

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt},
        ]

        response = llm_client.chat.completions.create(
            model="sonar",  # or any other supported model
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            stream=True,
        )

        full_response = ""
        for chunk in response:
            if getattr(chunk.choices[0].delta, "content", None) is not None:
                full_response += chunk.choices[0].delta.content
                yield full_response
    except Exception as e:
        yield f"Error: {str(e)}"


def reset_fields():
    # Return default/empty values for each component in order.
    return "", []


def start_gradio_client():
    try:
        with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as gradio_client:
            gr.Markdown("# Rephraser Assistant")
            user_input = gr.Textbox(
                lines=2, label="User Input", placeholder="Type your text here..."
            )
            options = gr.CheckboxGroup(
                choices=OPTIONS, label="User Tone (select one or more)"
            )
            submit = gr.Button("Rephrase")
            clear = gr.Button("Clear")
            output = gr.Textbox(
                lines=10,
                label="Rephrased Output",
                placeholder="Rephrased text will appear here...",
                show_copy_button=True,
                show_label="Rephrased Output",
            )

            submit.click(fn=query_llm, inputs=[user_input, options], outputs=output)

            # When "Rephrase" is clicked, reset the input fields
            clear.click(fn=reset_fields, inputs=[], outputs=[user_input, options])

        gradio_client.launch(debug=True, inbrowser=True)
    except Exception as e:
        print(f"Error starting Gradio client: {str(e)}")


if __name__ == "__main__":
    start_gradio_client()
