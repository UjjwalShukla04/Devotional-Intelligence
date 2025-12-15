import gradio as gr
from src.pipeline import InferencePipeline


pipeline = InferencePipeline()


def analyze(audio, text):
    transcription = ""
    if text and text.strip():
        transcription = text.strip()
    elif audio:
        transcription = pipeline.transcribe(audio)
    else:
        return "", {"label": "", "confidence": 0.0}, {"label": "", "confidence": 0.0}, {"label": "", "confidence": 0.0}
    preds = pipeline.analyze_text(transcription)
    sentiment_card = f"Sentiment: {preds['sentiment']['label']} ({preds['sentiment']['confidence']:.2f})"
    toxicity_card = f"Toxicity: {preds['toxicity']['label']} ({preds['toxicity']['confidence']:.2f})"
    topic_card = f"Topic: {preds['topic']['label']} ({preds['topic']['confidence']:.2f})"
    return transcription, sentiment_card, toxicity_card, topic_card


with gr.Blocks(title="Krishna Ji: Devotional Intelligence Dashboard") as demo:
    gr.Markdown("# Krishna Ji: Devotional Intelligence Dashboard")
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or upload audio")
        text_in = gr.Textbox(lines=4, label="Or type text")
    analyze_btn = gr.Button("Analyze")
    with gr.Row():
        transcript_out = gr.Textbox(label="Transcription")
    with gr.Row():
        sentiment_out = gr.Markdown()
        toxicity_out = gr.Markdown()
        topic_out = gr.Markdown()
    analyze_btn.click(analyze, inputs=[audio_in, text_in], outputs=[transcript_out, sentiment_out, toxicity_out, topic_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
