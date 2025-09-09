import gradio as gr
import numpy as np
import os

def synthesize_speech(text, speaker_id=0):
    """
    Placeholder function for speech synthesis
    Replace this with actual model inference when you have trained models
    """
    if not text.strip():
        return None
    
    sample_rate = 24000
    duration = max(1.0, len(text) * 0.08)  # rough estimate
    samples = int(sample_rate * duration)
    
    # Generate sine-based waveform
    t = np.linspace(0, duration, samples, endpoint=False)
    frequency = 440 + (speaker_id * 50)
    
    audio = (
        0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-t/(duration*0.8)) +
        0.1 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t/duration) +
        0.05 * np.random.randn(samples)
    )
    
    # Fade in/out safely
    fade_samples = min(int(0.1 * sample_rate), samples // 2)
    if fade_samples > 0:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return (sample_rate, audio.astype(np.float32))

def create_demo():
    with gr.Blocks(
        title="Learnable-Speech Demo",
        theme=gr.themes.Default(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🎤 Learnable-Speech: High-Quality 24kHz Speech Synthesis
            
            An unofficial implementation based on improvements of CosyVoice with learnable encoder and DAC-VAE.
            
            > **⚠️ This is a demo interface with placeholder audio. To use the actual model, you need to train it first!**
            """
        )
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Enter text here...",
                    lines=3,
                    value="Hello, this is a demo of Learnable-Speech synthesis."
                )
                
                speaker_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=1,
                    label="Speaker ID"
                )
                    
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy"
                )
        
        generate_btn.click(
            fn=synthesize_speech,
            inputs=[text_input, speaker_slider],
            outputs=audio_output
        )
        
        gr.Examples(
            examples=[
                ["Hello everyone! I am here to tell you that Learnable-Speech is amazing!"],
                ["The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle."],
                ["We propose Learnable-Speech, a new approach to neural text-to-speech synthesis."],
                ["This implementation uses flow matching for high-quality 24kHz audio generation."],
            ],
            inputs=[text_input],
        )
    
    return demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    demo = create_demo()
    
    try:
        demo.launch(
            server_name=host,
            server_port=port,
            share=False,
            show_error=True,
            quiet=False,
            enable_queue=True
        )
    except Exception:
        print(f"Failed to launch on {host}:{port}, trying with share=True")
        demo.launch(
            share=True,
            show_error=True,
            quiet=False,
            enable_queue=True
        )
