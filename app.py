import gradio as gr
import numpy as np

def synthesize_speech(text, speaker_id=0):
    """
    Placeholder function for speech synthesis
    Replace this with actual model inference when you have trained models
    """
    if not text.strip():
        return None
    
    # This is a placeholder - replace with actual model inference
    sample_rate = 24000
    duration = max(1.0, len(text) * 0.08)  # rough estimate
    samples = int(sample_rate * duration)
    
    # Generate simple sine wave as placeholder
    t = np.linspace(0, duration, samples)
    frequency = 440 + (speaker_id * 50)  # vary frequency by speaker
    
    # Create a more interesting waveform
    audio = (
        0.3 * np.sin(2 * np.pi * frequency * t) * np.exp(-t/(duration*0.8)) +
        0.1 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t/duration) +
        0.05 * np.random.randn(samples)  # add some noise
    )
    
    # Apply fade in/out
    fade_samples = int(0.1 * sample_rate)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return (sample_rate, audio.astype(np.float32))

def create_demo():
    with gr.Blocks(title="Learnable-Speech Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎤 Learnable-Speech: High-Quality 24kHz Speech Synthesis
            
            An unofficial implementation based on improvements of CosyVoice with learnable encoder and DAC-VAE.
            
            > **⚠️ This is a demo interface with placeholder audio. To use the actual model, you need to train it first!**
            
            ## 🚀 How to Train Your Own Model:
            
            1. **Follow the [Training Guide](https://github.com/primepake/learnable-speech/blob/main/TRAINING_GUIDE.md)**
            2. **Use the provided training scripts** in the `scripts/` directory
            3. **Upload your trained models** to Hugging Face Hub
            4. **Replace the placeholder code** in this Space with your models
            
            ### Quick Start:
            ```bash
            # 1. Prepare your dataset
            ./scripts/prepare_data.sh
            
            # 2. Train the model
            ./scripts/train_full_pipeline.sh
            
            # 3. Upload to Hugging Face
            python scripts/upload_to_hf.py --username your_username
            ```
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
                
                with gr.Row():
                    speaker_slider = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="Speaker ID"
                    )
                    
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy"
                )
        
        with gr.Accordion("🎯 Training Status & Next Steps", open=True):
            gr.Markdown(
                """
                ### 📋 Current Status:
                - ✅ **Demo Interface**: Ready
                - ❌ **Trained Models**: Not available (placeholder audio only)
                - ❌ **Model Inference**: Not implemented yet
                
                ### 🔧 To Enable Real Speech Synthesis:
                1. **Train the models** using the provided pipeline
                2. **Upload trained checkpoints** to Hugging Face Hub  
                3. **Update the inference code** in `synthesize_speech()` function
                4. **Test with real model outputs**
                
                ### 📚 Resources:
                - [📖 Complete Training Guide](https://github.com/primepake/learnable-speech/blob/main/TRAINING_GUIDE.md)
                - [🛠️ Training Scripts](https://github.com/primepake/learnable-speech/tree/main/scripts)
                - [📄 Research Paper](https://arxiv.org/pdf/2505.07916)
                - [💻 GitHub Repository](https://github.com/primepake/learnable-speech)
                """
            )
            gr.Markdown(
                """
                ### Key Features
                - **24kHz Audio Support**: High-quality audio generation at 24kHz sampling rate
                - **Flow matching AE**: Flow matching training for autoencoders  
                - **Immiscible assignment**: Support immiscible adding noise while training
                - **Contrastive Flow matching**: Support Contrastive training
                
                ### Architecture
                **Stage 1**: Audio to Discrete Tokens - Converts raw audio into discrete representations using FSQ (S3Tokenizer)
                
                **Stage 2**: Discrete Tokens to Continuous Latent Space - Maps discrete tokens to continuous latent space using VAE
                
                ### Training Pipeline
                1. Extract discrete tokens using trained FSQ S3Tokenizer
                2. Generate continuous latent representations using trained DAC-VAE
                3. Train Stage 1: BPE tokens → Discrete FSQ  
                4. Train Stage 2: Discrete FSQ → DAC-VAE Continuous latent space
                
                ### Links
                - [GitHub Repository](https://github.com/primepake/learnable-speech)
                - [Technical Paper](https://arxiv.org/pdf/2505.07916)
                """
            )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["Hello everyone! I am here to tell you that Learnable-Speech is amazing!", 0],
                ["The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle.", 1],
                ["We propose Learnable-Speech, a new approach to neural text-to-speech synthesis.", 2],
                ["This implementation uses flow matching for high-quality 24kHz audio generation.", 3],
            ],
            inputs=[text_input, speaker_slider],
            outputs=audio_output,
            fn=synthesize_speech,
            cache_examples=False,
        )
        
        generate_btn.click(
            fn=synthesize_speech,
            inputs=[text_input, speaker_slider],
            outputs=audio_output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )
