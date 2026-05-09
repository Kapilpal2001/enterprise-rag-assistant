import edge_tts
import os
import tempfile

async def talk(text, path):
    await edge_tts.Communicate(text, "en-US-AriaNeural").save(path)

def transcribe_audio(audio_data, groq_client):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t_file:
        t_file.write(audio_data.getvalue())
        temp_path = t_file.name
        
    try:
        with open(temp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_path, f.read()), 
                model="whisper-large-v3"
            )
            return transcription.text
    finally:
        os.unlink(temp_path)
