import whisper
import json
import os

model = whisper.load_model("medium")

os.makedirs('jsons', exist_ok=True)

audios = os.listdir('audios')
for audio in audios:
    # print(audio)
    parts = audio.split('_', 1)        # split only on FIRST underscore
    audio_number = parts[0]            # '01'
    audio_title = parts[1].replace('_', ' ')[:-4]   # 'what is data science.mp3'
    print(audio_number, " ", audio_title)

    # result = model.transcribe(audio=f"audios/Sample.mp3",
    result = model.transcribe(audio=f"audios/{audio}",
                              task="translate",
                              word_timestamps=False,
                              fp16=False,
                              verbose=True,)
    
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "video_number": audio_number,
            "video_title": audio_title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f)