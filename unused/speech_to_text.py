# Speech to Text using Whisper model
import whisper
import json
# model = whisper.load_model("base")
model = whisper.load_model("medium")

result = model.transcribe(audio="audios/Sample.mp3",
                          language="hi",
                          task="translate",
                          word_timestamps=False,
                          fp16=False,
                          verbose=True,)

# print(result["segments"])

chunks = []
for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f)