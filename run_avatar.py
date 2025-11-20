import os
import gdown
from subprocess import run

INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

face_path = os.path.join(INPUT_DIR, "face.jpg")
audio_path = os.path.join(INPUT_DIR, "audio.wav")
wav2lip_out = os.path.join(OUTPUT_DIR, "wav2lip.mp4")
final_out = os.path.join(OUTPUT_DIR, "result.mp4")

# 1️⃣ Baixar modelos
wav2lip_model = "Wav2Lip.pth"
fomm_model = "fomm.pth"

if not os.path.exists(wav2lip_model):
    gdown.download(
        "https://drive.google.com/uc?id=1z2ZbTjzRJhciQZZ1x7Uocbxw4qpH8F8E",
        wav2lip_model,
        quiet=False
    )

if not os.path.exists(fomm_model):
    gdown.download(
        "https://drive.google.com/uc?id=1tQd9VfK5S0wUkq3X3y_MhRxWcPhQXh6r",
        fomm_model,
        quiet=False
    )

# 2️⃣ Clonar Wav2Lip e rodar boca sincronizada
if not os.path.exists("Wav2Lip"):
    run("git clone https://github.com/Rudrabha/Wav2Lip.git", shell=True)

run(
    f"python Wav2Lip/inference.py --checkpoint_path {wav2lip_model} "
    f"--face {face_path} --audio {audio_path} --outfile {wav2lip_out}",
    shell=True
)

# 3️⃣ Clonar FOMM e rodar movimentos faciais e corporais
if not os.path.exists("first-order-model"):
    run("git clone https://github.com/AliaksandrSiarohin/first-order-model.git", shell=True)

# Usamos o vídeo gerado pelo Wav2Lip como driving video → todos os movimentos seguem o áudio
run(
    f"python first-order-model/demo.py "
    f"--config first-order-model/config/vox-256.yaml "
    f"--driving_video {wav2lip_out} "
    f"--source_image {face_path} "
    f"--checkpoint {fomm_model} "
    f"--result {final_out}",
    shell=True
)

# 4️⃣ Reinsere o áudio original usando ffmpeg (caso FOMM remova o áudio)
run(
    f"ffmpeg -y -i {final_out} -i {audio_path} -c:v copy -c:a aac -strict experimental {final_out}_with_audio.mp4",
    shell=True
)

print("Avatar final gerado em:", final_out + "_with_audio.mp4")
