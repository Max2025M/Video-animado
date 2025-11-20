import os
import glob
import gdown
from subprocess import run

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1️⃣ Procurar arquivo de imagem na raiz
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(ext))

if not image_files:
    raise FileNotFoundError("Nenhuma imagem encontrada na raiz do projeto (.jpg, .jpeg, .png).")
face_path = image_files[0]  # pegar a primeira imagem encontrada

# 2️⃣ Procurar arquivo de áudio na raiz
audio_extensions = ["*.wav", "*.mp3", "*.m4a"]
audio_files = []
for ext in audio_extensions:
    audio_files.extend(glob.glob(ext))

if not audio_files:
    raise FileNotFoundError("Nenhum áudio encontrado na raiz do projeto (.wav, .mp3, .m4a).")
audio_path = audio_files[0]  # pegar o primeiro áudio encontrado

# Arquivos temporários
wav2lip_out = os.path.join(OUTPUT_DIR, "wav2lip.mp4")
final_out = os.path.join(OUTPUT_DIR, "result.mp4")

# 3️⃣ Baixar modelos
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

# 4️⃣ Clonar Wav2Lip e rodar boca sincronizada
if not os.path.exists("Wav2Lip"):
    run("git clone https://github.com/Rudrabha/Wav2Lip.git", shell=True)

run(
    f"python Wav2Lip/inference.py --checkpoint_path {wav2lip_model} "
    f"--face {face_path} --audio {audio_path} --outfile {wav2lip_out}",
    shell=True
)

# 5️⃣ Clonar FOMM e rodar movimentos completos
if not os.path.exists("first-order-model"):
    run("git clone https://github.com/AliaksandrSiarohin/first-order-model.git", shell=True)

run(
    f"python first-order-model/demo.py "
    f"--config first-order-model/config/vox-256.yaml "
    f"--driving_video {wav2lip_out} "
    f"--source_image {face_path} "
    f"--checkpoint {fomm_model} "
    f"--result {final_out}",
    shell=True
)

# 6️⃣ Reinsere o áudio original (ffmpeg)
run(
    f"ffmpeg -y -i {final_out} -i {audio_path} -c:v copy -c:a aac -strict experimental {final_out}_with_audio.mp4",
    shell=True
)

print("Avatar final gerado com movimentos completos em:", final_out + "_with_audio.mp4")
