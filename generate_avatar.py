import os
import uuid
import base64
import tempfile
from pathlib import Path
import sys
import traceback

from pydub import AudioSegment
import numpy as np
import cv2
import moviepy.editor as mpy
import librosa
import soundfile as sf
import mediapipe as mp
import random

# ----------------------------
mp_face = mp.solutions.face_mesh

# ----------------------------
def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ----------------------------
def save_base64_file(base64_str, suffix):
    data = base64.b64decode(base64_str)
    path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        f.write(data)
    return path

# ----------------------------
def trim_audio(in_path, out_path, start_ms=0, end_ms=None):
    audio = AudioSegment.from_file(in_path)
    trimmed = audio[start_ms:] if end_ms is None else audio[start_ms:end_ms]
    trimmed.export(out_path, format="wav")
    return out_path

def amplitude_envelope(wav_path, sr=16000, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr)
    env = np.array([np.max(np.abs(y[i:i+hop_length])) for i in range(0, len(y), hop_length)])
    if env.max() > 0:
        env = env / env.max()
    return env, sr

# ----------------------------
def generate_animation(image_path, audio_path, out_path, fps=25,
                       mouth_amp=0.6, head_amp=3.0, eye_amp=0.2, brow_amp=1.5):
    try:
        log("Carregando imagem...")
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # Boca
                mouth_inds = [61, 291, 13, 14, 78, 308, 82, 312]
                xs = [int(lm[i].x*w) for i in mouth_inds]
                ys = [int(lm[i].y*h) for i in mouth_inds]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                pad_x = max(6, int((x2-x1)*0.3))
                pad_y = max(6, int((y2-y1)*0.6))
                mouth_box = (max(0,x1-pad_x), max(0,y1-pad_y), min(w,x2+pad_x), min(h,y2+pad_y))
                # Olhos
                left_eye_inds = [159,145]
                right_eye_inds = [386,374]
                left_eye_box = [int(lm[i].x*w) for i in left_eye_inds] + [int(lm[i].y*h) for i in left_eye_inds]
                right_eye_box = [int(lm[i].x*w) for i in right_eye_inds] + [int(lm[i].y*h) for i in right_eye_inds]
                # Sobrancelhas
                left_brow_inds = [70,63]
                right_brow_inds = [300,293]
                left_brow_box = [int(lm[i].x*w) for i in left_brow_inds] + [int(lm[i].y*h) for i in left_brow_inds]
                right_brow_box = [int(lm[i].x*w) for i in right_brow_inds] + [int(lm[i].y*h) for i in right_brow_inds]
            else:
                cx, cy = w//2, h//2
                mouth_box = (int(cx- w*0.15), int(cy + h*0.05), int(cx + w*0.15), int(cy + h*0.25))
                left_eye_box = right_eye_box = [cx-20,cx+20,cy-10,cy+10]
                left_brow_box = right_brow_box = [cx-20,cx+20,cy-30,cy-20]

        env, sr = amplitude_envelope(audio_path, sr=16000)
        audio_info = sf.info(audio_path)
        duration = audio_info.duration
        total_frames = max(1, int(duration*fps))
        frames = []
        env_frame = np.interp(np.linspace(0,len(env)-1,total_frames), np.arange(len(env)), env)

        log("Gerando frames do vídeo...")
        for i in range(total_frames):
            frame = img.copy()
            # Cabeça
            theta = np.deg2rad(head_amp * np.sin(2*np.pi*(i/total_frames)*1.2))
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            M_rot = np.array([[cos_t,-sin_t,0],[sin_t,cos_t,0]],dtype=np.float32)
            cx, cy = w//2, h//2
            M_rot[0,2] = (1 - cos_t)*cx - sin_t*cy
            M_rot[1,2] = sin_t*cx + (1 - cos_t)*cy
            frame = cv2.warpAffine(frame, M_rot, (w,h), borderMode=cv2.BORDER_REFLECT)

            # Boca
            x1,y1,x2,y2 = mouth_box
            mouth_roi = frame[y1:y2, x1:x2].copy()
            if mouth_roi.size !=0:
                mh, mw = mouth_roi.shape[:2]
                scale = 1.0 + mouth_amp*float(env_frame[i])
                new_h = max(2,int(mh*scale))
                resized = cv2.resize(mouth_roi,(mw,new_h),interpolation=cv2.INTER_LINEAR)
                ystart = max(0, y1 - (new_h-mh)//2)
                yend = ystart + new_h
                if yend <= frame.shape[0]:
                    overlay = frame.copy()
                    overlay[ystart:yend, x1:x2] = resized
                    frame = cv2.addWeighted(overlay, 0.95, frame, 0.05,0)

            # Piscar olhos
            blink_prob = 0.02 + 0.3*float(env_frame[i])
            blink = random.random() < blink_prob
            for eye_box in [left_eye_box, right_eye_box]:
                ex1, ex2, ey1, ey2 = eye_box[0], eye_box[1], eye_box[2], eye_box[3]
                if blink:
                    mid_y = (ey1+ey2)//2
                    frame[ey1:ey2,ex1:ex2] = frame[mid_y:mid_y+1, ex1:ex2]

            # Sobrancelhas
            for brow_box in [left_brow_box,right_brow_box]:
                bx1,bx2,by1,by2 = brow_box[0],brow_box[1],brow_box[2],brow_box[3]
                offset = int(env_frame[i]*brow_amp)
                frame[by1-offset:by2-offset,bx1:bx2] = frame[by1:by2,bx1:bx2]

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if i % 10 == 0:
                log(f"Progresso frames: {int((i+1)/total_frames*100)}%")

        log("Combinando frames e áudio...")
        clip = mpy.ImageSequenceClip(frames,fps=fps)
        clip = clip.set_audio(mpy.AudioFileClip(str(audio_path)))
        clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", verbose=False, logger=None)
        log("Vídeo finalizado com sucesso!")

    except Exception as e:
        log(f"Erro: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        try: os.remove(image_path)
        except: pass
        try: os.remove(audio_path)
        except: pass

# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera avatar animado a partir de imagem e áudio Base64")
    parser.add_argument("--image_b64", required=True, help="Imagem em Base64")
    parser.add_argument("--audio_b64", required=True, help="Áudio em Base64 (qualquer formato)")
    parser.add_argument("--start", type=float, default=0, help="Início do áudio em segundos")
    parser.add_argument("--end", type=float, default=0, help="Fim do áudio em segundos")
    parser.add_argument("--output", required=True, help="Caminho do vídeo de saída")
    args = parser.parse_args()

    log("Salvando arquivos temporários...")
    img_path = save_base64_file(args.image_b64, ".png")
    audio_path = save_base64_file(args.audio_b64, ".wav")

    if args.end > args.start:
        temp_audio = Path(tempfile.gettempdir()) / f"trim_{uuid.uuid4().hex}.wav"
        trim_audio(audio_path, temp_audio, int(args.start*1000), int(args.end*1000))
        os.remove(audio_path)
        audio_path = temp_audio

    generate_animation(img_path, audio_path, args.output)
