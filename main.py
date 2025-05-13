# -*- coding: utf-8 -*-
import os
import sys
import logging
import re
import torch
from pathlib import Path
from datetime import timedelta
from pydub import effects
from gtts import gTTS
from pydub import AudioSegment, effects
from pydub.utils import which
from PIL import Image, ImageDraw, ImageFont
# Handle Pillow rename: alias ANTIALIAS for Resampling
if not hasattr(Image, 'ANTIALIAS') and hasattr(Image, 'Resampling'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
from moviepy.editor import ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips
from transformers import VitsTokenizer, VitsModel
import scipy.io.wavfile as wavfile
import srt
import random

# Add first-order-model directory to path for FOMM imports
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "first-order-model"))

# FOMM imports
demo_path = BASE_DIR / "first-order-model" / "demo.py"
if (BASE_DIR / "first-order-model").exists() and (BASE_DIR / "first-order-model").is_dir():
    from demo import load_checkpoints, make_animation
    import imageio
    import numpy as np
else:
    load_checkpoints = None
    make_animation = None

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
# FFmpeg for pydub
AudioSegment.converter = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
if not AudioSegment.converter:
    raise EnvironmentError("FFmpeg not found. Please install FFmpeg.")

# Paths & Model settings
FONT_PATH = "asset/THSarabunNew.ttf"
MUSIC_PATH = "bg_music/calm_story1.mp3"
END_IMAGE = "asset/end.png"
OUTPUT_DIR = "output"
# Prosody settings
SPEED_FACTOR = 0.85  # slower speaking
PAUSE_MS = 500       # ms pause before and after narration
# FOMM settings
DRIVER_VIDEO = BASE_DIR / "first-order-model" / "driver.mp4"
CONFIG_PATH = BASE_DIR / "first-order-model" / "config" / "vox-256.yaml"
CHECKPOINT_PATH = BASE_DIR / "first-order-model" / "checkpoints" / "vox-256.pth"
ANIMATE = False

# Initialize FOMM if available
if load_checkpoints and DRIVER_VIDEO.exists():
    logging.info("Loading FOMM model for animation...")
    generator, kp_detector = load_checkpoints(
        str(CONFIG_PATH),
        str(CHECKPOINT_PATH)
    )
    generator.to("cpu"); kp_detector.to("cpu")
    generator.eval(); kp_detector.eval()
    ANIMATE = True
else:
    logging.warning("Driver video or FOMM not found; animation disabled.")

# Load Thai TTS model
logging.info("Loading Thai VITS TTS model...")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-tha")
model = VitsModel.from_pretrained(
    "facebook/mms-tts-tha",
    device_map={"": "cpu"},
    torch_dtype=torch.float32
)

# Section prefixes
SECTION_PREFIXES = {
    "🎬 Hook:": "Hook",
    "📛 Name:": "Name",
    "🐢 Story": "Story",
    "💡 Moral:": "Moral",
    "🔔 CTA:": "CTA"
}

# Audio helpers
def change_speed(sound: AudioSegment, speed: float) -> AudioSegment:
    new_frame_rate = int(sound.frame_rate * speed)
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(sound.frame_rate)

def shift_pitch(sound: AudioSegment, semitones: float) -> AudioSegment:
    factor = 2 ** (semitones / 12)
    new_frame_rate = int(sound.frame_rate * factor)
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(sound.frame_rate)

# Animate with FOMM or fallback to static
def get_scene_clip(img_path: str, audio_clip: AudioFileClip):
    if ANIMATE:
        # animate
        anim_out = img_path.replace('.png', '_anim.mp4')
        source_image = imageio.imread(img_path)
        reader = imageio.get_reader(str(DRIVER_VIDEO))
        fps = reader.get_meta_data()["fps"]
        writer = imageio.get_writer(anim_out, fps=fps)
        for frame in reader:
            # animated = make_animation(
            #     source_image, [frame], kp_detector, generator,
            animated = make_animation(
                source_image, [frame], generator, kp_detector,
                relative=True, adapt_movement_scale=True
            )
            # เอาเฟรมแรกออกมาแล้วคูณสเกล แปลงเป็น uint8
            frame_res = animated[0]
            writer.append_data((frame_res * 255).astype(np.uint8))
        writer.close()
        clip = VideoFileClip(anim_out).set_duration(audio_clip.duration)
    else:
        # static image
        clip = ImageClip(img_path).set_duration(audio_clip.duration)
    return clip.set_audio(audio_clip)

# Read story into sections
def read_story_sections(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Story file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    sections = {}
    current_key = None
    story_counter = 1
    for line in lines:
        if any(line.startswith(p) for p in SECTION_PREFIXES):
            if line.startswith("🐢 Story"):
                current_key = f"Story{story_counter}"
                story_counter += 1
            else:
                current_key = SECTION_PREFIXES[line]
            sections[current_key] = ""
        elif current_key:
            sections[current_key] = (sections[current_key] + " " + line).strip()
    return sections

# Subtitle generation
def generate_subtitles(sections, audio_paths, out_path):
    subs, idx, t0 = [], 1, 0.0
    keys = list(sections.keys())
    for i, ap in enumerate(audio_paths):
        dur = AudioSegment.from_file(ap).duration_seconds
        words = sections[keys[i]].split()
        chunk = max(5, len(words)//max(1, int(dur//2.5)))
        parts = [" ".join(words[j:j+chunk]) for j in range(0, len(words), chunk)]
        for part in parts:
            start = timedelta(seconds=t0)
            end = timedelta(seconds=t0 + dur/len(parts))
            subs.append(srt.Subtitle(index=idx, start=start, end=end, content=part))
            t0 += dur/len(parts)
            idx += 1
    with open(os.path.join(out_path, "subtitle.srt"), "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

# Thumbnail creation
def create_thumbnail(text, font_path, out_path):
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found: {font_path}")
    base = Image.new("RGB", (1280, 720), color=(255, 255, 255))
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(font_path, 72)
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((1280-w)/2, (720-h)/2), text, font=font, fill=(0, 0, 0))
    base.save(out_path)

# Synthesize speech (Thai via VITS, English via gTTS)
def synthesize_local(text: str, out_path: str):
    if re.search(r"[A-Za-z]", text):
        tts = gTTS(text=text, lang='en')
        tmp = out_path.replace('.wav', '_en.mp3')
        tts.save(tmp)
        AudioSegment.from_mp3(tmp).export(out_path, format='wav')
        os.remove(tmp)
    else:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        wavfile.write(out_path, model.config.sampling_rate, outputs.waveform[0].cpu().numpy())

# Main pipeline
def process_story(file_path):
    sections = read_story_sections(file_path)
    title = Path(file_path).stem
    out = os.path.join(OUTPUT_DIR, title)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 1) TTS synthesis
    audio_wavs = []
    logging.info("Synthesizing speech...")
    for i, key in enumerate(sections):
        wav = os.path.join(out, f"raw_{i+1}.wav")
        synthesize_local(sections[key], wav)

        # โหลดเสียงกลับมาเพื่อปรับ prosody
        sound = AudioSegment.from_file(wav)

        # ← ตรงนี้ให้เพิ่ม 2 บรรทัดนี้ เพื่อสุ่ม variation ของความเร็ว
        factor = random.uniform(0.9, 1.1)
        sound = change_speed(sound, SPEED_FACTOR * factor)
        # ← จบส่วนเพิ่ม

        # ปรับระดับเสียงให้คงที่
        sound = effects.normalize(sound)

        # เพิ่มช่วงเงียบหน้า–หลัง
        silence = AudioSegment.silent(duration=PAUSE_MS)
        sound = silence + sound + silence

        # ถ้ามี bg music ให้ overlay ได้ที่นี่
        # bg = AudioSegment.from_file(MUSIC_PATH).apply_gain(-20)
        # sound = bg.overlay(sound, position=0)

        # เซฟกลับเป็นไฟล์ .wav
        sound.export(wav, format="wav")
        audio_wavs.append(wav)

    # 2) Animate & combine audio
    clips = []
    logging.info("Creating scene clips...")
    for i, wav in enumerate(audio_wavs):
        scene_img = os.path.join(img_dir, f"scene_{i+1}.png")
        ac = AudioFileClip(wav)
        clip = get_scene_clip(scene_img, ac)
        clips.append(clip.crossfadein(0.5).crossfadeout(0.5))
    if os.path.exists(END_IMAGE):
        ec = ImageClip(END_IMAGE).set_duration(3).resize(height=1920).set_position("center")
        clips.append(ec.crossfadein(0.5).crossfadeout(0.5))

    # 3) Subtitles and thumbnail
    generate_subtitles(sections, audio_wavs, out)
    create_thumbnail(title, FONT_PATH, os.path.join(out, "thumbnail.png"))

    # 4) Final concatenation
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(os.path.join(out, "final.mp4"), fps=24)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py story/your_story.txt")
    else:
        process_story(sys.argv[1])
