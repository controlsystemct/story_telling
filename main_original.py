
# -*- coding: utf-8 -*-
import os
import random
from gtts import gTTS
from dotenv import load_dotenv
from pydub import AudioSegment, effects
from pydub.utils import which
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pathlib import Path
import srt
from datetime import timedelta

# === CONFIG ===
AudioSegment.converter = which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
load_dotenv()
FONT_PATH = "asset/THSarabunNew.ttf"
MUSIC_DIR = "bg_music"
OUTPUT_DIR = "output"
TITLE_FRAME_DURATION = 3

EMOJI_SECTION_MAP = {
    "🎬 Hook": "Hook",
    "📛 Name": "Name",
    "🐢 Story": "Story",
    "💡 Moral": "Moral",
    "🔔 CTA": "CTA"
}
SECTION_KEYS = ["Hook", "Name", "Story", "Story2", "Moral", "CTA"]

def read_story_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    sections = {}
    current_key = None
    for line in lines:
        if line in EMOJI_SECTION_MAP:
            current_key = EMOJI_SECTION_MAP[line]
            sections[current_key] = ""
        elif current_key:
            if sections[current_key]:
                sections[current_key] += " " + line.strip()
            else:
                sections[current_key] = line.strip()
    if "Story" in sections:
        words = sections["Story"].split()
        half = len(words) // 2
        sections["Story"], sections["Story2"] = " ".join(words[:half]), " ".join(words[half:])
    return sections

def generate_subtitles(sections, audio_paths, out_path):
    subtitles = []
    index = 1
    start_time = 0.0
    keys = list(sections.keys())
    for i, audio_path in enumerate(audio_paths):
        duration = AudioSegment.from_mp3(audio_path).duration_seconds
        content = sections.get(keys[i], "").strip()
        start = timedelta(seconds=start_time)
        end = timedelta(seconds=start_time + duration)
        subtitles.append(srt.Subtitle(index=index, start=start, end=end, content=content))
        start_time += duration
        index += 1
    srt_path = os.path.join(out_path, "subtitle.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

def create_title_frame(text, output_path):
    base = Image.new("RGB", (1080, 1920), color=(255, 255, 255))
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(FONT_PATH, 80)
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((1080 - w) / 2, (1920 - h) / 2), text, font=font, fill=(0, 0, 0))
    base.save(output_path)

def create_thumbnail(text, font_path, out_path):
    base = Image.new("RGB", (1280, 720), color=(255, 255, 255))
    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(font_path, 72)
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((1280 - w) / 2, (720 - h) / 2), text, font=font, fill=(0, 0, 0))
    base.save(out_path)

def process_story(file_path):
    title = Path(file_path).stem
    out_path = os.path.join(OUTPUT_DIR, title)
    image_dir = os.path.join(out_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    sections = read_story_sections(file_path)
    scene_count = len([key for key in SECTION_KEYS if key in sections and sections[key].strip()])
    print(f"\n📸 ระบบจะใช้ทั้งหมด {scene_count} scene (ภาพประกอบ)")

    print("\n⏸ รอให้พิมพ์ go ก่อนเริ่มสร้างเสียงและวิดีโอ...")
    while input(">> ").strip().lower() != "go":
        print("⏳ รอคำสั่ง go อยู่จ้า...")

    print("\n📢 กำลังสร้างเสียงแต่ละช่วง...")
    voice_paths = []
    for i, key in enumerate(sections.keys()):
        text = sections[key]
        tts = gTTS(text=text, lang='th')
        voice_path = os.path.join(out_path, f"voice_{i+1}.mp3")
        tts.save(voice_path)
        voice_paths.append(voice_path)

    print("🎵 กำลังรวมเสียงกับเพลงพื้นหลัง และ normalize format...")
    audio_paths = []
    for i, voice_path in enumerate(voice_paths):
        voice = effects.normalize(AudioSegment.from_mp3(voice_path)).set_frame_rate(44100).set_channels(2)
        chosen_music = random.choice([os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")])
        bg = AudioSegment.from_mp3(chosen_music)[:len(voice)]
        final_audio = voice.overlay(bg - 15)
        audio_path = os.path.join(out_path, f"narration_with_bg_{i+1}.mp3")
        final_audio.export(audio_path, format="mp3")
        audio_paths.append(audio_path)

    print("📝 กำลังสร้าง subtitle...")
    generate_subtitles(sections, audio_paths, out_path)

    print("🎞️ กำลังรวมวิดีโอจากภาพ...")
    clips = []

    # สร้าง title frame clip
    title_image_path = os.path.join(image_dir, "scene_0.png")
    if "Name" in sections:
        create_title_frame(sections["Name"], title_image_path)
        title_clip = ImageClip(title_image_path).set_duration(TITLE_FRAME_DURATION).resize(height=1920).set_position("center")
        clips.append(title_clip.crossfadein(1).crossfadeout(1))

    for i, audio_path in enumerate(audio_paths):
        img_path = None
        for ext in ['.png', '.jpg']:
            tentative = os.path.join(image_dir, f"scene_{i+1}{ext}")
            if os.path.exists(tentative):
                img_path = tentative
                break
        if not img_path:
            print(f"❌ ไม่พบภาพ: scene_{i+1}")
            continue
        audio = AudioFileClip(audio_path)
        clip = ImageClip(img_path).set_duration(audio.duration).resize(height=1920).set_position("center").set_audio(audio)
        clips.append(clip.crossfadein(1).crossfadeout(1))

    if not clips:
        print("❌ ไม่มีคลิปใด ๆ ให้รวมวิดีโอ")
        return

    final = concatenate_videoclips(clips, method="compose")
    final_path = os.path.join(out_path, "final_video.mp4")
    final.write_videofile(final_path, fps=24)

    print("🖼 กำลังสร้าง thumbnail...")
    thumbnail_path = os.path.join(out_path, "thumbnail.png")
    create_thumbnail(title, FONT_PATH, thumbnail_path)

    print(f"✅ เสร็จสมบูรณ์! ไฟล์ทั้งหมดอยู่ใน {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("⚠️ กรุณาระบุไฟล์นิทาน เช่น: python main_with_format_title_fade.py story/ชื่อเรื่อง.txt")
    else:
        process_story(sys.argv[1])
