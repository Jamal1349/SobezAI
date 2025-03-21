from flask import Blueprint, render_template, Response, redirect, url_for, request, jsonify
import cv2
import json
import os
import threading
import pyaudio
import wave

video_bp = Blueprint('video', __name__)
is_recording = False
video_writer = None
audio_frames = []
audio_thread = None
QUESTIONS_FILE = "questions.json"
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
AUDIO_FILE = "output.wav"
VIDEO_FILE = "output.avi"


def load_questions():
    """Загрузка вопросов из JSON-файла."""
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def record_audio():
    """Функция записи аудио (сохранение в WAV)."""
    global is_recording, audio_frames
    audio_frames = []

    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    while is_recording:
        data = stream.read(CHUNK)
        audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Сохранение аудио в WAV
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))


def generate_frames():
    """Функция захвата видео с веб-камеры."""
    global is_recording, video_writer
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        if is_recording and video_writer is not None:
            video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    if video_writer is not None:
        video_writer.release()


@video_bp.route('/video_page')
def video_page():
    return render_template('video.html')


@video_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@video_bp.route('/start_interview', methods=['POST'])
def start_interview():
    global is_recording, video_writer, audio_thread

    category = request.form.get('category', 'other')
    questions_dict = load_questions()
    questions = questions_dict.get(category, [])

    if not is_recording:
        is_recording = True

        # Настройка видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, 20.0, (640, 480))

        # Запуск потока записи аудио
        audio_thread = threading.Thread(target=record_audio)
        audio_thread.start()

    return render_template('video.html', questions=questions)


@video_bp.route('/stop_recording')
def stop_recording():
    global is_recording, video_writer, audio_thread

    if is_recording:
        is_recording = False

        if video_writer:
            video_writer.release()
            video_writer = None

        if audio_thread:
            audio_thread.join()  # Дождаться завершения записи аудио

    return redirect(url_for('video.video_page'))


@video_bp.route('/get_questions', methods=['GET'])
def get_questions():
    return jsonify(load_questions())