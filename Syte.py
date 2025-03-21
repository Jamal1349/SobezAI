from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib.pyplot as plt
from for_test_opt import video_proc
import numpy as np
import matplotlib
from video import video_bp
from moviepy import VideoFileClip


matplotlib.use('Agg')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['GRAPH_FOLDER'] = './static/graphs'
app.config['AUDIO_PATH'] = 'output.mp3'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)
app.register_blueprint(video_bp, url_prefix='/video')


def extract_audio(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Ошибка при извлечении аудио: {e}")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    audio_path = app.config['AUDIO_PATH']
    if 'video' not in request.files:
        return "Видео не было выбрано. Пожалуйста, выберите видео.", 400
    file = request.files['video']
    if file.filename == '':
        return "Имя файла отсутствует. Пожалуйста, выберите файл.", 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        extract_audio(file_path, audio_path)
        results_emotion, results_gesture, results_nitec = video_proc(file_path, num_processes=4, n=5)
        build_gesture_graph(results_gesture, os.path.join(app.config['GRAPH_FOLDER'], 'gesture_graph.png'))
        build_emotion_graph(results_emotion, os.path.join(app.config['GRAPH_FOLDER'], 'emotion_graph.png'))
        build_nitec_graph(results_nitec, os.path.join(app.config['GRAPH_FOLDER'], 'nitec_graph.png'))

        return redirect(url_for('results'))


@app.route('/results')
def results():
    return render_template('results.html',
                           nitec_graph=os.path.join('static/graphs', 'nitec_graph.png'),
                           emotion_graph=os.path.join('static/graphs', 'emotion_graph.png'),
                           gesture_graph=os.path.join('static/graphs', 'gesture_graph.png'))


def build_gesture_graph(data, save_path):
    time = np.arange(len(data))
    plt.figure(figsize=(10, 5))
    plt.step(time, data, where='mid', label="Жесты", color="#2E8B57", linewidth=2)
    bad_gestures = np.array([i for i, val in enumerate(data) if val == 0])
    if bad_gestures.size > 0:
        plt.scatter(bad_gestures, np.zeros_like(bad_gestures), color="red", label="Плохой жест", s=30, zorder=3)
    plt.xlabel("Время (кадры)")
    plt.ylabel("Жест")
    plt.yticks([0, 1], ["Плохой", "Хороший"])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path)
    plt.close()


def build_nitec_graph(data, save_path):
    time = np.arange(len(data))
    plt.figure(figsize=(10, 5))
    plt.step(time, data, where='mid', label="Зрительный контакт", color="#2E8B57", linewidth=2)
    no_contact = np.array([i for i, val in enumerate(data) if val == 0])
    if no_contact.size > 0:
        plt.scatter(no_contact, np.zeros_like(no_contact), color="red", label="Нет контакта", s=30, zorder=3)
    plt.xlabel("Время (кадры)")
    plt.ylabel("Контакт")
    plt.yticks([0, 1], ["Нет", "Есть"])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path)
    plt.close()


def build_emotion_graph(data, save_path):
    data = [emotion for emotion in data if emotion is not None]
    if not data:
        data = ["neutral"]

    emotions, counts = np.unique(data, return_counts=True)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(emotions))))
    plt.title("Распределение эмоций")
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
